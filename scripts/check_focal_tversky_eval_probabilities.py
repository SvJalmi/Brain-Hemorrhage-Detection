
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT = (
    ROOT / "models" / "segmentation"
    / "small_unet_mbh_seg25_focal_tversky_best.pt"
)
SPLIT_CSV = (
    ROOT / "preprocessing" / "segmentation"
    / "mbh_seg25_split.csv"
)
DATA_ROOT = ROOT / "datasets" / "MBH-Seg25"

DEVICE = torch.device("cpu")
IMAGE_SIZE = 256
NUM_CLASSES = 6


# ============================================================
# MODEL — same architecture as training
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 3,
                padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels, 3,
                padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SmallUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=NUM_CLASSES):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(32, 64)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(64, 96)

        self.up3 = nn.ConvTranspose2d(96, 64, 2, stride=2)
        self.dec3 = DoubleConv(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = DoubleConv(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = DoubleConv(32, 16)
        self.final = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.final(d1)


def find_column(df, candidates):
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"Required column not found. Columns: {list(df.columns)}"
    )


def load_checkpoint(model):
    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)


def prepare_slice(image_slice, label_slice):
    # Same robust per-slice normalization used during training.
    lower, upper = np.percentile(image_slice, [1, 99])

    if upper > lower:
        image_slice = np.clip(image_slice, lower, upper)
        image_slice = (image_slice - lower) / (upper - lower)
    else:
        image_slice = np.zeros_like(image_slice, dtype=np.float32)

    image_tensor = torch.from_numpy(
        image_slice.copy()
    ).float()[None, None]

    image_tensor = F.interpolate(
        image_tensor,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )

    label_tensor = torch.from_numpy(
        label_slice.copy()
    ).long()[None, None]

    label_tensor = F.interpolate(
        label_tensor.float(),
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="nearest",
    ).long().squeeze(1)

    return image_tensor, label_tensor


def main():
    torch.set_num_threads(4)

    print("=" * 80)
    print("SAVED CHECKPOINT — EVALUATION-MODE PROBABILITY CHECK")
    print("No training, optimizer, or checkpoint modification.")
    print("=" * 80)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")
    if not SPLIT_CSV.exists():
        raise FileNotFoundError(f"Split CSV not found: {SPLIT_CSV}")

    split_df = pd.read_csv(SPLIT_CSV)
    split_col = find_column(split_df, ["split", "set", "subset"])
    case_col = find_column(
        split_df, ["case_id", "case", "case_name", "id"]
    )

    train_df = split_df[
        split_df[split_col].astype(str).str.lower().eq("train")
    ]

    selected = None

    for _, row in train_df.iterrows():
        case_id = str(row[case_col])
        case_dir = DATA_ROOT / case_id
        image_path = case_dir / "image.nii.gz"
        label_path = case_dir / "label_annot_3.nii.gz"

        if not image_path.exists() or not label_path.exists():
            continue

        image = np.asarray(
            nib.load(str(image_path)).get_fdata(),
            dtype=np.float32,
        )
        label = np.asarray(
            nib.load(str(label_path)).get_fdata(),
            dtype=np.int16,
        )

        if image.shape != label.shape:
            continue

        counts = np.count_nonzero(label, axis=(0, 1))
        slice_idx = int(np.argmax(counts))

        if counts[slice_idx] > 0:
            selected = (
                case_id,
                slice_idx,
                image[:, :, slice_idx],
                label[:, :, slice_idx],
            )
            break

    if selected is None:
        raise RuntimeError(
            "No foreground slice found in the training split."
        )

    case_id, slice_idx, image_slice, label_slice = selected
    image_tensor, label_tensor = prepare_slice(
        image_slice, label_slice
    )

    model = SmallUNet().to(DEVICE)
    load_checkpoint(model)
    model.eval()

    with torch.inference_mode():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted = logits.argmax(dim=1)

        total_fg_prob = probabilities[:, 1:].sum(dim=1)
        target_fg = label_tensor > 0
        target_bg = ~target_fg
        predicted_fg = predicted > 0

        print(f"\nCase: {case_id}")
        print(f"Slice: {slice_idx}")
        print(f"Reference foreground pixels: {int(target_fg.sum())}")
        print(f"Predicted foreground pixels: {int(predicted_fg.sum())}")
        print()

        def report_region(name, region):
            if not region.any():
                print(f"{name}: no pixels in this region")
                return

            mean_fg = total_fg_prob[region].mean().item()
            mean_bg_class = probabilities[:, 0][region].mean().item()
            max_fg_class = probabilities[:, 1:].amax(dim=1)[region].max().item()

            print(
                f"{name}: mean total hemorrhage probability={mean_fg:.6f}, "
                f"mean background probability={mean_bg_class:.6f}, "
                f"maximum hemorrhage-class probability={max_fg_class:.6f}"
            )

        print("Probability comparison:")
        report_region("Labelled lesion pixels", target_fg)
        report_region("Labelled background pixels", target_bg)

        print("\nPer-subtype mean probability on labelled lesion pixels:")
        for class_id, class_name in enumerate(
            [
                "Epidural",
                "Intraparenchymal",
                "Intraventricular",
                "Subarachnoid",
                "Subdural",
            ],
            start=1,
        ):
            class_prob = probabilities[:, class_id][target_fg].mean().item()
            print(f"{class_name}: {class_prob:.6f}")

        print("\nCheckpoint evaluation completed.")
        print("No optimizer steps were performed.")
        print("No model weights or dataset files were changed.")


if __name__ == "__main__":
    main()
