

from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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
STEPS = 30
LEARNING_RATE = 1e-4
IMAGE_SIZE = 256
NUM_CLASSES = 6


# ============================================================
# MODEL — same architecture as the training script
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


# ============================================================
# FOCAL-TVERSKY LOSS — matches the training implementation
# ============================================================

def focal_tversky_loss(
    logits,
    targets,
    alpha=0.3,
    beta=0.7,
    gamma=0.75,
    smooth=1e-6,
):
    probabilities = torch.softmax(logits, dim=1)

    total_loss = 0.0
    valid_classes = 0

    for class_idx in range(1, NUM_CLASSES):
        prediction = probabilities[:, class_idx]
        target = (targets == class_idx).float()

        prediction_flat = prediction.reshape(
            prediction.shape[0], -1
        )
        target_flat = target.reshape(
            target.shape[0], -1
        )

        true_positive = (
            prediction_flat * target_flat
        ).sum(dim=1)

        false_positive = (
            prediction_flat * (1.0 - target_flat)
        ).sum(dim=1)

        false_negative = (
            (1.0 - prediction_flat) * target_flat
        ).sum(dim=1)

        tversky = (
            true_positive + smooth
        ) / (
            true_positive
            + alpha * false_positive
            + beta * false_negative
            + smooth
        )

        valid = target_flat.sum(dim=1) > 0

        if valid.any():
            class_loss = (
                1.0 - tversky[valid]
            ).pow(gamma)

            total_loss += class_loss.mean()
            valid_classes += 1

    if valid_classes == 0:
        return torch.tensor(
            0.0,
            device=logits.device,
            requires_grad=True,
        )

    return total_loss / valid_classes


def combined_loss(logits, targets):
    ce = F.cross_entropy(logits, targets)
    ft = focal_tversky_loss(logits, targets)
    return 0.5 * ce + 0.5 * ft, ce.detach(), ft.detach()


# ============================================================
# HELPERS
# ============================================================

def find_column(df, candidates):
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        f"Required column not found. Available columns: {list(df.columns)}"
    )


# ============================================================
# MAIN DIAGNOSTIC
# ============================================================

def main():
    torch.set_num_threads(4)

    print("=" * 100)
    print("FOCAL-TVERSKY SINGLE-SLICE PROBABILITY DIAGNOSTIC")
    print("In-memory model only; no checkpoint will be saved.")
    print("=" * 100)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    if not SPLIT_CSV.exists():
        raise FileNotFoundError(f"Split CSV not found: {SPLIT_CSV}")

    split_df = pd.read_csv(SPLIT_CSV)

    split_col = find_column(
        split_df, ["split", "set", "subset"]
    )
    case_col = find_column(
        split_df, ["case_id", "case", "case_name", "id"]
    )

    train_df = split_df[
        split_df[split_col].astype(str).str.lower().eq("train")
    ]

    if train_df.empty:
        raise RuntimeError("No training cases found in split CSV.")

    # Find a training case with a foreground-containing slice.
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
            "Could not find a foreground slice in the training cases."
        )

    case_id, slice_idx, image_slice, label_slice = selected

    # Match the training script's per-slice intensity normalization.
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

    image_tensor = image_tensor.to(DEVICE)
    label_tensor = label_tensor.to(DEVICE)

    foreground_pixels = int(
        (label_tensor > 0).sum().item()
    )

    if foreground_pixels == 0:
        raise RuntimeError(
            "Foreground labels disappeared after resizing."
        )

    # Load checkpoint without modifying the saved file.
    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    model = SmallUNet().to(DEVICE)

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]
    elif (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print(f"Device: {DEVICE}")
    print(f"Case: {case_id}")
    print(f"Slice: {slice_idx}")
    print(f"Reference foreground pixels: {foreground_pixels}")
    print(f"Optimizer: Adam, learning rate={LEARNING_RATE}")
    print(f"Steps: {STEPS}")
    print()

    print(
        "Step | Combined | CE | Focal-Tversky | Pred FG | "
        "Mean BG prob | Mean FG prob | Max FG prob | Lesion FG prob"
    )
    print("-" * 115)

    report_steps = {0, 1, 2, 5, 10, 15, 20, 25, 30}

    for step in range(STEPS + 1):
        optimizer.zero_grad(set_to_none=True)

        logits = model(image_tensor)
        loss, ce, ft = combined_loss(logits, label_tensor)

        with torch.no_grad():
            probabilities = torch.softmax(logits, dim=1)
            predicted = logits.argmax(dim=1)

            predicted_fg = int(
                (predicted > 0).sum().item()
            )

            mean_bg_prob = (
                probabilities[:, 0].mean().item()
            )

            total_fg_prob = probabilities[:, 1:].sum(dim=1)

            mean_fg_prob = total_fg_prob.mean().item()
            max_fg_prob = probabilities[:, 1:].max().item()

            target_fg = label_tensor > 0

            if target_fg.any():
                lesion_mean_fg_prob = (
                    total_fg_prob[target_fg].mean().item()
                )
            else:
                lesion_mean_fg_prob = 0.0

        if step in report_steps:
            print(
                f"{step:4d} | {loss.item():8.5f} | "
                f"{ce.item():7.5f} | {ft.item():13.6f} | "
                f"{predicted_fg:7d} | "
                f"{mean_bg_prob:12.5f} | "
                f"{mean_fg_prob:12.5f} | "
                f"{max_fg_prob:11.5f} | "
                f"{lesion_mean_fg_prob:14.5f}"
            )

        if step == STEPS:
            break

        loss.backward()
        optimizer.step()

    print()
    print("Diagnostic finished.")
    print("Only the in-memory model was updated.")
    print("The saved checkpoint was not modified.")
    print("No held-out test cases were loaded or evaluated.")


if __name__ == "__main__":
    main()
