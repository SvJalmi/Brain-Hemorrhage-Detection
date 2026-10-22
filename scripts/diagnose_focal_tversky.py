
from pathlib import Path
import csv

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = PROJECT_ROOT / "datasets" / "MBH-Seg25"
SPLIT_CSV = (
    PROJECT_ROOT / "preprocessing" / "segmentation"
    / "mbh_seg25_split.csv"
)
CHECKPOINT = (
    PROJECT_ROOT / "models" / "segmentation"
    / "small_unet_mbh_seg25_focal_tversky_best.pt"
)

IMAGE_SIZE = 256
NUM_CLASSES = 6
MAX_CASES_PER_SPLIT = 3

CLASS_NAMES = [
    "background",
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
]


# ============================================================
# SAME MODEL ARCHITECTURE AS TRAINING
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SmallUNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.enc1 = DoubleConv(1, 16)
        self.enc2 = DoubleConv(16, 32)
        self.enc3 = DoubleConv(32, 64)
        self.bottleneck = DoubleConv(64, 96)
        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(96, 64, 2, stride=2)
        self.dec3 = DoubleConv(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = DoubleConv(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = DoubleConv(32, 16)
        self.final = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.final(d1)


# ============================================================
# CASE SPLITS
# ============================================================

def load_cases(split_name):
    cases = []

    with open(SPLIT_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row["split"] == split_name:
                case_dir = DATASET_ROOT / row["case_id"]

                image_path = case_dir / "image.nii.gz"
                label_path = case_dir / "label_annot_3.nii.gz"

                if image_path.exists() and label_path.exists():
                    cases.append(
                        (row["case_id"], image_path, label_path)
                    )

    return cases


# ============================================================
# MATCH TRAINING PREPROCESSING
# ============================================================

def preprocess(image_slice, label_slice):
    image_slice = image_slice.astype(np.float32, copy=True)
    label_slice = np.rint(label_slice).astype(np.int64)

    lower = np.percentile(image_slice, 1)
    upper = np.percentile(image_slice, 99)

    if upper > lower:
        image_slice = np.clip(image_slice, lower, upper)
        image_slice = (image_slice - lower) / (upper - lower)
    else:
        image_slice = np.zeros_like(image_slice, dtype=np.float32)

    image_tensor = torch.from_numpy(image_slice).float()[None, None]
    image_tensor = F.interpolate(
        image_tensor,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )

    label_tensor = torch.from_numpy(label_slice).long()[None, None].float()
    label_tensor = F.interpolate(
        label_tensor,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="nearest",
    )[0, 0].long()

    return image_tensor, label_tensor


# ============================================================
# DIAGNOSIS
# ============================================================

def main():
    print("=" * 72)
    print("FOCAL-TVERSKY U-NET DIAGNOSTICS")
    print("=" * 72)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=True,
    )

    model = SmallUNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Checkpoint validation loss: {checkpoint.get('val_loss', 'unknown')}")

    for split_name in ("train", "validation"):
        cases = load_cases(split_name)

        if not cases:
            raise RuntimeError(f"No cases found for split: {split_name}")

        # Select a few cases only. Test cases are deliberately excluded.
        selected_cases = cases[:MAX_CASES_PER_SPLIT]

        print("\n" + "=" * 72)
        print(f"SPLIT: {split_name.upper()}")
        print(f"Cases inspected: {len(selected_cases)}")
        print("=" * 72)

        for case_id, image_path, label_path in selected_cases:
            image = nib.load(str(image_path)).get_fdata(
                dtype=np.float32
            )
            label = np.rint(
                nib.load(str(label_path)).get_fdata(dtype=np.float32)
            ).astype(np.int64)

            if image.shape != label.shape:
                raise ValueError(f"Shape mismatch in {case_id}")

            # Select a slice with the most annotated foreground voxels.
            foreground_per_slice = np.count_nonzero(
                label > 0, axis=(0, 1)
            )

            if foreground_per_slice.max() > 0:
                z = int(np.argmax(foreground_per_slice))
            else:
                z = image.shape[2] // 2

            raw_slice = label[:, :, z]
            image_tensor, target = preprocess(image[:, :, z], raw_slice)

            # Inspect class presence before and after resizing.
            raw_classes, raw_counts = np.unique(
                raw_slice, return_counts=True
            )
            resized_classes, resized_counts = np.unique(
                target.numpy(), return_counts=True
            )

            print(f"\nCase: {case_id}")
            print(f"Slice index: {z} / {image.shape[2] - 1}")
            print(f"Raw label shape: {raw_slice.shape}")
            print(f"Resized label shape: {tuple(target.shape)}")
            print("Raw label pixel counts:")
            for cls, count in zip(raw_classes, raw_counts):
                name = CLASS_NAMES[int(cls)] if 0 <= cls < NUM_CLASSES else "UNKNOWN"
                print(f"  {name:<22} {int(count):>8}")

            print("Resized label pixel counts:")
            for cls, count in zip(resized_classes, resized_counts):
                name = CLASS_NAMES[int(cls)] if 0 <= cls < NUM_CLASSES else "UNKNOWN"
                print(f"  {name:<22} {int(count):>8}")

            with torch.inference_mode():
                logits = model(image_tensor.to(device))
                probabilities = torch.softmax(logits, dim=1)[0].cpu()
                prediction = probabilities.argmax(dim=0).numpy()

            pred_classes, pred_counts = np.unique(
                prediction, return_counts=True
            )

            print("Predicted pixel counts:")
            for cls, count in zip(pred_classes, pred_counts):
                print(f"  {CLASS_NAMES[int(cls)]:<22} {int(count):>8}")

            print("Mean predicted probability by class:")
            for cls, name in enumerate(CLASS_NAMES):
                mean_prob = probabilities[cls].mean().item()
                max_prob = probabilities[cls].max().item()
                print(
                    f"  {name:<22} mean={mean_prob:.6f} "
                    f"max={max_prob:.6f}"
                )

            foreground_pixels = int(np.count_nonzero(prediction > 0))
            target_foreground_pixels = int(
                np.count_nonzero(target.numpy() > 0)
            )

            print(
                f"Reference foreground pixels after resize: "
                f"{target_foreground_pixels}"
            )
            print(
                f"Predicted foreground pixels: {foreground_pixels}"
            )

    print("\n" + "=" * 72)
    print("DIAGNOSTICS COMPLETE")
    print("Only training and validation cases were inspected.")
    print("=" * 72)


if __name__ == "__main__":
    main()
