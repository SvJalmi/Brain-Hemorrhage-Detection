
from pathlib import Path
import random

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "datasets" / "MBH-Seg25"
SPLIT_CSV = ROOT / "preprocessing" / "segmentation" / "mbh_seg25_split.csv"

IMAGE_SIZE = 256
NUM_CLASSES = 6
STEPS = 200
LEARNING_RATE = 1e-3
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.set_num_threads(4)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable.")

DEVICE = torch.device("cuda")


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
    def __init__(self):
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
        self.final = nn.Conv2d(16, NUM_CLASSES, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)


def load_slice():
    df = pd.read_csv(SPLIT_CSV)
    train_df = df[df["split"].astype(str).str.lower() == "train"]

    for case_id in train_df["case_id"].astype(str):
        case_dir = DATA_ROOT / case_id
        image_path = case_dir / "image.nii.gz"
        label_path = case_dir / "label_annot_3.nii.gz"

        if not image_path.exists() or not label_path.exists():
            continue

        image = nib.load(str(image_path)).get_fdata(dtype=np.float32)
        label_data = nib.load(str(label_path)).get_fdata(dtype=np.float32)

        if image.shape != label_data.shape:
            continue

        label = np.rint(label_data).astype(np.int64)
        counts = np.count_nonzero(label, axis=(0, 1))
        z = int(np.argmax(counts))

        if counts[z] == 0:
            continue

        image_slice = image[:, :, z]
        label_slice = label[:, :, z]

        lo, hi = np.percentile(image_slice, [1, 99])
        if hi > lo:
            image_slice = np.clip(image_slice, lo, hi)
            image_slice = (image_slice - lo) / (hi - lo)
        else:
            image_slice = np.zeros_like(image_slice, dtype=np.float32)

        x = torch.from_numpy(np.ascontiguousarray(image_slice)).float()[None, None]
        y = torch.from_numpy(np.ascontiguousarray(label_slice)).long()[None, None]

        x = F.interpolate(x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        y = F.interpolate(y.float(), size=(IMAGE_SIZE, IMAGE_SIZE), mode="nearest").long().squeeze(1)

        if not torch.all((y >= 0) & (y < NUM_CLASSES)):
            raise ValueError(f"Invalid label values in {case_id}")

        if (y > 0).sum().item() == 0:
            continue

        return case_id, z, x.to(DEVICE), y.to(DEVICE)

    raise RuntimeError("No foreground-containing training slice found.")


def main():
    print("=" * 78)
    print("WEIGHTED CROSS-ENTROPY — FRESH U-NET, ONE TRAINING SLICE")
    print("No checkpoint is loaded or saved.")
    print("=" * 78)
    print("GPU:", torch.cuda.get_device_name(0))

    case_id, z, x, y = load_slice()
    target = y[0]
    present_classes = torch.unique(target).tolist()

    # Weight only classes actually present. The rare foreground class
    # receives a moderate, capped weight instead of an extreme inverse frequency.
    counts = torch.bincount(target.reshape(-1), minlength=NUM_CLASSES).float()
    weights = torch.ones(NUM_CLASSES, device=DEVICE)
    foreground_present = [c for c in present_classes if c != 0]

    for c in foreground_present:
        weights[c] = min(20.0, max(1.0, (counts[0] / counts[c]).sqrt().item()))

    # Normalize weights to keep the average class weight near 1.
    weights = weights / weights.mean()

    model = SmallUNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Case:", case_id, "| Slice:", z)
    print("Pixel counts:", counts.long().tolist())
    print("Present classes:", present_classes)
    print("Class weights:", [round(v, 3) for v in weights.tolist()])
    print("Steps:", STEPS, "| Learning rate:", LEARNING_RATE)
    print()
    print("Step | Weighted CE | Pred FG | Recall | Foreground Dice")
    print("-" * 64)

    report_steps = {0, 1, 5, 10, 20, 40, 60, 80, 100, 150, 200}

    for step in range(STEPS + 1):
        model.eval()
        with torch.no_grad():
            logits = model(x)
            prediction = logits.argmax(dim=1)
            pred_fg = prediction > 0
            target_fg = y > 0

            recall = (
                (pred_fg & target_fg).sum().float()
                / target_fg.sum().clamp_min(1)
            ).item()

            dice = (
                2 * (pred_fg & target_fg).sum().float()
                / (pred_fg.sum() + target_fg.sum()).clamp_min(1)
            ).item()

        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y, weight=weights)
        loss.backward()
        optimizer.step()

        if step in report_steps:
            print(
                f"{step:4d} | {loss.item():11.5f} | "
                f"{int(pred_fg.sum().item()):7d} | "
                f"{recall:6.4f} | {dice:15.4f}"
            )

    print()
    print("Diagnostic complete. No model checkpoint was saved.")


if __name__ == "__main__":
    main()
