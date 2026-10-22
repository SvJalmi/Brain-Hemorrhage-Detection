
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

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. No experiment was started.")

DEVICE = torch.device("cuda")
torch.set_num_threads(4)


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


def load_one_training_slice():
    df = pd.read_csv(SPLIT_CSV)
    train_df = df[df["split"].astype(str).str.lower() == "train"]

    if train_df.empty:
        raise RuntimeError("No training cases found in the split CSV.")

    for case_id in train_df["case_id"].astype(str):
        case_dir = DATA_ROOT / case_id
        image_path = case_dir / "image.nii.gz"
        label_path = case_dir / "label_annot_3.nii.gz"

        if not image_path.exists() or not label_path.exists():
            continue

        image_obj = nib.load(str(image_path))
        label_obj = nib.load(str(label_path))
        image = image_obj.get_fdata(dtype=np.float32)
        label_float = label_obj.get_fdata(dtype=np.float32)

        if image.shape != label_float.shape:
            continue

        label = np.rint(label_float).astype(np.int64)
        counts = np.count_nonzero(label, axis=(0, 1))
        z = int(np.argmax(counts))

        if counts[z] == 0:
            continue

        image_slice = image[:, :, z]
        label_slice = label[:, :, z]

        lower, upper = np.percentile(image_slice, [1, 99])
        if upper > lower:
            image_slice = np.clip(image_slice, lower, upper)
            image_slice = (image_slice - lower) / (upper - lower)
        else:
            image_slice = np.zeros_like(image_slice, dtype=np.float32)

        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image_slice)
        ).float()[None, None]

        label_tensor = torch.from_numpy(
            np.ascontiguousarray(label_slice)
        ).long()[None, None]

        image_tensor = F.interpolate(
            image_tensor,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )

        label_tensor = F.interpolate(
            label_tensor.float(),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="nearest",
        ).long().squeeze(1)

        if not torch.all((label_tensor >= 0) & (label_tensor < NUM_CLASSES)):
            raise ValueError(f"Unexpected label value in case {case_id}, slice {z}")

        if int((label_tensor > 0).sum()) == 0:
            continue

        return case_id, z, image_tensor.to(DEVICE), label_tensor.to(DEVICE)

    raise RuntimeError("Could not find a valid foreground slice in training cases.")


def main():
    print("=" * 78)
    print("FRESH U-NET SINGLE-SLICE OVERFIT TEST — GPU")
    print("Uses one training slice; no checkpoint is loaded or saved.")
    print("=" * 78)
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM total (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))

    case_id, z, image, target = load_one_training_slice()
    reference_fg = int((target > 0).sum().item())

    model = SmallUNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Case:", case_id)
    print("Slice:", z)
    print("Input shape:", tuple(image.shape))
    print("Target shape:", tuple(target.shape))
    print("Reference foreground pixels:", reference_fg)
    print("Fresh initialization: yes")
    print("Steps:", STEPS, "| Learning rate:", LEARNING_RATE)
    print()
    print("Step | CE loss | Predicted FG | Mean FG probability | FG pixels recall")
    print("-" * 72)

    report_steps = {0, 1, 5, 10, 20, 40, 60, 80, 100, 150, 200}

    for step in range(STEPS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits = model(image)
        loss = F.cross_entropy(logits, target)

        with torch.no_grad():
            model.eval()
            eval_logits = model(image)
            probabilities = torch.softmax(eval_logits, dim=1)
            predicted = eval_logits.argmax(dim=1)
            predicted_fg = int((predicted > 0).sum().item())
            mean_fg_probability = probabilities[:, 1:].sum(dim=1).mean().item()

            target_fg = target > 0
            recall = (
                ((predicted > 0) & target_fg).sum().float()
                / target_fg.sum().clamp_min(1)
            ).item()

        if step in report_steps:
            print(
                f"{step:4d} | {loss.item():8.5f} | "
                f"{predicted_fg:12d} | {mean_fg_probability:18.6f} | "
                f"{recall:.4f}"
            )

        if step == STEPS:
            break

        model.train()
        logits = model(image)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()

    print()
    print("Diagnostic complete.")
    print("Existing model checkpoints were not read, changed, or overwritten.")
    print("No validation or test cases were used.")
    print("This is a pipeline diagnostic, not a final performance evaluation.")


if __name__ == "__main__":
    main()
