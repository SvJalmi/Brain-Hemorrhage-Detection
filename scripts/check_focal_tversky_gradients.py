
from pathlib import Path
import csv

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "MBH-Seg25"
SPLIT = ROOT / "preprocessing" / "segmentation" / "mbh_seg25_split.csv"
CHECKPOINT = (
    ROOT / "models" / "segmentation"
    / "small_unet_mbh_seg25_focal_tversky_best.pt"
)

SIZE = 256
CLASSES = 6
ALPHA, BETA, GAMMA = 0.3, 0.7, 0.75


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
        self.final = nn.Conv2d(16, CLASSES, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)


def combined_loss(logits, target):
    ce = F.cross_entropy(logits, target)
    probabilities = torch.softmax(logits, dim=1)

    total = logits.new_tensor(0.0)
    valid_classes = 0

    for class_idx in range(1, CLASSES):
        pred = probabilities[:, class_idx].reshape(
            probabilities.shape[0], -1
        )
        truth = (target == class_idx).float().reshape(
            target.shape[0], -1
        )

        tp = (pred * truth).sum(dim=1)
        fp = (pred * (1 - truth)).sum(dim=1)
        fn = ((1 - pred) * truth).sum(dim=1)

        tv = (tp + 1e-6) / (
            tp + ALPHA * fp + BETA * fn + 1e-6
        )
        valid = truth.sum(dim=1) > 0

        if valid.any():
            total = total + (1 - tv[valid]).pow(GAMMA).mean()
            valid_classes += 1

    focal = total / valid_classes if valid_classes else logits.new_tensor(0.0)
    return 0.5 * ce + 0.5 * focal, ce, focal


def main():
    print("=" * 65)
    print("FOCAL-TVERSKY GRADIENT CHECK — NO TRAINING")
    print("=" * 65)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=True)

    model = SmallUNet().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.train()  # Allows normal training-mode BatchNorm behaviour.
    model.zero_grad(set_to_none=True)

    selected = None
    with open(SPLIT, "r", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] != "train":
                continue
            case = DATA / row["case_id"]
            ip = case / "image.nii.gz"
            lp = case / "label_annot_3.nii.gz"
            if not ip.exists() or not lp.exists():
                continue

            image = nib.load(str(ip)).get_fdata(dtype=np.float32)
            label = np.rint(
                nib.load(str(lp)).get_fdata(dtype=np.float32)
            ).astype(np.int64)

            counts = np.count_nonzero(label > 0, axis=(0, 1))
            z_candidates = np.where(counts > 0)[0]
            if len(z_candidates) == 0:
                continue

            z = int(z_candidates[np.argmax(counts[z_candidates])])
            img_slice = image[:, :, z].copy()
            mask_slice = label[:, :, z]

            lo, hi = np.percentile(img_slice, [1, 99])
            if hi > lo:
                img_slice = np.clip(img_slice, lo, hi)
                img_slice = (img_slice - lo) / (hi - lo)
            else:
                img_slice = np.zeros_like(img_slice, dtype=np.float32)

            x = torch.from_numpy(img_slice).float()[None, None]
            x = F.interpolate(
                x, size=(SIZE, SIZE), mode="bilinear",
                align_corners=False
            )

            y = torch.from_numpy(mask_slice).long()[None, None].float()
            y = F.interpolate(
                y, size=(SIZE, SIZE), mode="nearest"
            )[0, 0].long()[None]

            if (y > 0).any():
                selected = (row["case_id"], z, x.to(device), y.to(device))
                break

    if selected is None:
        raise RuntimeError("No foreground-containing training slice found.")

    case_id, z, x, y = selected
    logits = model(x)
    loss, ce, focal = combined_loss(logits, y)

    print(f"Device: {device}")
    print(f"Checkpoint epoch: {ckpt.get('epoch', 'unknown')}")
    print(f"Case: {case_id}; slice: {z}")
    print(f"Foreground pixels after resize: {int((y > 0).sum())}")
    print(f"Cross-entropy: {ce.item():.8f}")
    print(f"Focal-Tversky: {focal.item():.8f}")
    print(f"Combined loss: {loss.item():.8f}")

    loss.backward()

    grad = model.final.weight.grad
    if grad is None:
        print("Output-layer gradient: MISSING")
    else:
        print(f"Output-layer gradient finite: {bool(torch.isfinite(grad).all())}")
        print(f"Output-layer gradient L1 mean: {grad.abs().mean().item():.10e}")
        print(f"Output-layer gradient L2 norm: {grad.norm().item():.10e}")
        print(f"Output-layer gradient nonzero values: {int((grad != 0).sum())}")

    total_grad_sq = 0.0
    bad_parameters = []
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                bad_parameters.append(name)
            total_grad_sq += parameter.grad.detach().float().pow(2).sum().item()

    print(f"Total model gradient L2 norm: {total_grad_sq ** 0.5:.10e}")
    print(f"Parameters with non-finite gradients: {bad_parameters}")
    print("No optimizer was created and no weights were updated.")


if __name__ == "__main__":
    main()
