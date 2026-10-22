
from pathlib import Path
import csv
import random

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F


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
ALPHA = 0.3
BETA = 0.7
GAMMA = 0.75

CLASS_NAMES = [
    "background",
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
]


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


def focal_tversky_loss(logits, targets):
    probabilities = torch.softmax(logits, dim=1)
    total_loss = 0.0
    valid_classes = 0

    for class_idx in range(1, NUM_CLASSES):
        prediction = probabilities[:, class_idx].reshape(
            probabilities.shape[0], -1
        )
        target = (targets == class_idx).float().reshape(
            targets.shape[0], -1
        )

        tp = (prediction * target).sum(dim=1)
        fp = (prediction * (1.0 - target)).sum(dim=1)
        fn = ((1.0 - prediction) * target).sum(dim=1)

        tversky = (tp + 1e-6) / (
            tp + ALPHA * fp + BETA * fn + 1e-6
        )

        valid = target.sum(dim=1) > 0
        if valid.any():
            total_loss += (1.0 - tversky[valid]).pow(GAMMA).mean()
            valid_classes += 1

    if valid_classes == 0:
        return logits.new_tensor(0.0)

    return total_loss / valid_classes


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


def load_training_cases():
    cases = []
    with open(SPLIT_CSV, "r", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == "train":
                case_dir = DATASET_ROOT / row["case_id"]
                image_path = case_dir / "image.nii.gz"
                label_path = case_dir / "label_annot_3.nii.gz"

                if image_path.exists() and label_path.exists():
                    cases.append((row["case_id"], image_path, label_path))

    return cases


def main():
    print("=" * 70)
    print("FOCAL-TVERSKY LOSS DIAGNOSTIC")
    print("=" * 70)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=True,
    )

    model = SmallUNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Device: {device}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Checkpoint validation loss: {checkpoint.get('val_loss', 'unknown')}")

    cases = load_training_cases()
    if not cases:
        raise RuntimeError("No training cases found.")

    # Reproducible selection; never reads the test split.
    random.seed(42)
    selected = random.sample(cases, min(3, len(cases)))

    all_ce = []
    all_ft = []
    all_combined = []
    all_fg_probability = []
    all_bg_probability = []
    all_predicted_fg = []
    all_target_fg = []

    for case_id, image_path, label_path in selected:
        image = nib.load(str(image_path)).get_fdata(dtype=np.float32)
        label = np.rint(
            nib.load(str(label_path)).get_fdata(dtype=np.float32)
        ).astype(np.int64)

        foreground_counts = np.count_nonzero(label > 0, axis=(0, 1))
        foreground_slices = np.where(foreground_counts > 0)[0]

        if len(foreground_slices):
            z = int(foreground_slices[len(foreground_slices) // 2])
        else:
            z = image.shape[2] // 2

        image_tensor, target = preprocess(image[:, :, z], label[:, :, z])
        image_tensor = image_tensor.to(device)
        target_batch = target.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(image_tensor)
            ce = F.cross_entropy(logits, target_batch)
            ft = focal_tversky_loss(logits, target_batch)
            combined = 0.5 * ce + 0.5 * ft

            probabilities = torch.softmax(logits, dim=1)
            prediction = probabilities.argmax(dim=1)

        target_np = target.numpy()
        prediction_np = prediction[0].cpu().numpy()
        probs = probabilities[0].cpu().numpy()

        all_ce.append(ce.item())
        all_ft.append(ft.item())
        all_combined.append(combined.item())
        all_fg_probability.append(float(probs[1:].sum(axis=0).mean()))
        all_bg_probability.append(float(probs[0].mean()))
        all_predicted_fg.append(int((prediction_np > 0).sum()))
        all_target_fg.append(int((target_np > 0).sum()))

        print(f"\nCase: {case_id} | slice: {z}")
        print(f"Cross-entropy loss: {ce.item():.6f}")
        print(f"Focal-Tversky loss: {ft.item():.6f}")
        print(f"Combined loss:      {combined.item():.6f}")
        print(f"Mean background probability: {probs[0].mean():.6f}")
        print(
            "Mean total foreground probability: "
            f"{probs[1:].sum(axis=0).mean():.6f}"
        )
        print(f"Reference foreground pixels: {(target_np > 0).sum()}")
        print(f"Predicted foreground pixels: {(prediction_np > 0).sum()}")

        print("Per-class target pixel counts:")
        for class_idx, name in enumerate(CLASS_NAMES):
            print(f"  {name:<22} {(target_np == class_idx).sum():>8}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Mean cross-entropy: {np.mean(all_ce):.6f}")
    print(f"Mean Focal-Tversky: {np.mean(all_ft):.6f}")
    print(f"Mean combined loss: {np.mean(all_combined):.6f}")
    print(f"Mean background probability: {np.mean(all_bg_probability):.6f}")
    print(
        "Mean total foreground probability: "
        f"{np.mean(all_fg_probability):.6f}"
    )
    print(f"Total predicted foreground pixels: {sum(all_predicted_fg)}")
    print(f"Total reference foreground pixels: {sum(all_target_fg)}")
    print("\nNo model weights, datasets, or checkpoints were modified.")


if __name__ == "__main__":
    main()
