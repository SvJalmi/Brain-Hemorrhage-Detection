
from pathlib import Path
import csv
import random

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# MBH-Seg25 — U-NET V2
# Foreground-aware training and validation Dice monitoring
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = PROJECT_ROOT / "datasets" / "MBH-Seg25"
SPLIT_CSV = (
    PROJECT_ROOT / "preprocessing" / "segmentation"
    / "mbh_seg25_split.csv"
)
MODEL_DIR = PROJECT_ROOT / "models" / "segmentation"

CHECKPOINT_PATH = (
    MODEL_DIR / "small_unet_mbh_seg25_v2_best.pt"
)
METRICS_PATH = (
    PROJECT_ROOT / "results" / "segmentation"
    / "small_unet_mbh_seg25_v2_validation.csv"
)

IMAGE_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 30
PATIENCE = 7
LEARNING_RATE = 3e-4
NUM_WORKERS = 0
NUM_CLASSES = 6
SEED = 42

# Class order: background, EDH, IPH, IVH, SAH, SDH
CLASS_NAMES = [
    "background", "epidural", "intraparenchymal",
    "intraventricular", "subarachnoid", "subdural"
]

# Focal Tversky: greater penalty for false negatives.
ALPHA = 0.3
BETA = 0.7
GAMMA = 0.75

# Give cross-entropy less influence than in the previous run.
CE_WEIGHT = 0.25
TV_WEIGHT = 0.75

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# DATASET
# ============================================================

class MBHSliceDataset(Dataset):
    def __init__(self, split_name, foreground_probability=0.75):
        self.samples = []

        with open(SPLIT_CSV, "r", newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))

        rows = [r for r in rows if r["split"].strip() == split_name]

        for row in rows:
            image_path = Path(row["image_path"])
            label_path = Path(row["annot_3_path"])

            if not image_path.exists() or not label_path.exists():
                raise FileNotFoundError(
                    f"Missing image or annotation for {row['case_id']}"
                )

            image = nib.load(str(image_path)).get_fdata(dtype=np.float32)
            label = np.rint(
                nib.load(str(label_path)).get_fdata(dtype=np.float32)
            ).astype(np.int64)

            if image.shape != label.shape:
                raise ValueError(f"Image/label shape mismatch: {row['case_id']}")

            foreground_slices = [
                z for z in range(image.shape[2])
                if np.any(label[:, :, z] > 0)
            ]
            background_slices = [
                z for z in range(image.shape[2])
                if not np.any(label[:, :, z] > 0)
            ]

            # Keep all foreground slices and sample background slices.
            if foreground_slices and background_slices:
                n_background = int(
                    len(foreground_slices)
                    * (1 - foreground_probability)
                    / foreground_probability
                )
                n_background = min(
                    len(background_slices),
                    max(1, n_background),
                )
                selected = foreground_slices + random.sample(
                    background_slices, n_background
                )
            else:
                selected = foreground_slices or background_slices

            self.samples.extend(
                (str(image_path), str(label_path), z)
                for z in selected
            )

        if not self.samples:
            raise RuntimeError(f"No slices found for split '{split_name}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label_path, z = self.samples[index]

        image = nib.load(image_path).get_fdata(dtype=np.float32)[:, :, z]
        label = np.rint(
            nib.load(label_path).get_fdata(dtype=np.float32)[:, :, z]
        ).astype(np.int64)

        if np.any((label < 0) | (label >= NUM_CLASSES)):
            raise ValueError(f"Invalid label values: {np.unique(label)}")

        low, high = np.percentile(image, [1, 99])
        if high > low:
            image = (np.clip(image, low, high) - low) / (high - low)
        else:
            image = np.zeros_like(image, dtype=np.float32)

        image_tensor = torch.from_numpy(image.copy()).float()[None, None]
        image_tensor = F.interpolate(
            image_tensor,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )[0]

        label_tensor = torch.from_numpy(label.copy()).float()[None, None]
        label_tensor = F.interpolate(
            label_tensor,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="nearest",
        )[0, 0].long()

        return image_tensor, label_tensor


# ============================================================
# MODEL — same Small U-Net architecture
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


# ============================================================
# LOSS
# ============================================================

def focal_tversky_loss(logits, targets):
    probabilities = torch.softmax(logits, dim=1)
    total_loss = logits.new_tensor(0.0)
    valid_classes = 0

    for class_id in range(1, NUM_CLASSES):
        prediction = probabilities[:, class_id].flatten(1)
        target = (targets == class_id).float().flatten(1)

        tp = (prediction * target).sum(dim=1)
        fp = (prediction * (1 - target)).sum(dim=1)
        fn = ((1 - prediction) * target).sum(dim=1)

        tversky = (tp + 1e-6) / (
            tp + ALPHA * fp + BETA * fn + 1e-6
        )

        valid = target.sum(dim=1) > 0
        if valid.any():
            total_loss = total_loss + (1 - tversky[valid]).pow(GAMMA).mean()
            valid_classes += 1

    # If a batch has no foreground labels, return differentiable zero.
    if valid_classes == 0:
        return logits.sum() * 0.0

    return total_loss / valid_classes


def combined_loss(logits, targets):
    ce = F.cross_entropy(logits, targets)
    tv = focal_tversky_loss(logits, targets)
    return CE_WEIGHT * ce + TV_WEIGHT * tv


# ============================================================
# VALIDATION METRICS
# ============================================================

def validate(model, loader):
    model.eval()

    loss_sum = 0.0
    sample_count = 0
    counts = np.zeros((NUM_CLASSES, 3), dtype=np.int64)
    reference_fg = 0
    predicted_fg = 0

    with torch.inference_mode():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = combined_loss(logits, masks)
            predictions = logits.argmax(dim=1)

            loss_sum += loss.item() * images.size(0)
            sample_count += images.size(0)

            reference_fg += int((masks > 0).sum().item())
            predicted_fg += int((predictions > 0).sum().item())

            for c in range(NUM_CLASSES):
                truth = masks == c
                guess = predictions == c
                counts[c, 0] += torch.logical_and(truth, guess).sum().item()
                counts[c, 1] += torch.logical_and(~truth, guess).sum().item()
                counts[c, 2] += torch.logical_and(truth, ~guess).sum().item()

    class_dice = []
    class_sensitivity = []

    for c in range(1, NUM_CLASSES):
        tp, fp, fn = counts[c]
        denominator = 2 * tp + fp + fn
        target_count = tp + fn

        dice = 2 * tp / denominator if denominator else 0.0
        sensitivity = tp / target_count if target_count else 0.0

        class_dice.append(dice)
        class_sensitivity.append(sensitivity)

    # Primary selection metric: mean Dice over the five hemorrhage classes.
    mean_fg_dice = float(np.mean(class_dice))

    return {
        "val_loss": loss_sum / max(sample_count, 1),
        "mean_fg_dice": mean_fg_dice,
        "mean_fg_sensitivity": float(np.mean(class_sensitivity)),
        "reference_fg_pixels": reference_fg,
        "predicted_fg_pixels": predicted_fg,
        "counts": counts,
        "class_dice": class_dice,
        "class_sensitivity": class_sensitivity,
    }


# ============================================================
# MAIN TRAINING
# ============================================================

def main():
    print("=" * 65)
    print("MBH-Seg25 — U-Net V2")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 65)

    train_dataset = MBHSliceDataset("train", foreground_probability=0.75)

    # Validation includes every slice, not a sampled subset.
    val_dataset = MBHSliceDataset("validation", foreground_probability=1.0)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda")
    )

    print(f"Training slices: {len(train_dataset)}")
    print(f"Validation slices: {len(val_dataset)}")

    model = SmallUNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
    )

    best_dice = -1.0
    epochs_without_improvement = 0
    history = []

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        sample_count = 0

        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = combined_loss(logits, masks)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item() * images.size(0)
            sample_count += images.size(0)

        train_loss = train_loss_sum / max(sample_count, 1)
        metrics = validate(model, val_loader)
        scheduler.step(metrics["mean_fg_dice"])

        print(f"\nEpoch {epoch}/{EPOCHS}")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Validation loss: {metrics['val_loss']:.4f}")
        print(f"Mean foreground Dice: {metrics['mean_fg_dice']:.4f}")
        print(f"Mean foreground sensitivity: {metrics['mean_fg_sensitivity']:.4f}")
        print(f"Reference FG pixels: {metrics['reference_fg_pixels']}")
        print(f"Predicted FG pixels: {metrics['predicted_fg_pixels']}")

        for i, name in enumerate(CLASS_NAMES[1:]):
            print(
                f"  {name}: Dice={metrics['class_dice'][i]:.4f}, "
                f"Sensitivity={metrics['class_sensitivity'][i]:.4f}"
            )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": metrics["val_loss"],
            "mean_fg_dice": metrics["mean_fg_dice"],
            "mean_fg_sensitivity": metrics["mean_fg_sensitivity"],
            "reference_fg_pixels": metrics["reference_fg_pixels"],
            "predicted_fg_pixels": metrics["predicted_fg_pixels"],
        })

        if metrics["mean_fg_dice"] > best_dice:
            best_dice = metrics["mean_fg_dice"]
            epochs_without_improvement = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": metrics["val_loss"],
                "mean_fg_dice": metrics["mean_fg_dice"],
                "mean_fg_sensitivity": metrics["mean_fg_sensitivity"],
                "class_names": CLASS_NAMES,
                "image_size": IMAGE_SIZE,
                "alpha": ALPHA,
                "beta": BETA,
                "gamma": GAMMA,
            }, CHECKPOINT_PATH)

            print(f"Saved best V2 checkpoint: {CHECKPOINT_PATH}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping after {epoch} epochs.")
            break

    with open(METRICS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    print("\nTraining complete.")
    print(f"Best validation foreground Dice: {best_dice:.4f}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Training history: {METRICS_PATH}")
    print("The held-out test set was not used.")


if __name__ == "__main__":
    main()
