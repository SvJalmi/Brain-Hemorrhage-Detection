
from pathlib import Path
import csv
import random

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_CSV = PROJECT_ROOT / "preprocessing" / "segmentation" / "mbh_seg25_split.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "segmentation"
CHECKPOINT_IN = MODEL_DIR / "small_unet_mbh_seg25_v2_best.pt"
CHECKPOINT_OUT = MODEL_DIR / "small_unet_mbh_seg25_v2_resumed_best.pt"
METRICS_OUT = PROJECT_ROOT / "results" / "segmentation" / "small_unet_mbh_seg25_v2_resumed.csv"

IMAGE_SIZE = 256
BATCH_SIZE = 4
NUM_CLASSES = 6
NUM_WORKERS = 0
START_EPOCH = 27
END_EPOCH = 30
LEARNING_RATE = 3e-4
ALPHA, BETA, GAMMA = 0.3, 0.7, 0.75
CE_WEIGHT, TV_WEIGHT = 0.25, 0.75
PATIENCE = 7
CLASS_NAMES = [
    "background", "epidural", "intraparenchymal",
    "intraventricular", "subarachnoid", "subdural"
]

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
                raise FileNotFoundError(f"Missing data for {row['case_id']}")

            image = nib.load(str(image_path)).get_fdata(dtype=np.float32)
            label = np.rint(
                nib.load(str(label_path)).get_fdata(dtype=np.float32)
            ).astype(np.int64)
            if image.shape != label.shape:
                raise ValueError(f"Shape mismatch: {row['case_id']}")

            fg = [z for z in range(image.shape[2]) if np.any(label[:, :, z] > 0)]
            bg = [z for z in range(image.shape[2]) if not np.any(label[:, :, z] > 0)]
            if fg and bg:
                n_bg = min(
                    len(bg),
                    max(1, int(len(fg) * (1 - foreground_probability) / foreground_probability))
                )
                selected = fg + random.sample(bg, n_bg)
            else:
                selected = fg or bg

            self.samples.extend((str(image_path), str(label_path), z) for z in selected)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label_path, z = self.samples[index]
        image = nib.load(image_path).get_fdata(dtype=np.float32)[:, :, z]
        label = np.rint(
            nib.load(label_path).get_fdata(dtype=np.float32)[:, :, z]
        ).astype(np.int64)

        if np.any((label < 0) | (label >= NUM_CLASSES)):
            raise ValueError(f"Unexpected labels: {np.unique(label)}")

        low, high = np.percentile(image, [1, 99])
        if high > low:
            image = (np.clip(image, low, high) - low) / (high - low)
        else:
            image = np.zeros_like(image, dtype=np.float32)

        x = torch.from_numpy(image.copy()).float()[None, None]
        x = F.interpolate(
            x, size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear", align_corners=False
        )[0].float()

        y = torch.from_numpy(label.copy()).float()[None, None]
        y = F.interpolate(
            y, size=(IMAGE_SIZE, IMAGE_SIZE), mode="nearest"
        )[0, 0].long()
        return x, y


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


def focal_tversky_loss(logits, targets):
    probs = torch.softmax(logits, dim=1)
    total = logits.new_tensor(0.0)
    valid_classes = 0

    for c in range(1, NUM_CLASSES):
        pred = probs[:, c].flatten(1)
        truth = (targets == c).float().flatten(1)
        tp = (pred * truth).sum(dim=1)
        fp = (pred * (1 - truth)).sum(dim=1)
        fn = ((1 - pred) * truth).sum(dim=1)
        tv = (tp + 1e-6) / (tp + ALPHA * fp + BETA * fn + 1e-6)
        valid = truth.sum(dim=1) > 0
        if valid.any():
            total = total + (1 - tv[valid]).pow(GAMMA).mean()
            valid_classes += 1

    return total / valid_classes if valid_classes else logits.sum() * 0.0


def combined_loss(logits, targets):
    return CE_WEIGHT * F.cross_entropy(logits, targets) + TV_WEIGHT * focal_tversky_loss(logits, targets)


def validate(model, loader):
    model.eval()
    loss_sum = 0.0
    samples = 0
    counts = np.zeros((NUM_CLASSES, 3), dtype=np.int64)
    ref_fg = pred_fg = 0

    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = combined_loss(logits, y)
            pred = logits.argmax(dim=1)
            loss_sum += loss.item() * x.size(0)
            samples += x.size(0)
            ref_fg += int((y > 0).sum().item())
            pred_fg += int((pred > 0).sum().item())

            for c in range(NUM_CLASSES):
                truth, guess = y == c, pred == c
                counts[c, 0] += torch.logical_and(truth, guess).sum().item()
                counts[c, 1] += torch.logical_and(~truth, guess).sum().item()
                counts[c, 2] += torch.logical_and(truth, ~guess).sum().item()

    dices, sensitivities = [], []
    for c in range(1, NUM_CLASSES):
        tp, fp, fn = counts[c]
        dices.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
        sensitivities.append(tp / (tp + fn) if (tp + fn) else 0.0)

    return {
        "val_loss": loss_sum / max(samples, 1),
        "mean_fg_dice": float(np.mean(dices)),
        "mean_fg_sensitivity": float(np.mean(sensitivities)),
        "reference_fg_pixels": ref_fg,
        "predicted_fg_pixels": pred_fg,
        "class_dice": dices,
        "class_sensitivity": sensitivities,
    }


def main():
    if not CHECKPOINT_IN.exists():
        raise FileNotFoundError(CHECKPOINT_IN)

    checkpoint = torch.load(CHECKPOINT_IN, map_location=device, weights_only=True)
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    if start_epoch > END_EPOCH:
        raise RuntimeError(
            f"Checkpoint epoch is {start_epoch - 1}; no epochs remain up to {END_EPOCH}. "
            "Do not restart training from scratch."
        )

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint epoch: {start_epoch - 1}")
    print(f"Will run epochs {start_epoch} through {END_EPOCH}")

    train_ds = MBHSliceDataset("train", foreground_probability=0.75)
    val_ds = MBHSliceDataset("validation", foreground_probability=1.0)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    model = SmallUNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
    )
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Use a new scheduler for the remaining epochs; its prior state was not saved.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
    )

    best_dice = float(checkpoint.get("mean_fg_dice", -1.0))
    history = []

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, END_EPOCH + 1):
        model.train()
        loss_sum = 0.0
        samples = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = combined_loss(logits, y)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * x.size(0)
            samples += x.size(0)

        train_loss = loss_sum / max(samples, 1)
        metrics = validate(model, val_loader)
        scheduler.step(metrics["mean_fg_dice"])

        print(f"\nEpoch {epoch}/{END_EPOCH}")
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
            }, CHECKPOINT_OUT)
            print(f"Saved improved resumed checkpoint: {CHECKPOINT_OUT}")

    if history:
        with open(METRICS_OUT, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    print("\nResume run complete.")
    print(f"Best Dice seen (including original checkpoint): {best_dice:.4f}")
    print(f"New checkpoint, if improved: {CHECKPOINT_OUT}")
    print(f"Resume metrics: {METRICS_OUT}")
    print("The held-out test set was not used.")


if __name__ == "__main__":
    main()
