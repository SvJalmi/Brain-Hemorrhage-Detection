import csv
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# ============================================================
# MBH-Seg25 — SUBTYPE-AWARE U-NET V3
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_CSV = (
    PROJECT_ROOT
    / "preprocessing"
    / "segmentation"
    / "mbh_seg25_split.csv"
)

MODEL_DIR = PROJECT_ROOT / "models" / "segmentation"
RESULT_DIR = PROJECT_ROOT / "results" / "segmentation"

CHECKPOINT_PATH = MODEL_DIR / "small_unet_mbh_seg25_v3_best.pt"
METRICS_PATH = RESULT_DIR / "small_unet_mbh_seg25_v3_validation.csv"

IMAGE_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 30
PATIENCE = 7
LEARNING_RATE = 3e-4
NUM_WORKERS = 0
NUM_CLASSES = 6
SEED = 42

CLASS_NAMES = [
    "background",
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
]

# Higher weights increase the probability of selecting slices
# containing the corresponding subtype.
CLASS_SAMPLING_WEIGHTS = {
    1: 5.0,   # EDH
    2: 1.0,   # IPH
    3: 1.5,   # IVH
    4: 1.5,   # SAH
    5: 1.5,   # SDH
}
BACKGROUND_SLICE_WEIGHT = 0.20

ALPHA = 0.3
BETA = 0.7
GAMMA = 0.75
CE_WEIGHT = 0.25
TV_WEIGHT = 0.75

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_slice(image, label):
    image = np.asarray(image, dtype=np.float32)
    label = np.asarray(label, dtype=np.int64)

    low, high = np.percentile(image, [1, 99])

    if high > low:
        image = (np.clip(image, low, high) - low) / (high - low)
    else:
        image = np.zeros_like(image, dtype=np.float32)

    # Explicit float32 conversion prevents CUDA DoubleTensor errors.
    image_tensor = torch.from_numpy(image.copy()).float()[None, None]
    image_tensor = F.interpolate(
        image_tensor,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )[0]

    label_tensor = torch.from_numpy(label.copy()).long()[None, None]
    label_tensor = F.interpolate(
        label_tensor.float(),
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="nearest",
    )[0, 0].long()

    return image_tensor, label_tensor


# ============================================================
# DATASET
# ============================================================

class MBHSliceDataset(Dataset):
    def __init__(self, split_name):
        self.samples = []
        self.sample_weights = []

        with SPLIT_CSV.open(
            "r", newline="", encoding="utf-8-sig"
        ) as file:
            rows = [
                row for row in csv.DictReader(file)
                if row["split"].strip() == split_name
            ]

        if not rows:
            raise RuntimeError(
                f"No cases found for split '{split_name}'"
            )

        for case_index, row in enumerate(rows, start=1):
            image_path = Path(row["image_path"])
            label_path = Path(row["annot_3_path"])

            if not image_path.exists() or not label_path.exists():
                raise FileNotFoundError(
                    f"Missing image or annotation for {row['case_id']}"
                )

            image = nib.load(str(image_path)).get_fdata(
                dtype=np.float32
            )
            label = np.rint(
                nib.load(str(label_path)).get_fdata(
                    dtype=np.float32
                )
            ).astype(np.int64)

            if image.shape != label.shape:
                raise ValueError(
                    f"Image/label shape mismatch: {row['case_id']}"
                )

            if np.any((label < 0) | (label >= NUM_CLASSES)):
                raise ValueError(
                    f"Invalid label values in {label_path}"
                )

            for z in range(image.shape[2]):
                slice_label = label[:, :, z]

                present_classes = [
                    class_id
                    for class_id in range(1, NUM_CLASSES)
                    if np.any(slice_label == class_id)
                ]

                if present_classes:
                    weight = max(
                        CLASS_SAMPLING_WEIGHTS[class_id]
                        for class_id in present_classes
                    )
                else:
                    weight = BACKGROUND_SLICE_WEIGHT

                self.samples.append(
                    (str(image_path), str(label_path), z)
                )
                self.sample_weights.append(float(weight))

            print(
                f"[{case_index}/{len(rows)}] "
                f"{row['case_id']} indexed"
            )

        self.sample_weights = np.asarray(
            self.sample_weights, dtype=np.float64
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label_path, z = self.samples[index]

        image = nib.load(image_path).get_fdata(
            dtype=np.float32
        )[:, :, z]

        label = np.rint(
            nib.load(label_path).get_fdata(
                dtype=np.float32
            )[:, :, z]
        ).astype(np.int64)

        return preprocess_slice(image, label)


# ============================================================
# U-NET MODEL
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
        bottleneck = self.bottleneck(self.pool(e3))

        d3 = self.dec3(
            torch.cat([self.up3(bottleneck), e3], dim=1)
        )
        d2 = self.dec2(
            torch.cat([self.up2(d3), e2], dim=1)
        )
        d1 = self.dec1(
            torch.cat([self.up1(d2), e1], dim=1)
        )

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
            total_loss += (
                1 - tversky[valid]
            ).pow(GAMMA).mean()
            valid_classes += 1

    if valid_classes == 0:
        return logits.sum() * 0.0

    return total_loss / valid_classes


def combined_loss(logits, targets):
    ce = F.cross_entropy(logits, targets)
    tv = focal_tversky_loss(logits, targets)
    return CE_WEIGHT * ce + TV_WEIGHT * tv


# ============================================================
# FULL VALIDATION — EVERY VALIDATION SLICE
# ============================================================

def validate(model, loader):
    model.eval()

    loss_sum = 0.0
    slice_count = 0
    counts = np.zeros((NUM_CLASSES, 3), dtype=np.int64)

    with torch.inference_mode():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = combined_loss(logits, masks)
            predictions = logits.argmax(dim=1)

            loss_sum += loss.item() * images.size(0)
            slice_count += images.size(0)

            for class_id in range(NUM_CLASSES):
                truth = masks == class_id
                guess = predictions == class_id

                counts[class_id, 0] += (
                    torch.logical_and(truth, guess).sum().item()
                )
                counts[class_id, 1] += (
                    torch.logical_and(~truth, guess).sum().item()
                )
                counts[class_id, 2] += (
                    torch.logical_and(truth, ~guess).sum().item()
                )

    class_metrics = []

    for class_id in range(1, NUM_CLASSES):
        tp, fp, fn = counts[class_id]

        dice_denominator = 2 * tp + fp + fn
        dice = (
            2 * tp / dice_denominator
            if dice_denominator else 0.0
        )
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0

        class_metrics.append({
            "class": CLASS_NAMES[class_id],
            "dice": float(dice),
            "sensitivity": float(sensitivity),
            "precision": float(precision),
        })

    mean_fg_dice = float(
        np.mean([item["dice"] for item in class_metrics])
    )

    return {
        "val_loss": loss_sum / max(slice_count, 1),
        "validation_slices": slice_count,
        "mean_fg_dice": mean_fg_dice,
        "class_metrics": class_metrics,
    }


# ============================================================
# TRAINING
# ============================================================

def main():
    print("=" * 65)
    print("MBH-Seg25 — SUBTYPE-AWARE U-NET V3")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("=" * 65)

    train_dataset = MBHSliceDataset("train")
    val_dataset = MBHSliceDataset("validation")

    if len(train_dataset.sample_weights) != len(train_dataset):
        raise RuntimeError("Training sampler weight count mismatch.")

    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(SEED)

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(
            train_dataset.sample_weights,
            dtype=torch.double,
        ),
        num_samples=len(train_dataset),
        replacement=True,
        generator=sampler_generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    # No sampler or filtering: every validation slice appears once.
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Training slices indexed: {len(train_dataset)}")
    print(f"Validation slices indexed: {len(val_dataset)}")
    print("Validation uses every slice exactly once.")

    model = SmallUNet().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    best_dice = -1.0
    epochs_without_improvement = 0
    history = []

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        seen_slices = 0

        for images, masks in train_loader:
            images = images.to(
                device, non_blocking=(device.type == "cuda")
            )
            masks = masks.to(
                device, non_blocking=(device.type == "cuda")
            )

            # Guard against future data-type mismatches.
            images = images.float()
            masks = masks.long()

            optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = combined_loss(logits, masks)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss at epoch {epoch}"
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            optimizer.step()

            train_loss_sum += loss.item() * images.size(0)
            seen_slices += images.size(0)

        train_loss = train_loss_sum / max(seen_slices, 1)
        metrics = validate(model, val_loader)
        scheduler.step(metrics["mean_fg_dice"])

        print(f"\nEpoch {epoch}/{EPOCHS}")
        print(f"Training loss: {train_loss:.4f}")
        print(f"Full validation loss: {metrics['val_loss']:.4f}")
        print(f"Validation slices: {metrics['validation_slices']}")
        print(f"Mean hemorrhage Dice: {metrics['mean_fg_dice']:.4f}")

        for item in metrics["class_metrics"]:
            print(
                f"  {item['class']}: "
                f"Dice={item['dice']:.4f}, "
                f"Sensitivity={item['sensitivity']:.4f}, "
                f"Precision={item['precision']:.4f}"
            )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": metrics["val_loss"],
            "validation_slices": metrics["validation_slices"],
            "mean_fg_dice": metrics["mean_fg_dice"],
        }

        for item in metrics["class_metrics"]:
            name = item["class"]
            record[f"{name}_dice"] = item["dice"]
            record[f"{name}_sensitivity"] = item["sensitivity"]
            record[f"{name}_precision"] = item["precision"]

        history.append(record)

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
                "class_names": CLASS_NAMES,
                "image_size": IMAGE_SIZE,
                "alpha": ALPHA,
                "beta": BETA,
                "gamma": GAMMA,
                "class_sampling_weights": CLASS_SAMPLING_WEIGHTS,
                "background_slice_weight": BACKGROUND_SLICE_WEIGHT,
                "seed": SEED,
            }, CHECKPOINT_PATH)

            print(f"Saved best V3 checkpoint: {CHECKPOINT_PATH}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping after epoch {epoch}.")
            break

    if history:
        with METRICS_PATH.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=list(history[0].keys())
            )
            writer.writeheader()
            writer.writerows(history)

    print("\nV3 training complete.")
    print(f"Best full-validation mean Dice: {best_dice:.4f}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Training history: {METRICS_PATH}")
    print("The held-out test set was not used.")


if __name__ == "__main__":
    main()