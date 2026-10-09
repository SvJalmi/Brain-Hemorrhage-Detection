
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "MBH-Seg25"
SPLIT_CSV = (
    PROJECT_ROOT / "preprocessing" / "segmentation"
    / "mbh_seg25_split.csv"
)
CHECKPOINT = (
    PROJECT_ROOT / "models" / "segmentation"
    / "small_unet_mbh_seg25_focal_tversky_best.pt"
)
OUTPUT_DIR = PROJECT_ROOT / "results" / "segmentation"

IMAGE_SIZE = 256
NUM_CLASSES = 6
BATCH_SIZE = 4

CLASS_NAMES = [
    "background",
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
]


# ============================================================
# MODEL: MUST MATCH THE TRAINING ARCHITECTURE
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
# TEST CASES
# ============================================================

def load_test_cases():
    cases = []
    with open(SPLIT_CSV, "r", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == "test":
                case_dir = DATASET_ROOT / row["case_id"]
                image_path = case_dir / "image.nii.gz"
                label_path = case_dir / "label_annot_3.nii.gz"

                if not image_path.exists() or not label_path.exists():
                    raise FileNotFoundError(
                        f"Missing CT or annot_3 label for {row['case_id']}"
                    )

                cases.append((row["case_id"], image_path, label_path))

    if not cases:
        raise RuntimeError("No test cases found. Check the split CSV.")

    return cases


# ============================================================
# PREPROCESSING: MATCH TRAINING NORMALIZATION
# ============================================================

def preprocess_slice(image_slice, label_slice):
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

    return image_tensor[0], label_tensor


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(tp, fp, fn):
    denom_dice = 2 * tp + fp + fn
    denom_iou = tp + fp + fn

    dice = (2 * tp / denom_dice) if denom_dice else float("nan")
    iou = (tp / denom_iou) if denom_iou else float("nan")
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")

    return dice, iou, sensitivity, precision


# ============================================================
# EVALUATION
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 68)
    print("MBH-Seg25 — FOCAL-TVERSKY TEST EVALUATION")
    print("=" * 68)
    print(f"Device: {device}")

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=True,
    )

    model = SmallUNet(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Checkpoint validation loss: {checkpoint.get('val_loss', 'unknown')}")

    cases = load_test_cases()
    print(f"Test cases: {len(cases)}")
    print("Evaluating full test volumes slice by slice...")

    # Aggregate voxel counts by subtype across all test cases.
    totals = {
        name: {"tp": 0, "fp": 0, "fn": 0, "target_voxels": 0,
               "predicted_voxels": 0}
        for name in CLASS_NAMES
    }

    case_rows = []
    case_foreground_rows = []
    evaluated_slices = 0

    with torch.inference_mode():
        for case_index, (case_id, image_path, label_path) in enumerate(cases, 1):
            image_data = nib.load(str(image_path)).get_fdata(dtype=np.float32)
            label_data = np.rint(
                nib.load(str(label_path)).get_fdata(dtype=np.float32)
            ).astype(np.int64)

            if image_data.shape != label_data.shape:
                raise ValueError(
                    f"Shape mismatch in {case_id}: "
                    f"{image_data.shape} vs {label_data.shape}"
                )

            case_counts = {
                name: {"tp": 0, "fp": 0, "fn": 0,
                       "target_voxels": 0, "predicted_voxels": 0}
                for name in CLASS_NAMES
            }

            for z in range(image_data.shape[2]):
                image_tensor, target = preprocess_slice(
                    image_data[:, :, z], label_data[:, :, z]
                )
                logits = model(image_tensor.unsqueeze(0).to(device))
                prediction = logits.argmax(dim=1)[0].cpu().numpy()
                target = target.numpy()
                evaluated_slices += 1

                for class_idx, name in enumerate(CLASS_NAMES):
                    true_mask = target == class_idx
                    pred_mask = prediction == class_idx

                    tp = int(np.logical_and(true_mask, pred_mask).sum())
                    fp = int(np.logical_and(~true_mask, pred_mask).sum())
                    fn = int(np.logical_and(true_mask, ~pred_mask).sum())

                    for bucket in (totals[name], case_counts[name]):
                        bucket["tp"] += tp
                        bucket["fp"] += fp
                        bucket["fn"] += fn
                        bucket["target_voxels"] += int(true_mask.sum())
                        bucket["predicted_voxels"] += int(pred_mask.sum())

            # Per-case foreground macro Dice; absent reference subtypes
            # are excluded from that case's macro score.
            per_case_dices = []
            for class_idx in range(1, NUM_CLASSES):
                name = CLASS_NAMES[class_idx]
                c = case_counts[name]
                if c["target_voxels"] > 0:
                    dice, iou, sensitivity, precision = calculate_metrics(
                        c["tp"], c["fp"], c["fn"]
                    )
                    per_case_dices.append(dice)

                    case_rows.append({
                        "case_id": case_id,
                        "class": name,
                        "target_voxels": c["target_voxels"],
                        "predicted_voxels": c["predicted_voxels"],
                        "tp": c["tp"],
                        "fp": c["fp"],
                        "fn": c["fn"],
                        "dice": dice,
                        "iou": iou,
                        "sensitivity": sensitivity,
                        "precision": precision,
                    })

            case_foreground_rows.append({
                "case_id": case_id,
                "foreground_macro_dice_present_classes": (
                    float(np.mean(per_case_dices)) if per_case_dices else float("nan")
                ),
                "reference_foreground_voxels": sum(
                    case_counts[n]["target_voxels"] for n in CLASS_NAMES[1:]
                ),
                "predicted_foreground_voxels": sum(
                    case_counts[n]["predicted_voxels"] for n in CLASS_NAMES[1:]
                ),
            })

            print(f"[{case_index}/{len(cases)}] {case_id} complete")

    # Write pooled voxel-level metrics by class.
    metrics_path = OUTPUT_DIR / "focal_tversky_test_metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        fields = [
            "class", "target_voxels", "predicted_voxels", "tp", "fp", "fn",
            "dice", "iou", "sensitivity", "precision",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for name in CLASS_NAMES:
            c = totals[name]
            dice, iou, sensitivity, precision = calculate_metrics(
                c["tp"], c["fp"], c["fn"]
            )
            writer.writerow({
                "class": name,
                "target_voxels": c["target_voxels"],
                "predicted_voxels": c["predicted_voxels"],
                "tp": c["tp"],
                "fp": c["fp"],
                "fn": c["fn"],
                "dice": dice,
                "iou": iou,
                "sensitivity": sensitivity,
                "precision": precision,
            })

    case_metrics_path = OUTPUT_DIR / "focal_tversky_test_case_metrics.csv"
    with open(case_metrics_path, "w", newline="") as f:
        fields = [
            "case_id", "class", "target_voxels", "predicted_voxels",
            "tp", "fp", "fn", "dice", "iou", "sensitivity", "precision",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(case_rows)

    case_macro_path = OUTPUT_DIR / "focal_tversky_test_case_summary.csv"
    with open(case_macro_path, "w", newline="") as f:
        fields = [
            "case_id", "foreground_macro_dice_present_classes",
            "reference_foreground_voxels", "predicted_foreground_voxels",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(case_foreground_rows)

    print("\n" + "=" * 68)
    print("POOLED TEST RESULTS")
    print("=" * 68)
    print(f"Test cases: {len(cases)}")
    print(f"Test slices: {evaluated_slices}")
    print(f"{'Class':<22}{'Dice':>10}{'IoU':>10}{'Sens.':>10}{'Prec.':>10}")

    foreground_dice = []
    for name in CLASS_NAMES:
        c = totals[name]
        dice, iou, sensitivity, precision = calculate_metrics(
            c["tp"], c["fp"], c["fn"]
        )
        print(
            f"{name:<22}{dice:>10.4f}{iou:>10.4f}"
            f"{sensitivity:>10.4f}{precision:>10.4f}"
        )
        if name != "background" and np.isfinite(dice):
            foreground_dice.append(dice)

    print("-" * 68)
    print(
        "Foreground macro Dice (pooled subtype Dice average): "
        f"{np.mean(foreground_dice):.4f}"
        if foreground_dice else
        "Foreground macro Dice: unavailable"
    )

    total_ref_fg = sum(totals[n]["target_voxels"] for n in CLASS_NAMES[1:])
    total_pred_fg = sum(totals[n]["predicted_voxels"] for n in CLASS_NAMES[1:])
    print(f"Reference foreground voxels: {total_ref_fg}")
    print(f"Predicted foreground voxels:  {total_pred_fg}")

    if total_pred_fg == 0:
        print("\nWARNING: Model predicted background for every test voxel.")
    elif total_ref_fg > 0:
        print("\nThe model predicted foreground voxels; inspect subtype metrics.")

    print("\nSaved:")
    print(metrics_path)
    print(case_metrics_path)
    print(case_macro_path)
    print(
        "\nNote: evaluation is on resized 256x256 slices. "
        "Small-lesion analysis by physical volume requires a separate "
        "volume-aware analysis and is not inferred from these resized voxel counts."
    )


if __name__ == "__main__":
    main()
