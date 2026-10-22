
from pathlib import Path
import csv
import sys

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.small_unet_focal_tversky_v2 import (
    SmallUNet,
    IMAGE_SIZE,
    NUM_CLASSES,
    CLASS_NAMES,
)

SPLIT_CSV = (
    PROJECT_ROOT / "preprocessing" / "segmentation"
    / "mbh_seg25_split.csv"
)

CHECKPOINT = (
    PROJECT_ROOT / "models" / "segmentation"
    / "small_unet_mbh_seg25_v2_resumed_best.pt"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "segmentation"
METRICS_PATH = OUTPUT_DIR / "v2_resumed_full_validation_metrics.csv"
CASE_METRICS_PATH = OUTPUT_DIR / "v2_resumed_full_validation_case_metrics.csv"


def load_validation_cases():
    with SPLIT_CSV.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    rows = [r for r in rows if r["split"].strip() == "validation"]

    if not rows:
        raise RuntimeError("No validation cases found in split CSV.")

    return rows


def normalize_and_resize(image_slice, label_slice):
    image = np.asarray(image_slice, dtype=np.float32)
    label = np.rint(label_slice).astype(np.int64)

    if np.any((label < 0) | (label >= NUM_CLASSES)):
        raise ValueError(f"Invalid label values: {np.unique(label)}")

    low, high = np.percentile(image, [1, 99])
    if high > low:
        image = (np.clip(image, low, high) - low) / (high - low)
    else:
        image = np.zeros_like(image, dtype=np.float32)

    image_tensor = torch.from_numpy(image.copy())[None, None]
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    if not SPLIT_CSV.exists():
        raise FileNotFoundError(f"Split CSV not found: {SPLIT_CSV}")

    checkpoint = torch.load(
        CHECKPOINT, map_location=device, weights_only=True
    )

    model = SmallUNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Checkpoint:", CHECKPOINT.name)
    print("Checkpoint epoch:", checkpoint.get("epoch", "unknown"))

    cases = load_validation_cases()
    print("Validation cases:", len(cases))
    print("Test cases will not be loaded.")
    print("Running inference on every slice in each validation volume...")

    # Columns: TP, FP, FN for each class.
    counts = np.zeros((NUM_CLASSES, 3), dtype=np.int64)
    case_rows = []
    total_slices = 0
    total_reference_fg = 0
    total_predicted_fg = 0

    with torch.inference_mode():
        for case_index, row in enumerate(cases, start=1):
            case_id = row["case_id"]
            image_path = Path(row["image_path"])
            label_path = Path(row["annot_3_path"])

            if not image_path.exists() or not label_path.exists():
                raise FileNotFoundError(
                    f"Missing volume or annotation for case {case_id}"
                )

            image_volume = nib.load(str(image_path)).get_fdata(
                dtype=np.float32
            )
            label_volume = np.rint(
                nib.load(str(label_path)).get_fdata(dtype=np.float32)
            ).astype(np.int64)

            if image_volume.shape != label_volume.shape:
                raise ValueError(
                    f"Image/label shape mismatch for {case_id}: "
                    f"{image_volume.shape} vs {label_volume.shape}"
                )

            case_counts = np.zeros((NUM_CLASSES, 3), dtype=np.int64)
            case_reference_fg = 0
            case_predicted_fg = 0

            for z in range(image_volume.shape[2]):
                image_tensor, target = normalize_and_resize(
                    image_volume[:, :, z],
                    label_volume[:, :, z],
                )

                image_tensor = image_tensor.unsqueeze(0).float().to(device)
                target = target.to(device)

                logits = model(image_tensor)
                prediction = logits.argmax(dim=1).squeeze(0)

                total_slices += 1
                case_reference_fg += int((target > 0).sum().item())
                case_predicted_fg += int((prediction > 0).sum().item())

                for class_id in range(NUM_CLASSES):
                    truth = target == class_id
                    guess = prediction == class_id

                    tp = torch.logical_and(truth, guess).sum().item()
                    fp = torch.logical_and(~truth, guess).sum().item()
                    fn = torch.logical_and(truth, ~guess).sum().item()

                    case_counts[class_id, 0] += tp
                    case_counts[class_id, 1] += fp
                    case_counts[class_id, 2] += fn

            counts += case_counts
            total_reference_fg += case_reference_fg
            total_predicted_fg += case_predicted_fg

            case_row = {
                "case_id": case_id,
                "slices": image_volume.shape[2],
                "reference_fg_pixels": case_reference_fg,
                "predicted_fg_pixels": case_predicted_fg,
            }

            for class_id, class_name in enumerate(CLASS_NAMES):
                tp, fp, fn = case_counts[class_id]
                denom = 2 * tp + fp + fn
                case_row[f"{class_name}_dice"] = (
                    2 * tp / denom if denom else 1.0
                )
                case_row[f"{class_name}_tp"] = int(tp)
                case_row[f"{class_name}_fp"] = int(fp)
                case_row[f"{class_name}_fn"] = int(fn)

            case_rows.append(case_row)

            print(
                f"[{case_index}/{len(cases)}] {case_id}: "
                f"slices={image_volume.shape[2]}, "
                f"reference FG={case_reference_fg}, "
                f"predicted FG={case_predicted_fg}"
            )

    print("\n========== FULL VALIDATION SUMMARY ==========")
    print("Checkpoint epoch:", checkpoint.get("epoch", "unknown"))
    print("Cases evaluated:", len(cases))
    print("Slices evaluated:", total_slices)
    print("Reference foreground pixels:", total_reference_fg)
    print("Predicted foreground pixels:", total_predicted_fg)

    metric_rows = []
    print("\nPooled class metrics:")

    for class_id, class_name in enumerate(CLASS_NAMES):
        tp, fp, fn = counts[class_id]
        denom = 2 * tp + fp + fn
        target_count = tp + fn
        predicted_count = tp + fp

        dice = 2 * tp / denom if denom else 1.0
        sensitivity = tp / target_count if target_count else float("nan")
        precision = tp / predicted_count if predicted_count else float("nan")

        print(
            f"{class_name:20s} Dice={dice:.4f} "
            f"Sensitivity={sensitivity:.4f} "
            f"Precision={precision:.4f} "
            f"Target={target_count} Predicted={predicted_count}"
        )

        metric_rows.append({
            "class_name": class_name,
            "dice": dice,
            "sensitivity": sensitivity,
            "precision": precision,
            "target_pixels": target_count,
            "predicted_pixels": predicted_count,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "checkpoint_epoch": checkpoint.get("epoch", "unknown"),
            "cases_evaluated": len(cases),
            "slices_evaluated": total_slices,
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with METRICS_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)

    with CASE_METRICS_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(case_rows[0].keys()))
        writer.writeheader()
        writer.writerows(case_rows)

    print("\nSaved pooled metrics:", METRICS_PATH)
    print("Saved case metrics:", CASE_METRICS_PATH)
    print("No training or checkpoint modification was performed.")


if __name__ == "__main__":
    main()
