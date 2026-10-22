
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
OUTPUT_CSV = (
    PROJECT_ROOT / "results" / "segmentation"
    / "v2_validation_threshold_sweep.csv"
)

# Candidate threshold values for one foreground subtype at a time.
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]


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
    )[0].float()

    label_tensor = torch.from_numpy(label.copy())[None, None].float()
    label_tensor = F.interpolate(
        label_tensor,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="nearest",
    )[0, 0].long()

    return image_tensor, label_tensor


def calculate_metrics(tp, fp, fn):
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    target_pixels = tp + fn
    predicted_pixels = tp + fp
    ratio = predicted_pixels / target_pixels if target_pixels else float("nan")

    return dice, sensitivity, precision, ratio, target_pixels, predicted_pixels


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    if not SPLIT_CSV.exists():
        raise FileNotFoundError(f"Split CSV not found: {SPLIT_CSV}")

    with SPLIT_CSV.open("r", newline="", encoding="utf-8-sig") as f:
        cases = [
            row for row in csv.DictReader(f)
            if row["split"].strip() == "validation"
        ]

    if not cases:
        raise RuntimeError("No validation cases found.")

    checkpoint = torch.load(
        CHECKPOINT, map_location=device, weights_only=True
    )
    model = SmallUNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Cache probabilities and targets to avoid repeating GPU inference.
    all_probabilities = []
    all_targets = []
    total_slices = 0

    print("Checkpoint epoch:", checkpoint.get("epoch", "unknown"))
    print("Validation cases:", len(cases))
    print("Caching validation predictions...")

    with torch.inference_mode():
        for case_index, row in enumerate(cases, start=1):
            case_id = row["case_id"]
            image_path = Path(row["image_path"])
            label_path = Path(row["annot_3_path"])

            if not image_path.exists() or not label_path.exists():
                raise FileNotFoundError(
                    f"Missing image or annotation for {case_id}"
                )

            image_volume = nib.load(
                str(image_path)
            ).get_fdata(dtype=np.float32)

            label_volume = np.rint(
                nib.load(str(label_path)).get_fdata(dtype=np.float32)
            ).astype(np.int64)

            if image_volume.shape != label_volume.shape:
                raise ValueError(
                    f"Shape mismatch for {case_id}: "
                    f"{image_volume.shape} vs {label_volume.shape}"
                )

            for z in range(image_volume.shape[2]):
                image_tensor, target = normalize_and_resize(
                    image_volume[:, :, z],
                    label_volume[:, :, z],
                )

                logits = model(image_tensor.unsqueeze(0).to(device))
                probabilities = torch.softmax(logits, dim=1)[0]

                all_probabilities.append(
                    probabilities.cpu().numpy().astype(np.float32)
                )
                all_targets.append(target.numpy().astype(np.uint8))
                total_slices += 1

            print(
                f"[{case_index}/{len(cases)}] {case_id}: "
                f"{image_volume.shape[2]} slices cached"
            )

    print("\nCached slices:", total_slices)
    print("Testing candidate thresholds:", THRESHOLDS)

    output_rows = []

    # Baseline: ordinary argmax.
    configurations = [("argmax_baseline", None)]

    # Change one subtype's threshold at a time.
    for class_id in range(1, NUM_CLASSES):
        for threshold in THRESHOLDS:
            configurations.append(
                (f"{CLASS_NAMES[class_id]}_threshold", (class_id, threshold))
            )

    for config_name, config in configurations:
        counts = np.zeros((NUM_CLASSES, 3), dtype=np.int64)

        for probs, target_array in zip(all_probabilities, all_targets):
            probabilities = torch.from_numpy(probs)
            target = torch.from_numpy(target_array.astype(np.int64))

            if config is None:
                prediction = probabilities.argmax(dim=0)
            else:
                selected_class, threshold = config

                # Normalize each class score by its decision threshold.
                # A threshold of 0.5 for every class reproduces argmax.
                thresholds = torch.full(
                    (NUM_CLASSES, 1, 1), 0.5, dtype=torch.float32
                )
                thresholds[selected_class, 0, 0] = threshold

                adjusted_scores = probabilities / thresholds
                prediction = adjusted_scores.argmax(dim=0)

            for class_id in range(NUM_CLASSES):
                truth = target == class_id
                guess = prediction == class_id

                tp = torch.logical_and(truth, guess).sum().item()
                fp = torch.logical_and(~truth, guess).sum().item()
                fn = torch.logical_and(truth, ~guess).sum().item()

                counts[class_id] += (tp, fp, fn)

        if config is None:
            print("\nBASELINE ARGMAX")
        else:
            print(f"\n{config_name}, threshold={config[1]:.2f}")

        for class_id in range(1, NUM_CLASSES):
            tp, fp, fn = counts[class_id]
            dice, sensitivity, precision, ratio, target_n, predicted_n = (
                calculate_metrics(tp, fp, fn)
            )

            output_rows.append({
                "configuration": config_name,
                "threshold": "" if config is None else config[1],
                "class_name": CLASS_NAMES[class_id],
                "dice": dice,
                "sensitivity": sensitivity,
                "precision": precision,
                "predicted_reference_ratio": ratio,
                "target_pixels": target_n,
                "predicted_pixels": predicted_n,
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "checkpoint_epoch": checkpoint.get("epoch", "unknown"),
                "validation_cases": len(cases),
                "validation_slices": total_slices,
            })

            if config is None or config[0] == class_id:
                print(
                    f"  {CLASS_NAMES[class_id]:20s} "
                    f"Dice={dice:.4f} "
                    f"Sensitivity={sensitivity:.4f} "
                    f"Precision={precision:.4f} "
                    f"Pred/Ref={ratio:.2f}x"
                )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(output_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print("\nSaved threshold results:", OUTPUT_CSV)
    print("No training or checkpoint modification was performed.")
    print("Thresholds are evaluated on validation data only.")


if __name__ == "__main__":
    main()
