
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
    / "v2_resumed_validation_probabilities.csv"
)


def normalize_and_resize(image_slice):
    image = np.asarray(image_slice, dtype=np.float32)
    low, high = np.percentile(image, [1, 99])

    if high > low:
        image = (np.clip(image, low, high) - low) / (high - low)
    else:
        image = np.zeros_like(image, dtype=np.float32)

    tensor = torch.from_numpy(image.copy())[None, None]
    tensor = F.interpolate(
        tensor,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )[0]

    return tensor.float()


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

    probability_sum = np.zeros(NUM_CLASSES, dtype=np.float64)
    argmax_pixels = np.zeros(NUM_CLASSES, dtype=np.int64)
    selected_probability_sum = np.zeros(NUM_CLASSES, dtype=np.float64)

    total_pixels = 0
    total_slices = 0

    print("Checkpoint epoch:", checkpoint.get("epoch", "unknown"))
    print("Validation cases:", len(cases))
    print("Processing validation slices only...")

    with torch.inference_mode():
        for case_index, row in enumerate(cases, start=1):
            case_id = row["case_id"]
            image_path = Path(row["image_path"])

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Missing image volume for case {case_id}: {image_path}"
                )

            volume = nib.load(str(image_path)).get_fdata(dtype=np.float32)

            for z in range(volume.shape[2]):
                image_tensor = normalize_and_resize(volume[:, :, z])
                logits = model(image_tensor.unsqueeze(0).to(device))
                probabilities = torch.softmax(logits, dim=1)[0]
                prediction = probabilities.argmax(dim=0)

                probability_sum += (
                    probabilities.mean(dim=(1, 2)).cpu().numpy()
                )

                counts = torch.bincount(
                    prediction.reshape(-1),
                    minlength=NUM_CLASSES,
                )
                argmax_pixels += counts.cpu().numpy()

                for class_id in range(NUM_CLASSES):
                    mask = prediction == class_id
                    if mask.any():
                        selected_probability_sum[class_id] += (
                            probabilities[class_id][mask].sum().item()
                        )

                total_pixels += prediction.numel()
                total_slices += 1

            print(
                f"[{case_index}/{len(cases)}] {case_id}: "
                f"{volume.shape[2]} slices processed"
            )

    rows = []

    print("\n========== VALIDATION PROBABILITY SUMMARY ==========")
    print("Slices:", total_slices)
    print("Pixels:", total_pixels)
    print("Checkpoint epoch:", checkpoint.get("epoch", "unknown"))

    print(
        "\nClass | Mean probability (%) | "
        "Argmax pixel share (%) | Mean confidence when selected (%)"
    )

    for class_id, class_name in enumerate(CLASS_NAMES):
        mean_probability = probability_sum[class_id] / total_slices
        pixel_share = 100.0 * argmax_pixels[class_id] / total_pixels

        selected_confidence = (
            selected_probability_sum[class_id] / argmax_pixels[class_id]
            if argmax_pixels[class_id] else 0.0
        )

        print(
            f"{class_name:20s} | {100 * mean_probability:8.4f} | "
            f"{pixel_share:8.4f} | {100 * selected_confidence:8.4f}"
        )

        rows.append({
            "class_name": class_name,
            "mean_probability": mean_probability,
            "argmax_pixel_share": argmax_pixels[class_id] / total_pixels,
            "argmax_pixel_count": int(argmax_pixels[class_id]),
            "mean_confidence_when_selected": selected_confidence,
            "checkpoint_epoch": checkpoint.get("epoch", "unknown"),
            "validation_cases": len(cases),
            "validation_slices": total_slices,
        })

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved:", OUTPUT_CSV)
    print("No training or checkpoint modification was performed.")


if __name__ == "__main__":
    main()
