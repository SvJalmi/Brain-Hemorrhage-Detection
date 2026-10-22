
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
    PROJECT_ROOT
    / "preprocessing"
    / "segmentation"
    / "mbh_seg25_split.csv"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "segmentation"
    / "small_unet_mbh_seg25_v2_resumed_best.pt"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "segmentation"
    / "v2_resumed_validation_confusion_matrix.csv"
)


def normalize_and_resize(image_slice, label_slice):
    """Apply the same slice normalization and resizing used in validation."""

    image = np.asarray(image_slice, dtype=np.float32)
    label = np.rint(label_slice).astype(np.int64)

    if np.any((label < 0) | (label >= NUM_CLASSES)):
        raise ValueError(
            f"Invalid label values: {np.unique(label)}"
        )

    low, high = np.percentile(image, [1, 99])

    if high > low:
        image = (
            np.clip(image, low, high) - low
        ) / (high - low)
    else:
        image = np.zeros_like(image, dtype=np.float32)

    image_tensor = torch.from_numpy(
        image.copy()
    )[None, None]

    image_tensor = F.interpolate(
        image_tensor,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )[0].float()

    label_tensor = torch.from_numpy(
        label.copy()
    )[None, None].float()

    label_tensor = F.interpolate(
        label_tensor,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="nearest",
    )[0, 0].long()

    return image_tensor, label_tensor


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    if not SPLIT_CSV.exists():
        raise FileNotFoundError(
            f"Split CSV not found: {SPLIT_CSV}"
        )

    with SPLIT_CSV.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        cases = [
            row
            for row in csv.DictReader(f)
            if row["split"].strip() == "validation"
        ]

    if not cases:
        raise RuntimeError(
            "No validation cases found."
        )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=True,
    )

    model = SmallUNet().to(device)
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()

    # Rows = reference class.
    # Columns = predicted class.
    confusion = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64,
    )

    total_slices = 0

    print(
        "Checkpoint epoch:",
        checkpoint.get("epoch", "unknown"),
    )
    print("Validation cases:", len(cases))
    print("Rows = reference; columns = prediction")
    print("Processing validation slices only...")

    with torch.inference_mode():
        for case_index, row in enumerate(
            cases,
            start=1,
        ):
            case_id = row["case_id"]

            image_path = Path(row["image_path"])
            label_path = Path(row["annot_3_path"])

            if (
                not image_path.exists()
                or not label_path.exists()
            ):
                raise FileNotFoundError(
                    f"Missing image or annotation "
                    f"for case {case_id}"
                )

            image_volume = nib.load(
                str(image_path)
            ).get_fdata(dtype=np.float32)

            label_volume = np.rint(
                nib.load(
                    str(label_path)
                ).get_fdata(dtype=np.float32)
            ).astype(np.int64)

            if image_volume.shape != label_volume.shape:
                raise ValueError(
                    f"Shape mismatch for {case_id}: "
                    f"{image_volume.shape} vs "
                    f"{label_volume.shape}"
                )

            for z in range(image_volume.shape[2]):
                image_tensor, target = normalize_and_resize(
                    image_volume[:, :, z],
                    label_volume[:, :, z],
                )

                image_tensor = (
                    image_tensor.unsqueeze(0).to(device)
                )

                # FIX: target and prediction must use
                # the same device for confusion counting.
                target = target.to(device)

                logits = model(image_tensor)

                prediction = logits.argmax(
                    dim=1
                ).squeeze(0)

                encoded = (
                    target.reshape(-1) * NUM_CLASSES
                    + prediction.reshape(-1)
                )

                slice_counts = torch.bincount(
                    encoded,
                    minlength=NUM_CLASSES * NUM_CLASSES,
                ).reshape(
                    NUM_CLASSES,
                    NUM_CLASSES,
                )

                confusion += slice_counts.cpu().numpy()
                total_slices += 1

            print(
                f"[{case_index}/{len(cases)}] "
                f"{case_id}: "
                f"{image_volume.shape[2]} slices processed"
            )

    print(
        "\n========== VALIDATION CONFUSION MATRIX =========="
    )
    print(
        "Rows = reference class; columns = predicted class"
    )
    print("Class order:", " | ".join(CLASS_NAMES))
    print("Total slices:", total_slices)
    print("Total pixels:", int(confusion.sum()))

    header = "Reference \\ Predicted".ljust(22)

    for name in CLASS_NAMES:
        header += f"{name[:12]:>14}"

    print("\n" + header)

    for i, name in enumerate(CLASS_NAMES):
        line = f"{name[:21]:22}"

        for j in range(NUM_CLASSES):
            line += f"{confusion[i, j]:14d}"

        print(line)

    print(
        "\n========== ROW-NORMALIZED CONFUSION (%) =========="
    )
    print(
        "Each row shows the predictions for one reference class."
    )
    print(header)

    for i, name in enumerate(CLASS_NAMES):
        row_total = confusion[i].sum()
        line = f"{name[:21]:22}"

        for j in range(NUM_CLASSES):
            percentage = (
                100.0 * confusion[i, j] / row_total
                if row_total
                else 0.0
            )
            line += f"{percentage:13.2f}%"

        print(line)

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "reference_class",
            *CLASS_NAMES,
            "row_total",
        ])

        for i, name in enumerate(CLASS_NAMES):
            writer.writerow([
                name,
                *[int(value) for value in confusion[i]],
                int(confusion[i].sum()),
            ])

    print("\nSaved confusion matrix:")
    print(OUTPUT_CSV)
    print(
        "No training or checkpoint modification was performed."
    )


if __name__ == "__main__":
    main()
