import csv
from pathlib import Path

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_CSV = (
    PROJECT_ROOT
    / "preprocessing"
    / "segmentation"
    / "mbh_seg25_split.csv"
)

CLASS_NAMES = [
    "background",
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
]

FOREGROUND_PROBABILITY = 0.75
SEED = 42


def main():
    rng = np.random.default_rng(SEED)

    with SPLIT_CSV.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        rows = [
            row
            for row in csv.DictReader(file)
            if row["split"].strip() == "train"
        ]

    if not rows:
        raise RuntimeError("No training cases found in the split CSV.")

    selected_slices = 0
    selected_class_slices = np.zeros(6, dtype=np.int64)
    selected_class_voxels = np.zeros(6, dtype=np.int64)
    cases_with_class = np.zeros(6, dtype=np.int64)

    print("Training cases:", len(rows))

    for case_index, row in enumerate(rows, start=1):
        label_path = Path(row["annot_3_path"])

        if not label_path.exists():
            raise FileNotFoundError(label_path)

        label = np.rint(
            nib.load(str(label_path)).get_fdata(dtype=np.float32)
        ).astype(np.int16)

        if np.any((label < 0) | (label >= len(CLASS_NAMES))):
            raise ValueError(
                f"Unexpected label values in {label_path}"
            )

        foreground = [
            z
            for z in range(label.shape[2])
            if np.any(label[:, :, z] > 0)
        ]

        background = [
            z
            for z in range(label.shape[2])
            if not np.any(label[:, :, z] > 0)
        ]

        if foreground and background:
            n_background = int(
                len(foreground)
                * (1 - FOREGROUND_PROBABILITY)
                / FOREGROUND_PROBABILITY
            )

            n_background = min(
                len(background),
                max(1, n_background),
            )

            selected_background = rng.choice(
                background,
                size=n_background,
                replace=False,
            ).tolist()

            selected = foreground + selected_background
        else:
            selected = foreground or background

        case_has_class = np.zeros(6, dtype=bool)

        for z in selected:
            slice_label = label[:, :, z]
            selected_slices += 1

            for class_id in range(6):
                mask = slice_label == class_id

                if mask.any():
                    selected_class_slices[class_id] += 1
                    case_has_class[class_id] = True

                selected_class_voxels[class_id] += int(mask.sum())

        cases_with_class += case_has_class.astype(np.int64)

        print(
            f"[{case_index}/{len(rows)}] "
            f"{row['case_id']} processed"
        )

    print("\n========== TRAINING SAMPLER AUDIT ==========")
    print("Training cases:", len(rows))
    print("Selected slices:", selected_slices)
    print(
        "\nClass | Slices containing class | "
        "Cases containing class | Selected voxels"
    )

    for class_id, name in enumerate(CLASS_NAMES):
        print(
            f"{name:20s} | "
            f"{selected_class_slices[class_id]:7d} | "
            f"{cases_with_class[class_id]:4d} | "
            f"{selected_class_voxels[class_id]:12d}"
        )

    print("\nAudit complete. No dataset or checkpoint was modified.")


if __name__ == "__main__":
    main()