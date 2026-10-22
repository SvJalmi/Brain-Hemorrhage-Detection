
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


def main():
    with SPLIT_CSV.open(
        "r", newline="", encoding="utf-8-sig"
    ) as f:
        rows = [
            row
            for row in csv.DictReader(f)
            if row["split"].strip() == "train"
        ]

    if not rows:
        raise RuntimeError("No training cases found.")

    total_voxels = np.zeros(6, dtype=np.int64)
    slices_with_class = np.zeros(6, dtype=np.int64)
    cases_with_class = np.zeros(6, dtype=np.int64)

    print("Training cases:", len(rows))

    for index, row in enumerate(rows, start=1):
        label_path = Path(row["annot_3_path"])

        if not label_path.exists():
            raise FileNotFoundError(label_path)

        volume = np.rint(
            nib.load(str(label_path)).get_fdata()
        ).astype(np.int16)

        if np.any((volume < 0) | (volume >= 6)):
            raise ValueError(
                f"Unexpected label values in {label_path}"
            )

        for class_id in range(6):
            mask = volume == class_id

            total_voxels[class_id] += int(mask.sum())

            # Count each slice at most once for each class.
            slices_with_class[class_id] += int(
                np.count_nonzero(mask.any(axis=(0, 1)))
            )

            if mask.any():
                cases_with_class[class_id] += 1

        print(
            f"[{index}/{len(rows)}] {row['case_id']} processed"
        )

    print("\n========== TRAINING LABEL DISTRIBUTION ==========")
    print(
        "Class | Total voxels | Slices containing class "
        "| Cases containing class"
    )

    for class_id, name in enumerate(CLASS_NAMES):
        print(
            f"{name:20s} | "
            f"{total_voxels[class_id]:12d} | "
            f"{slices_with_class[class_id]:6d} | "
            f"{cases_with_class[class_id]:4d}"
        )

    print("\nAudit complete. No data or checkpoint was modified.")


if __name__ == "__main__":
    main()
