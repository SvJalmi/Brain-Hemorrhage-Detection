

from pathlib import Path
from collections import Counter

import nibabel as nib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "datasets" / "MBH-Seg25"
SPLIT_CSV = (
    ROOT
    / "preprocessing"
    / "segmentation"
    / "mbh_seg25_split.csv"
)

LABEL_NAMES = {
    0: "Background",
    1: "Epidural",
    2: "Intraparenchymal",
    3: "Intraventricular",
    4: "Subarachnoid",
    5: "Subdural",
}


def main():
    print("=" * 72)
    print("MBH-Seg25 - READ-ONLY SLICE DISTRIBUTION DIAGNOSTIC")
    print("=" * 72)

    if not SPLIT_CSV.exists():
        raise FileNotFoundError(
            f"Split CSV not found: {SPLIT_CSV}"
        )

    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATA_ROOT}"
        )

    split_df = pd.read_csv(SPLIT_CSV)

    required_columns = {"case_id", "split"}
    missing_columns = required_columns - set(split_df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing CSV columns: {missing_columns}. "
            f"Available columns: {list(split_df.columns)}"
        )

    for split_name in ["train", "validation", "test"]:
        cases = split_df[
            split_df["split"]
            .astype(str)
            .str.lower()
            .eq(split_name)
        ]

        slice_counts = Counter()
        subtype_case_counts = Counter()
        subtype_slice_counts = Counter()
        invalid_labels = Counter()

        total_slices = 0
        cases_processed = 0
        cases_with_foreground = 0
        missing_cases = 0
        shape_mismatches = 0

        print("\n" + "-" * 72)
        print(f"SPLIT: {split_name.upper()}")
        print(f"Cases listed in split CSV: {len(cases)}")

        for _, row in cases.iterrows():
            case_id = str(row["case_id"])
            case_dir = DATA_ROOT / case_id

            image_path = case_dir / "image.nii.gz"
            label_path = case_dir / "label_annot_3.nii.gz"

            if not image_path.exists() or not label_path.exists():
                print(f"Missing image or label: {case_id}")
                missing_cases += 1
                continue

            image_nii = nib.load(str(image_path))
            label_nii = nib.load(str(label_path))

            if image_nii.shape != label_nii.shape:
                print(
                    f"Shape mismatch for {case_id}: "
                    f"image={image_nii.shape}, "
                    f"label={label_nii.shape}"
                )
                shape_mismatches += 1
                continue

            labels = np.asarray(label_nii.dataobj)
            labels = np.rint(labels).astype(np.int16)

            if labels.ndim != 3:
                print(
                    f"Unexpected label dimensions for {case_id}: "
                    f"{labels.shape}"
                )
                continue

            total_slices += labels.shape[2]
            cases_processed += 1

            values = np.unique(labels)

            for value in values:
                if int(value) not in LABEL_NAMES:
                    invalid_labels[int(value)] += int(
                        np.count_nonzero(labels == value)
                    )

            case_has_foreground = False
            case_subtypes = set()
            foreground_slice_count = 0

            for z in range(labels.shape[2]):
                slice_labels = labels[:, :, z]
                present = np.unique(slice_labels)
                present = present[present > 0]

                if len(present) == 0:
                    slice_counts["background_only"] += 1
                else:
                    slice_counts["with_foreground"] += 1
                    foreground_slice_count += 1
                    case_has_foreground = True

                for class_id in present:
                    class_id = int(class_id)

                    if class_id in LABEL_NAMES:
                        subtype_slice_counts[class_id] += 1
                        case_subtypes.add(class_id)

            if case_has_foreground:
                cases_with_foreground += 1

            for class_id in case_subtypes:
                subtype_case_counts[class_id] += 1

            print(
                f"Processed {case_id}: "
                f"depth={labels.shape[2]}, "
                f"foreground_slices={foreground_slice_count}"
            )

        print("\nSLICE SUMMARY")
        print(f"Cases successfully processed: {cases_processed}")
        print(f"Missing cases/files: {missing_cases}")
        print(f"Image-label shape mismatches: {shape_mismatches}")
        print(f"Cases containing foreground: {cases_with_foreground}")
        print(f"Total slices: {total_slices}")
        print(
            "Foreground-containing slices: "
            f"{slice_counts['with_foreground']}"
        )
        print(
            "Background-only slices: "
            f"{slice_counts['background_only']}"
        )

        print("\nSUBTYPE REPRESENTATION")

        for class_id in range(1, 6):
            print(
                f"{LABEL_NAMES[class_id]}: "
                f"{subtype_case_counts[class_id]} cases; "
                f"{subtype_slice_counts[class_id]} slices"
            )

        if invalid_labels:
            print("\nWARNING: Unexpected label values found:")
            print(dict(invalid_labels))
        else:
            print("\nNo unexpected label values found.")

    print("\n" + "=" * 72)
    print("DIAGNOSTIC COMPLETE")
    print("No dataset, model, or checkpoint files were modified.")
    print("=" * 72)


if __name__ == "__main__":
    main()
