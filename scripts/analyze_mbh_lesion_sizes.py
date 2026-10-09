from pathlib import Path
import csv

import nibabel as nib
import numpy as np
import pandas as pd


# ============================================================
# MBH-Seg25 — LESION SIZE / VOLUME ANALYSIS
# Uses reference training annotation: annot_3
# Raw CT images and annotations are read-only.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = PROJECT_ROOT / "datasets" / "MBH-Seg25"

SPLIT_CSV = (
    PROJECT_ROOT
    / "preprocessing"
    / "segmentation"
    / "mbh_seg25_split.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "segmentation"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "mbh_seg25_lesion_size_analysis.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "mbh_seg25_lesion_size_summary.csv"
)

NUM_CLASSES = 6

CLASS_NAMES = {
    1: "epidural",
    2: "intraparenchymal",
    3: "intraventricular",
    4: "subarachnoid",
    5: "subdural",
}


# ============================================================
# LOAD SPLIT CASES
# ============================================================

def load_cases():

    cases = []

    with open(
        SPLIT_CSV,
        "r",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            case_id = row["case_id"]
            split = row["split"]

            case_dir = DATASET_ROOT / case_id

            cases.append(
                {
                    "case_id": case_id,
                    "split": split,
                    "case_dir": case_dir,
                }
            )

    return cases


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MBH-Seg25 — LESION SIZE / VOLUME ANALYSIS")
    print("=" * 70)

    print()
    print(
        "Reference annotation: label_annot_3.nii.gz"
    )

    cases = load_cases()

    print()
    print(
        "Cases found:",
        len(cases),
    )

    rows = []

    # --------------------------------------------------------
    # Process every case
    # --------------------------------------------------------

    for case_number, case_info in enumerate(
        cases,
        start=1,
    ):

        case_id = case_info["case_id"]
        split = case_info["split"]
        case_dir = case_info["case_dir"]

        label_path = (
            case_dir
            / "label_annot_3.nii.gz"
        )

        if not label_path.exists():

            print(
                f"WARNING: Missing annotation "
                f"for {case_id}"
            )

            continue

        label_nii = nib.load(
            str(label_path)
        )

        label = label_nii.get_fdata(
            dtype=np.float32
        )

        label = np.rint(
            label
        ).astype(np.uint8)

        spacing = (
            label_nii.header
            .get_zooms()[:3]
        )

        voxel_volume_mm3 = (
            float(spacing[0])
            * float(spacing[1])
            * float(spacing[2])
        )

        voxel_volume_ml = (
            voxel_volume_mm3
            / 1000.0
        )

        # ----------------------------------------------------
        # Overall foreground
        # ----------------------------------------------------

        foreground_mask = label > 0

        total_foreground_voxels = int(
            foreground_mask.sum()
        )

        total_foreground_volume_ml = (
            total_foreground_voxels
            * voxel_volume_ml
        )

        # ----------------------------------------------------
        # Per-subtype measurements
        # ----------------------------------------------------

        row = {
            "case_id": case_id,
            "split": split,
            "spacing_x_mm": spacing[0],
            "spacing_y_mm": spacing[1],
            "spacing_z_mm": spacing[2],
            "voxel_volume_ml": voxel_volume_ml,
            "total_foreground_voxels":
                total_foreground_voxels,
            "total_foreground_volume_ml":
                total_foreground_volume_ml,
        }

        for class_id, class_name in CLASS_NAMES.items():

            class_mask = (
                label == class_id
            )

            voxel_count = int(
                class_mask.sum()
            )

            volume_ml = (
                voxel_count
                * voxel_volume_ml
            )

            row[
                f"{class_name}_voxels"
            ] = voxel_count

            row[
                f"{class_name}_volume_ml"
            ] = volume_ml

        rows.append(row)

        if (
            case_number == 1
            or case_number % 25 == 0
            or case_number == len(cases)
        ):

            print(
                f"Processed "
                f"{case_number}/{len(cases)} "
                f"cases"
            )

    # --------------------------------------------------------
    # Create dataframe
    # --------------------------------------------------------

    df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Total hemorrhage volume statistics
    # --------------------------------------------------------

    volume_values = (
        df["total_foreground_volume_ml"]
        .to_numpy()
    )

    positive_volumes = (
        volume_values[
            volume_values > 0
        ]
    )

    print()
    print("=" * 70)
    print("OVERALL LESION VOLUME")
    print("=" * 70)

    print()
    print(
        "Cases with foreground:",
        len(positive_volumes),
        "/",
        len(df),
    )

    if len(positive_volumes) > 0:

        print(
            f"Minimum volume: "
            f"{np.min(positive_volumes):.4f} mL"
        )

        print(
            f"Median volume: "
            f"{np.median(positive_volumes):.4f} mL"
        )

        print(
            f"Mean volume: "
            f"{np.mean(positive_volumes):.4f} mL"
        )

        print(
            f"Maximum volume: "
            f"{np.max(positive_volumes):.4f} mL"
        )

    # --------------------------------------------------------
    # Small-lesion categories
    #
    # These are analysis bins, not clinical thresholds.
    # --------------------------------------------------------

    df["small_lt_1ml"] = (
        (df["total_foreground_volume_ml"] > 0)
        & (df["total_foreground_volume_ml"] < 1.0)
    )

    df["small_1_to_5ml"] = (
        (df["total_foreground_volume_ml"] >= 1.0)
        & (df["total_foreground_volume_ml"] < 5.0)
    )

    df["medium_5_to_20ml"] = (
        (df["total_foreground_volume_ml"] >= 5.0)
        & (df["total_foreground_volume_ml"] < 20.0)
    )

    df["large_ge_20ml"] = (
        df["total_foreground_volume_ml"] >= 20.0
    )

    # --------------------------------------------------------
    # Summary table
    # --------------------------------------------------------

    summary_rows = []

    categories = [
        (
            "Small <1 mL",
            "small_lt_1ml",
        ),
        (
            "Small 1-5 mL",
            "small_1_to_5ml",
        ),
        (
            "Medium 5-20 mL",
            "medium_5_to_20ml",
        ),
        (
            "Large >=20 mL",
            "large_ge_20ml",
        ),
    ]

    for category_name, column_name in categories:

        count = int(
            df[column_name].sum()
        )

        percentage = (
            100.0 * count / len(df)
            if len(df) > 0
            else 0.0
        )

        summary_rows.append(
            {
                "category": category_name,
                "cases": count,
                "percentage_of_all_cases":
                    percentage,
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    print()
    print("=" * 70)
    print("LESION SIZE DISTRIBUTION")
    print("=" * 70)

    for _, row in summary_df.iterrows():

        print(
            f"{row['category']:<20}"
            f"{int(row['cases']):>6} cases "
            f"({row['percentage_of_all_cases']:.2f}%)"
        )

    # --------------------------------------------------------
    # Subtype summary
    # --------------------------------------------------------

    subtype_rows = []

    for class_id, class_name in CLASS_NAMES.items():

        volume_column = (
            f"{class_name}_volume_ml"
        )

        voxel_column = (
            f"{class_name}_voxels"
        )

        positive = df[
            df[volume_column] > 0
        ][volume_column]

        subtype_rows.append(
            {
                "subtype": class_name,
                "cases_with_subtype":
                    int(len(positive)),
                "median_volume_ml":
                    float(
                        np.median(positive)
                    )
                    if len(positive) > 0
                    else 0.0,
                "mean_volume_ml":
                    float(
                        np.mean(positive)
                    )
                    if len(positive) > 0
                    else 0.0,
                "max_volume_ml":
                    float(
                        np.max(positive)
                    )
                    if len(positive) > 0
                    else 0.0,
                "total_voxels":
                    int(
                        df[voxel_column].sum()
                    ),
            }
        )

    subtype_df = pd.DataFrame(
        subtype_rows
    )

    print()
    print("=" * 70)
    print("SUBTYPE VOLUME SUMMARY")
    print("=" * 70)

    for _, row in subtype_df.iterrows():

        print(
            f"{row['subtype']:<20}"
            f"Cases: "
            f"{int(row['cases_with_subtype']):>3}   "
            f"Median: "
            f"{row['median_volume_ml']:.4f} mL   "
            f"Mean: "
            f"{row['mean_volume_ml']:.4f} mL   "
            f"Max: "
            f"{row['max_volume_ml']:.4f} mL"
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    summary_output = pd.concat(
        [
            summary_df.assign(
                summary_type="size_category"
            ),
            subtype_df.assign(
                summary_type="subtype"
            ),
        ],
        ignore_index=True,
        sort=False,
    )

    summary_output.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print()
    print("Detailed results:")
    print(OUTPUT_FILE)

    print()
    print("Summary results:")
    print(SUMMARY_FILE)


if __name__ == "__main__":
    main()