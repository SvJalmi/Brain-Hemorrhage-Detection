from pathlib import Path
import re

import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing.common.config import TBIHEMOCT_DIR


# ============================================================
# TBI-HemoCT BASE-IMAGE / LEAKAGE-SAFE SPLIT
# ============================================================

SEED = 42

RAW_CSV = (
    TBIHEMOCT_DIR
    / "TBI-HemoCT Data set"
    / "TBI-HemoCT Data set"
    / "final_log_file.csv"
)

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = OUTPUT_DIR / "tbihemoct_base_split.csv"

CLASS_ORDER = [
    "normal",
    "epidural",
    "subdural",
    "subarachnoid",
    "intraparenchymal",
    "intraventricular",
]


# ============================================================
# HELPERS
# ============================================================

def extract_image_id(filename):
    """
    Extract numeric image ID.

    Examples:
        ID_123_subdural_UNIQUE.png
        ID_123_subdural_aug_1234.png
    """

    match = re.search(r"ID_(\d+)", str(filename))

    if match:
        return int(match.group(1))

    return None


def remove_augmentation_suffix(filename):
    """
    Convert an augmented filename into its base filename.

    Example:
        ID_123_subdural_aug_1234.png
        ->
        ID_123_subdural.png

    Non-augmented files remain unchanged.
    """

    return re.sub(
        r"_aug_\d+(?=\.png$)",
        "",
        str(filename),
        flags=re.IGNORECASE,
    )


# ============================================================
# BUILD BASE-IMAGE TABLE
# ============================================================

def build_base_image_table(df):
    """
    Build exactly one record per unique base image.

    Augmented variants are grouped with their original base image.
    """

    working = df.copy()

    # --------------------------------------------------------
    # Extract image ID
    # --------------------------------------------------------

    working["image_id"] = working["File Name"].apply(
        extract_image_id
    )

    if working["image_id"].isna().any():
        raise ValueError(
            "Some rows do not contain a valid image ID."
        )

    # --------------------------------------------------------
    # Remove augmentation suffix
    # --------------------------------------------------------

    working["base_file"] = working["File Name"].apply(
        remove_augmentation_suffix
    )

    # --------------------------------------------------------
    # Verify base-file metadata consistency
    # --------------------------------------------------------

    grouped = working.groupby("base_file")

    label_conflicts = (
        grouped["Label"]
        .nunique()
    )

    destination_conflicts = (
        grouped["Destination"]
        .nunique()
    )

    if (label_conflicts > 1).any():
        bad = label_conflicts[label_conflicts > 1]

        raise ValueError(
            "Label conflicts detected among augmentation variants:\n"
            f"{bad.head(20).to_string()}"
        )

    if (destination_conflicts > 1).any():
        bad = destination_conflicts[
            destination_conflicts > 1
        ]

        raise ValueError(
            "Destination conflicts detected among augmentation variants:\n"
            f"{bad.head(20).to_string()}"
        )

    # --------------------------------------------------------
    # One row per base image
    # --------------------------------------------------------

    base_df = (
        working[
            [
                "image_id",
                "base_file",
                "Destination",
                "Label",
            ]
        ]
        .drop_duplicates("base_file")
        .copy()
    )

    # --------------------------------------------------------
    # Rename Destination -> class_name
    # --------------------------------------------------------

    base_df["class_name"] = (
        base_df["Destination"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # --------------------------------------------------------
    # Validate class names
    # --------------------------------------------------------

    unexpected = sorted(
        set(base_df["class_name"])
        - set(CLASS_ORDER)
    )

    if unexpected:
        raise ValueError(
            "Unexpected class names found:\n"
            f"{unexpected}"
        )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    base_df = base_df.sort_values(
        ["image_id", "class_name", "base_file"]
    ).reset_index(drop=True)

    return base_df


# ============================================================
# CREATE ONE SIGNATURE PER IMAGE ID
# ============================================================

def create_id_level_table(base_df):
    """
    Create exactly ONE row per image ID.

    Each ID may contain 5 or 6 base images/classes.

    Example:

        ID 1:
            normal
            epidural
            subdural
            subarachnoid
            intraparenchymal
            intraventricular

        becomes:

            image_id = 1
            signature =
            normal+epidural+subdural+subarachnoid+
            intraparenchymal+intraventricular
    """

    # --------------------------------------------------------
    # Create one signature per image ID.
    #
    # IMPORTANT:
    # groupby().apply() produces exactly one value per ID.
    # --------------------------------------------------------

    class_signature = (
        base_df
        .groupby("image_id")["class_name"]
        .apply(
            lambda values: "+".join(
                sorted(
                    set(values),
                    key=CLASS_ORDER.index,
                )
            )
        )
        .rename("signature")
    )

    # --------------------------------------------------------
    # Convert Series to DataFrame.
    #
    # This guarantees exactly ONE row per image_id.
    # --------------------------------------------------------

    id_df = (
        class_signature
        .reset_index()
        .sort_values("image_id")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if id_df["image_id"].duplicated().any():
        raise RuntimeError(
            "Internal error: duplicate image IDs remain "
            "in the ID-level table."
        )

    # --------------------------------------------------------
    # Print signature distribution
    # --------------------------------------------------------

    print()
    print("Image-ID signature distribution:")
    print(
        id_df["signature"]
        .value_counts()
        .to_string()
    )

    return id_df


# ============================================================
# CREATE GROUPED SPLIT
# ============================================================

def create_grouped_split(base_df):
    """
    Create a leakage-safe 70/15/15 split.

    The split is performed at IMAGE-ID level.

    Therefore:
        all base images belonging to one ID
        stay in the same split.
    """

    # --------------------------------------------------------
    # ONE row per image ID
    # --------------------------------------------------------

    id_df = create_id_level_table(base_df)

    # --------------------------------------------------------
    # Rare signatures
    # --------------------------------------------------------

    counts = id_df["signature"].value_counts()

    rare = counts[counts < 2].index

    if len(rare) > 0:

        print()
        print("Rare signatures (<2 image IDs):")

        for key in rare:
            print(
                f"  {key}: {counts[key]}"
            )

        id_df["stratification_key"] = (
            id_df["signature"].where(
                ~id_df["signature"].isin(rare),
                other="rare",
            )
        )

    else:

        id_df["stratification_key"] = (
            id_df["signature"]
        )

    # --------------------------------------------------------
    # First split:
    #
    # 70% train
    # 30% temporary
    # --------------------------------------------------------

    train_ids, temp_ids = train_test_split(
        id_df,
        test_size=0.30,
        random_state=SEED,
        stratify=id_df["stratification_key"],
    )

    # --------------------------------------------------------
    # Second split:
    #
    # 15% validation
    # 15% test
    # --------------------------------------------------------

    temp_counts = (
        temp_ids["stratification_key"]
        .value_counts()
    )

    rare_temp = (
        temp_counts[temp_counts < 2]
        .index
    )

    if len(rare_temp) > 0:

        temp_stratification = (
            temp_ids["stratification_key"]
            .where(
                ~temp_ids["stratification_key"].isin(
                    rare_temp
                ),
                other="rare",
            )
        )

    else:

        temp_stratification = (
            temp_ids["stratification_key"]
        )

    validation_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_stratification,
    )

    # --------------------------------------------------------
    # Assign split names
    # --------------------------------------------------------

    train_ids = train_ids.copy()
    validation_ids = validation_ids.copy()
    test_ids = test_ids.copy()

    train_ids["split"] = "train"
    validation_ids["split"] = "validation"
    test_ids["split"] = "test"

    # --------------------------------------------------------
    # Combine split assignments.
    #
    # This table MUST contain exactly ONE row per image_id.
    # --------------------------------------------------------

    split_ids = pd.concat(
        [
            train_ids,
            validation_ids,
            test_ids,
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Critical safety check BEFORE merge
    # --------------------------------------------------------

    if split_ids["image_id"].duplicated().any():

        duplicates = (
            split_ids[
                split_ids["image_id"].duplicated(
                    keep=False
                )
            ]
            .sort_values("image_id")
        )

        raise RuntimeError(
            "Internal error: split assignment contains "
            "duplicate image IDs:\n"
            f"{duplicates.head(20).to_string(index=False)}"
        )

    # --------------------------------------------------------
    # Merge split assignment back onto BASE IMAGES.
    #
    # base_df:
    #     many rows per image_id
    #
    # split_ids:
    #     exactly ONE row per image_id
    #
    # Therefore many_to_one is correct.
    # --------------------------------------------------------

    result = base_df.merge(
        split_ids[
            [
                "image_id",
                "signature",
                "split",
            ]
        ],
        on="image_id",
        how="left",
        validate="many_to_one",
    )

    # --------------------------------------------------------
    # Ensure every base image received a split
    # --------------------------------------------------------

    if result["split"].isna().any():

        missing = (
            result.loc[
                result["split"].isna(),
                "image_id",
            ]
            .drop_duplicates()
            .tolist()
        )

        raise RuntimeError(
            "Some image IDs did not receive a split:\n"
            f"{missing[:20]}"
        )

    return result


# ============================================================
# VALIDATION / SUMMARY
# ============================================================

def print_summary(result):
    print()
    print("=" * 70)
    print("TBI-HemoCT BASE SPLIT SUMMARY")
    print("=" * 70)

    print()
    print(
        "Total unique base images:",
        result["base_file"].nunique(),
    )

    print(
        "Total unique image IDs:",
        result["image_id"].nunique(),
    )

    print()
    print("Base images by split:")

    print(
        result.groupby(
            ["split", "class_name"]
        ).size()
        .unstack(fill_value=0)
        .to_string()
    )

    print()
    print("Image IDs by split:")

    print(
        result.groupby("split")["image_id"]
        .nunique()
        .to_string()
    )

    # --------------------------------------------------------
    # Check image ID uniqueness across splits
    # --------------------------------------------------------

    print()
    print("Leakage checks:")

    id_split_counts = (
        result.groupby("image_id")["split"]
        .nunique()
    )

    leaked_ids = id_split_counts[
        id_split_counts > 1
    ]

    print(
        "IDs appearing in multiple splits:",
        len(leaked_ids),
    )

    if len(leaked_ids) > 0:

        print(
            leaked_ids.head(20)
            .to_string()
        )

        raise RuntimeError(
            "DATA LEAKAGE DETECTED."
        )

    # --------------------------------------------------------
    # Check duplicate base files
    # --------------------------------------------------------

    duplicate_base_files = (
        result["base_file"]
        .duplicated()
        .sum()
    )

    print(
        "Duplicate base files:",
        duplicate_base_files,
    )

    if duplicate_base_files > 0:

        raise RuntimeError(
            "Duplicate base files detected."
        )

    # --------------------------------------------------------
    # Check every ID has exactly one split
    # --------------------------------------------------------

    split_counts = (
        result.groupby("image_id")["split"]
        .nunique()
    )

    invalid_ids = split_counts[
        split_counts != 1
    ]

    print(
        "IDs without exactly one split:",
        len(invalid_ids),
    )

    if len(invalid_ids) > 0:

        raise RuntimeError(
            "Invalid split assignment detected."
        )

    print()
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TBI-HemoCT BASE-IMAGE / LEAKAGE-SAFE SPLIT")
    print("=" * 70)

    print()
    print("Dataset:")
    print(TBIHEMOCT_DIR)

    print()
    print("CSV:")
    print(RAW_CSV)

    # --------------------------------------------------------
    # Check CSV
    # --------------------------------------------------------

    if not RAW_CSV.exists():

        raise FileNotFoundError(
            f"CSV not found: {RAW_CSV}"
        )

    # --------------------------------------------------------
    # Read original metadata
    # --------------------------------------------------------

    df = pd.read_csv(RAW_CSV)

    print()
    print("Raw CSV rows:", len(df))

    print(
        "Raw unique filenames:",
        df["File Name"].nunique(),
    )

    # --------------------------------------------------------
    # Build base-image table
    # --------------------------------------------------------

    base_df = build_base_image_table(df)

    print()
    print(
        "Unique base images:",
        base_df["base_file"].nunique(),
    )

    print(
        "Unique image IDs:",
        base_df["image_id"].nunique(),
    )

    # --------------------------------------------------------
    # Base-image class distribution
    # --------------------------------------------------------

    print()
    print("Base-image class distribution:")

    print(
        base_df["class_name"]
        .value_counts()
        .reindex(CLASS_ORDER, fill_value=0)
        .to_string()
    )

    # --------------------------------------------------------
    # Create grouped split
    # --------------------------------------------------------

    result = create_grouped_split(
        base_df
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if result["image_id"].nunique() != 10000:

        print(
            "WARNING: Expected approximately "
            "10,000 unique image IDs."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    result = result.sort_values(
        [
            "split",
            "image_id",
            "class_name",
            "base_file",
        ]
    ).reset_index(drop=True)

    result.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print_summary(result)

    print()
    print("Saved:")
    print(OUTPUT_CSV)

    print()
    print("STATUS: SUCCESS")


if __name__ == "__main__":
    main()