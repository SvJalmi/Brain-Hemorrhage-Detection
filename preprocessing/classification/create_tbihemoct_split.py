from pathlib import Path
import re
import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing.common.config import (
    TBIHEMOCT_DIR,
)

# ============================================================
# TBI-HemoCT GROUPED / LEAKAGE-SAFE SPLIT
# ============================================================

SEED = 42

RAW_CSV = (
    TBIHEMOCT_DIR
    / "TBI-HemoCT Data set"
    / "TBI-HemoCT Data set"
    / "final_log_file.csv"
)

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = OUTPUT_DIR / "tbihemoct_split.csv"

LABELS = [
    "epidural",
    "subdural",
    "subarachnoid",
    "intraparenchymal",
    "intraventricular",
]


def extract_image_id(filename):
    """
    Extract numeric image ID from filenames such as:
    ID_123_subdural_UNIQUE.png
    ID_123_subdural_aug7.png
    """
    match = re.search(r"ID_(\d+)", str(filename))

    if match:
        return int(match.group(1))

    return None


def parse_positive_locations(value):
    """
    Convert the 'All positive locations' field into
    a set of standardized hemorrhage labels.

    Example:
        intraparenchymal- intraventricular- subdural

    becomes:
        {'intraparenchymal', 'intraventricular', 'subdural'}
    """

    if pd.isna(value):
        return set()

    text = str(value).lower()

    found = set()

    for label in LABELS:
        if label in text:
            found.add(label)

    return found

def build_id_level_table(df):
    """
    Build exactly one record per unique image ID.

    All files/rows belonging to the same image ID are combined.

    TBI-HemoCT is treated as a multi-label dataset:
        - normal
        - epidural
        - subdural
        - subarachnoid
        - intraparenchymal
        - intraventricular

    A case is considered NORMAL only when:
        1. Label == 0 for all rows belonging to the ID, and
        2. no hemorrhage location is present in the metadata.

    For positive cases, hemorrhage labels are collected from
    "All positive locations".
    """

    records = []

    grouped = df.groupby("image_id", sort=True)

    for image_id, group in grouped:

        label_set = set()

        # ----------------------------------------------------
        # Collect hemorrhage locations from metadata.
        # ----------------------------------------------------

        for value in group["All positive locations"]:
            label_set.update(
                parse_positive_locations(value)
            )

        # ----------------------------------------------------
        # Use Destination as an additional source of labels.
        # Ignore "normal" as a hemorrhage label.
        # ----------------------------------------------------

        for value in group["Destination"]:

            if pd.isna(value):
                continue

            destination = str(value).strip().lower()

            if destination in LABELS:
                label_set.add(destination)

        # ----------------------------------------------------
        # Determine whether this image ID is normal.
        #
        # Label == 0 is used together with the absence of
        # hemorrhage labels.
        # ----------------------------------------------------

        has_positive_label = (
            pd.to_numeric(
                group["Label"],
                errors="coerce"
            )
            .fillna(0)
            .eq(1)
            .any()
        )

        is_normal = (
            not has_positive_label
            and len(label_set) == 0
        )

        # ----------------------------------------------------
        # Safety check:
        # A normal case must not have hemorrhage labels.
        # ----------------------------------------------------

        if is_normal and len(label_set) > 0:
            raise RuntimeError(
                f"Label inconsistency for image ID {image_id}: "
                f"case marked normal but hemorrhage labels were found: "
                f"{sorted(label_set)}"
            )

        record = {
            "image_id": image_id,
            "normal": int(is_normal),
        }

        # ----------------------------------------------------
        # Multi-label hemorrhage indicators.
        # ----------------------------------------------------

        for label in LABELS:
            record[label] = int(label in label_set)

        # ----------------------------------------------------
        # Metadata row count.
        # ----------------------------------------------------

        record["source_row_count"] = len(group)

        records.append(record)

    result = pd.DataFrame(records)

    return result


def create_stratification_key(df):
    """
    Create a deterministic multi-label signature.

    This is used only to help distribute similar cases
    across train/validation/test.

    Examples:
        normal
        epidural
        subdural
        intraparenchymal+intraventricular
        subarachnoid+subdural
    """

    keys = []

    for _, row in df.iterrows():

        positive = [
            label
            for label in LABELS
            if int(row[label]) == 1
        ]

        if not positive:
            keys.append("normal")
        else:
            keys.append("+".join(positive))

    return pd.Series(keys, index=df.index)


def safe_stratified_split(df):
    """
    Perform 70/15/15 grouped split.

    The grouping has already happened at image-ID level,
    so no image ID can appear in more than one split.
    """

    strat_key = create_stratification_key(df)

    # --------------------------------------------------------
    # Some very rare multi-label combinations may occur only
    # once. They cannot be stratified directly.
    # --------------------------------------------------------

    counts = strat_key.value_counts()

    rare_keys = counts[counts < 2].index

    if len(rare_keys) > 0:

        print()
        print("Rare label combinations (<2 cases):")

        for key in rare_keys:
            print(f"  {key}: {counts[key]}")

        # Put rare combinations into a generic stratum so
        # train_test_split can operate safely.
        strat_key = strat_key.where(
            ~strat_key.isin(rare_keys),
            other="rare",
        )

    # --------------------------------------------------------
    # First split:
    # 70% train
    # 30% temporary
    # --------------------------------------------------------

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=SEED,
        stratify=strat_key,
    )

    # --------------------------------------------------------
    # Second split:
    # 15% validation
    # 15% test
    # --------------------------------------------------------

    temp_key = create_stratification_key(temp_df)

    temp_counts = temp_key.value_counts()

    rare_temp_keys = temp_counts[temp_counts < 2].index

    if len(rare_temp_keys) > 0:
        temp_key = temp_key.where(
            ~temp_key.isin(rare_temp_keys),
            other="rare",
        )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_key,
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "validation"
    test_df["split"] = "test"

    result = pd.concat(
        [train_df, val_df, test_df],
        ignore_index=True,
    )

    # Sort for reproducibility
    result = result.sort_values(
        ["split", "image_id"]
    ).reset_index(drop=True)

    return result


def print_summary(df):
    print()
    print("=" * 70)
    print("TBI-HemoCT SPLIT SUMMARY")
    print("=" * 70)

    print()
    print("Total unique image IDs:", len(df))

    print()
    print("Split counts:")
    print(df["split"].value_counts())

    print()
    print("Expected approximate:")
    print("  Train      : 7000")
    print("  Validation : 1500")
    print("  Test       : 1500")

    print()
    print("Label distribution by split:")

    for split in ["train", "validation", "test"]:

        subset = df[df["split"] == split]

        print()
        print(f"[{split.upper()}]")

        for label in ["normal"] + LABELS:
            print(
                f"  {label:18s}: "
                f"{int(subset[label].sum())}"
            )

    print()
    print("Image-ID uniqueness check:")

    duplicated_ids = df[
        df["image_id"].duplicated(keep=False)
    ]

    print(
        "Duplicate IDs inside split table:",
        len(duplicated_ids),
    )

    # Check whether any ID has more than one split
    split_counts = (
        df.groupby("image_id")["split"]
        .nunique()
    )

    leaked_ids = split_counts[
        split_counts > 1
    ]

    print(
        "IDs appearing in multiple splits:",
        len(leaked_ids),
    )

    print("=" * 70)


def main():

    print("=" * 70)
    print("TBI-HemoCT GROUPED / LEAKAGE-SAFE SPLIT")
    print("=" * 70)

    print()
    print("Dataset:")
    print(TBIHEMOCT_DIR)

    print()
    print("CSV:")
    print(RAW_CSV)

    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"CSV not found: {RAW_CSV}"
        )

    # --------------------------------------------------------
    # Read original metadata
    # --------------------------------------------------------

    df = pd.read_csv(RAW_CSV)

    print()
    print("CSV rows:", len(df))

    print("CSV columns:")
    for column in df.columns:
        print(" ", column)

    # --------------------------------------------------------
    # Extract image IDs
    # --------------------------------------------------------

    df["image_id"] = df["File Name"].apply(
        extract_image_id
    )

    missing_ids = df["image_id"].isna().sum()

    print()
    print("Rows with missing image ID:", missing_ids)

    if missing_ids > 0:
        raise ValueError(
            "Some CSV rows do not contain a valid image ID."
        )

    # --------------------------------------------------------
    # Build one record per image ID
    # --------------------------------------------------------

    id_df = build_id_level_table(df)

    print()
    print("Unique image IDs:", len(id_df))

    if len(id_df) != 10000:
        print(
            "WARNING: Expected approximately 10,000 "
            "unique image IDs based on the audit."
        )

    # --------------------------------------------------------
    # Create grouped split
    # --------------------------------------------------------

    result = safe_stratified_split(id_df)

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if result["image_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate image IDs detected."
        )

    split_per_id = (
        result.groupby("image_id")["split"]
        .nunique()
    )

    if (split_per_id > 1).any():
        raise RuntimeError(
            "DATA LEAKAGE: an image ID occurs in "
            "multiple splits."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    result.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print_summary(result)

    print()
    print("Saved:")
    print(OUTPUT_CSV)

    print()
    print("STATUS: SUCCESS")


if __name__ == "__main__":
    main()