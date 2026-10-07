from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "preprocessing"
    / "segmentation"
    / "mbh_seg25_manifest.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "preprocessing"
    / "segmentation"
    / "mbh_seg25_split.csv"
)

RANDOM_STATE = 42


print("=" * 70)
print("MBH-SEG25 CASE-LEVEL SPLIT")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"\nTotal cases: {len(df)}")

# First split: 70% train, 30% temporary
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=RANDOM_STATE,
    shuffle=True,
)

# Second split: half validation, half test
validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_STATE,
    shuffle=True,
)

train_df = train_df.copy()
validation_df = validation_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"
validation_df["split"] = "validation"
test_df["split"] = "test"

final_df = pd.concat(
    [train_df, validation_df, test_df],
    ignore_index=True,
)

# Verify every case appears exactly once
split_counts = final_df["case_id"].value_counts()

if (split_counts != 1).any():
    raise RuntimeError(
        "A case ID appears in more than one split."
    )

# Save
final_df.to_csv(
    OUTPUT_FILE,
    index=False,
)

print("\nSplit distribution:")
print(
    final_df["split"]
    .value_counts()
    .to_string()
)

print("\nAnnotation-count distribution by split:")
print(
    pd.crosstab(
        final_df["split"],
        final_df["annotation_count"],
    ).to_string()
)

print("\nUnique case IDs:", final_df["case_id"].nunique())
print(
    "Case IDs appearing in multiple splits:",
    (split_counts > 1).sum()
)

print("\nSaved:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("STATUS: MBH-SEG25 CASE-LEVEL SPLIT COMPLETE")
print("=" * 70)