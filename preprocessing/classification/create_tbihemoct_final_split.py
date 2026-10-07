import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

INPUT = Path(r"preprocessing\classification\tbihemoct_labels.csv")
OUTPUT = Path(r"preprocessing\classification\tbihemoct_final_split.csv")

df = pd.read_csv(INPUT)

# Extract case/base ID from filename
df["base_id"] = df["image_path"].str.extract(r"(ID_\d+)")[0]

if df["base_id"].isna().any():
    print("WARNING: Some rows have no base_id")
    print(df[df["base_id"].isna()].head())

print("=" * 70)
print("TBI-HemoCT LEAKAGE-SAFE GROUP SPLIT")
print("=" * 70)
print("Total rows:", len(df))
print("Unique base IDs:", df["base_id"].nunique())

# We split by base_id so all images belonging to one CT/case
# stay in exactly one split.

groups = df["base_id"]

# First: 70% train, 30% temporary
gss1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=42
)

train_idx, temp_idx = next(
    gss1.split(df, groups=groups)
)

train_df = df.iloc[train_idx].copy()
temp_df = df.iloc[temp_idx].copy()

# Second: split temporary 50/50 -> validation/test
gss2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=42
)

val_rel, test_rel = next(
    gss2.split(temp_df, groups=temp_df["base_id"])
)

val_df = temp_df.iloc[val_rel].copy()
test_df = temp_df.iloc[test_rel].copy()

train_df["split"] = "train"
val_df["split"] = "validation"
test_df["split"] = "test"

final_df = pd.concat(
    [train_df, val_df, test_df],
    ignore_index=True
)

# ------------------------------------------------------------
# Leakage checks
# ------------------------------------------------------------

split_counts = final_df.groupby("base_id")["split"].nunique()

leaking_base_ids = split_counts[split_counts > 1]

print()
print("=" * 70)
print("IMAGE COUNTS")
print("=" * 70)
print(final_df["split"].value_counts())

print()
print("=" * 70)
print("UNIQUE BASE IDs BY SPLIT")
print("=" * 70)
print(
    final_df.groupby("split")["base_id"]
    .nunique()
    .sort_index()
)

print()
print("=" * 70)
print("LABEL COUNTS")
print("=" * 70)

print(
    final_df.groupby("split")[
        [
            "epidural",
            "subdural",
            "subarachnoid",
            "intraparenchymal",
            "intraventricular",
        ]
    ].sum()
)

print()
print("=" * 70)
print("NORMAL COUNTS")
print("=" * 70)

print(
    final_df.groupby("split")["is_normal"].sum()
)

print()
print("=" * 70)
print("GROUP LEAKAGE CHECK")
print("=" * 70)

print("Base IDs crossing splits:", len(leaking_base_ids))

if len(leaking_base_ids) > 0:
    print("ERROR: GROUP LEAKAGE STILL EXISTS")
    print(leaking_base_ids.head(20))
else:
    print("SUCCESS: No base ID crosses train/validation/test")

# Save
final_df.to_csv(OUTPUT, index=False)

print()
print("Saved:")
print(OUTPUT)

print()
print("STATUS: TBI-HemoCT GROUP SPLIT CREATED")