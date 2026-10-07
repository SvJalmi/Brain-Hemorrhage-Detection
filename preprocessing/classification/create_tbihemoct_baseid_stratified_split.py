import pandas as pd
import numpy as np
import os

INPUT = r"preprocessing\classification\tbihemoct_labels.csv"
OUTPUT = r"preprocessing\classification\tbihemoct_baseid_stratified_split.csv"

RANDOM_SEED = 42

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

print("=" * 70)
print("TBI-HemoCT BASE-ID STRATIFIED SPLIT")
print("=" * 70)

df = pd.read_csv(INPUT)

# ---------------------------------------------------------
# Create base ID if it does not already exist
# ---------------------------------------------------------

if "base_id" not in df.columns:
    df["base_id"] = df["image_path"].str.extract(r"(ID_\d+)")[0]

if df["base_id"].isna().any():
    raise RuntimeError("Some rows have missing base_id values.")

print(f"Total rows: {len(df)}")
print(f"Unique base IDs: {df['base_id'].nunique()}")

# ---------------------------------------------------------
# Verify every base ID stays intact
# ---------------------------------------------------------

cols = [
    "epidural",
    "subdural",
    "subarachnoid",
    "intraparenchymal",
    "intraventricular"
]

missing = [c for c in cols if c not in df.columns]

if missing:
    raise RuntimeError(f"Missing label columns: {missing}")

# ---------------------------------------------------------
# Build one row per base ID
# ---------------------------------------------------------

groups = (
    df.groupby("base_id")[cols]
      .max()
      .reset_index()
)

groups["n_images"] = (
    df.groupby("base_id")
      .size()
      .reindex(groups["base_id"])
      .values
)

# Epidural is the only label that varies at base-ID level.
groups["epidural_group"] = groups["epidural"].astype(int)

# ---------------------------------------------------------
# Shuffle reproducibly
# ---------------------------------------------------------

rng = np.random.default_rng(RANDOM_SEED)

groups = groups.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)

# ---------------------------------------------------------
# Stratify epidural-containing IDs
# ---------------------------------------------------------

train_ids = []
val_ids = []
test_ids = []

for epidural_value in [0, 1]:

    subset = groups[
        groups["epidural_group"] == epidural_value
    ].copy()

    subset = subset.sample(
        frac=1,
        random_state=RANDOM_SEED + epidural_value
    ).reset_index(drop=True)

    n = len(subset)

    n_train = int(round(n * TRAIN_FRAC))
    n_val = int(round(n * VAL_FRAC))

    train_part = subset.iloc[:n_train]
    val_part = subset.iloc[n_train:n_train + n_val]
    test_part = subset.iloc[n_train + n_val:]

    train_ids.extend(train_part["base_id"])
    val_ids.extend(val_part["base_id"])
    test_ids.extend(test_part["base_id"])

# ---------------------------------------------------------
# Assign split
# ---------------------------------------------------------

split_map = {}

for x in train_ids:
    split_map[x] = "train"

for x in val_ids:
    split_map[x] = "validation"

for x in test_ids:
    split_map[x] = "test"

df["split"] = df["base_id"].map(split_map)

if df["split"].isna().any():
    raise RuntimeError("Some rows were not assigned to a split.")

# ---------------------------------------------------------
# Leakage checks
# ---------------------------------------------------------

base_split_counts = (
    df.groupby("base_id")["split"]
      .nunique()
)

base_leakage = int((base_split_counts > 1).sum())

# ---------------------------------------------------------
# Print results
# ---------------------------------------------------------

print()
print("=" * 70)
print("IMAGE COUNTS")
print("=" * 70)

print(df["split"].value_counts().sort_index())

print()
print("=" * 70)
print("UNIQUE BASE IDs")
print("=" * 70)

print(
    df.groupby("split")["base_id"]
      .nunique()
      .sort_index()
)

print()
print("=" * 70)
print("CLASS × SPLIT")
print("=" * 70)

print(
    pd.crosstab(
        df["split"],
        df[cols].sum(axis=1)
    )
)

print()
print("=" * 70)
print("EPIDURAL BASE-ID DISTRIBUTION")
print("=" * 70)

epidural_distribution = (
    df.groupby("split")
      .apply(
          lambda x: x.groupby("base_id")["epidural"].max().sum(),
          include_groups=False
      )
)

print(epidural_distribution)

print()
print("=" * 70)
print("LEAKAGE CHECK")
print("=" * 70)

print("Base IDs crossing splits:", base_leakage)

if base_leakage != 0:
    raise RuntimeError("BASE-ID LEAKAGE DETECTED.")

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

df.to_csv(OUTPUT, index=False)

print()
print("Saved:")
print(OUTPUT)

print()
print("=" * 70)
print("STATUS: SUCCESS - BASE-ID SPLIT IS LEAKAGE-SAFE")
print("=" * 70)