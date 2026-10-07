import os
import pandas as pd


INPUT = r"preprocessing\classification\tbihemoct_final_leakage_safe.csv"

OUTPUT_DIR = r"preprocessing\classification\prepared"

LABEL_COLS = [
    "epidural",
    "subdural",
    "subarachnoid",
    "intraparenchymal",
    "intraventricular",
]


print("=" * 70)
print("TBI-HemoCT PREPARATION")
print("=" * 70)


# ------------------------------------------------------------
# Load
# ------------------------------------------------------------

df = pd.read_csv(INPUT)

print("Rows:", len(df))


# ------------------------------------------------------------
# Required columns
# ------------------------------------------------------------

required = [
    "image_path",
    "base_id",
    "split",
    "is_normal",
] + LABEL_COLS

missing = [c for c in required if c not in df.columns]

if missing:
    raise RuntimeError(f"Missing columns: {missing}")


# ------------------------------------------------------------
# Check paths
# ------------------------------------------------------------

print()
print("Checking image paths...")

missing_files = ~df["image_path"].map(os.path.isfile)

print("Missing files:", int(missing_files.sum()))

if missing_files.any():
    print(df.loc[missing_files, "image_path"].head(20))
    raise RuntimeError("Missing image files detected.")


# ------------------------------------------------------------
# Check labels
# ------------------------------------------------------------

print()
print("Checking labels...")

for col in LABEL_COLS:

    values = set(df[col].dropna().unique())

    if not values.issubset({0, 1}):
        raise RuntimeError(
            f"Invalid values in {col}: {values}"
        )


# ------------------------------------------------------------
# Create multi-label target
# ------------------------------------------------------------

df["label_vector"] = df[LABEL_COLS].astype(int).astype(str).agg(
    ",".join,
    axis=1
)


# ------------------------------------------------------------
# Verify normal images
# ------------------------------------------------------------

label_sum = df[LABEL_COLS].sum(axis=1)

normal_mismatch = (
    ((label_sum == 0) & (df["is_normal"] != 1))
    |
    ((label_sum > 0) & (df["is_normal"] != 0))
)

print()
print("Normal-label consistency errors:",
      int(normal_mismatch.sum()))

if normal_mismatch.any():
    raise RuntimeError(
        "is_normal does not agree with hemorrhage labels."
    )


# ------------------------------------------------------------
# Split verification
# ------------------------------------------------------------

print()
print("Split distribution:")

print(df["split"].value_counts())


allowed_splits = {"train", "validation", "test"}

if not set(df["split"].unique()).issubset(allowed_splits):
    raise RuntimeError("Unexpected split detected.")


# ------------------------------------------------------------
# Leakage verification
# ------------------------------------------------------------

print()
print("Checking base-ID leakage...")

base_split_counts = df.groupby("base_id")["split"].nunique()

base_leakage = int((base_split_counts > 1).sum())

print("Base IDs crossing splits:", base_leakage)

if base_leakage != 0:
    raise RuntimeError("Base-ID leakage detected.")


# ------------------------------------------------------------
# Create output directory
# ------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# Save split-specific CSV files
# ------------------------------------------------------------

for split in ["train", "validation", "test"]:

    subset = df[df["split"] == split].copy()

    output = os.path.join(
        OUTPUT_DIR,
        f"{split}.csv"
    )

    subset.to_csv(output, index=False)

    print(
        f"Saved {split}: "
        f"{len(subset)} rows -> {output}"
    )


# ------------------------------------------------------------
# Print label statistics
# ------------------------------------------------------------

print()
print("=" * 70)
print("LABEL DISTRIBUTION")
print("=" * 70)

print(
    df.groupby("split")[LABEL_COLS].sum()
)


print()
print("Images by number of positive labels:")

print(
    df[LABEL_COLS]
    .sum(axis=1)
    .value_counts()
    .sort_index()
)


# ------------------------------------------------------------
# Final status
# ------------------------------------------------------------

print()
print("=" * 70)
print("STATUS: PREPROCESSING METADATA READY")
print("=" * 70)