from pathlib import Path
import pandas as pd
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = PROJECT_ROOT / "preprocessing" / "classification" / "tbi6000_final_split.csv"
DATASET_ROOT = PROJECT_ROOT / "datasets" / "TBI-6000"
OUTPUT_DIR = PROJECT_ROOT / "preprocessing" / "classification" / "prepared"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("TBI-6000 CLASSIFICATION PREPARATION")
print("=" * 70)

df = pd.read_csv(INPUT_CSV)

print(f"\nRows: {len(df)}")

# Resolve dataset paths
df["full_path"] = df["path"].apply(
    lambda x: str(DATASET_ROOT / str(x))
)

# Verify files
missing = df.loc[~df["full_path"].apply(os.path.exists)]

print(f"Missing files: {len(missing)}")

if len(missing) > 0:
    print("\nFirst missing files:")
    print(missing["full_path"].head(10).to_string(index=False))
    raise FileNotFoundError("Some TBI-6000 files could not be found.")

# Check splits
print("\nSplit distribution:")
print(df["split"].value_counts().to_string())

# Check classes
print("\nClass distribution:")
print(df["label"].value_counts().to_string())

# Save separate files
for split_name in ["train", "validation", "test"]:
    split_df = df[df["split"] == split_name].copy()

    output_file = OUTPUT_DIR / f"tbi6000_{split_name}.csv"
    split_df.to_csv(output_file, index=False)

    print(f"\nSaved {split_name}: {len(split_df)} rows")
    print(output_file)

print("\n" + "=" * 70)
print("STATUS: TBI-6000 CLASSIFICATION PREPARATION COMPLETE")
print("=" * 70)