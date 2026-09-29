from pathlib import Path
import re
import pandas as pd
from collections import Counter, defaultdict

ROOT = Path(
    r"datasets\TBI-HemoCT\TBI-HemoCT Data set\TBI-HemoCT Data set"
)

FOLDERS = [
    "Any(normal)",
    "Epidural",
    "Intraparenchymal",
    "Intraventricular",
    "Subarachnoid",
    "Subdural",
]

OUTPUT_DIR = Path("dataset_check")
OUTPUT_DIR.mkdir(exist_ok=True)

rows = []

for folder in FOLDERS:
    folder_path = ROOT / folder

    for path in folder_path.glob("*.png"):
        name = path.name

        # Extract numeric ID
        id_match = re.match(r"ID_(\d+)_", name)
        image_id = int(id_match.group(1)) if id_match else None

        # Determine whether this is an augmentation
        aug_match = re.search(r"_aug_(\d+)\.png$", name)
        is_augmented = aug_match is not None
        aug_id = int(aug_match.group(1)) if aug_match else None

        # Remove ID and augmentation suffix
        stem = path.stem
        base = re.sub(r"^ID_\d+_", "", stem)
        base = re.sub(r"_aug_\d+$", "", base)

        # Determine UNIQUE marker
        is_unique = "_UNIQUE" in base

        # Remove UNIQUE marker
        label_text = base.replace("_UNIQUE", "")

        # Normalize spacing
        label_text = label_text.strip()

        rows.append({
            "folder": folder,
            "filename": name,
            "image_id": image_id,
            "is_unique": is_unique,
            "is_augmented": is_augmented,
            "augmentation_id": aug_id,
            "filename_label_text": label_text,
        })

df = pd.DataFrame(rows)

print("=" * 70)
print("TBI-HemoCT FILENAME AUDIT")
print("=" * 70)

print(f"Dataset root: {ROOT}")
print(f"Total PNG files: {len(df):,}")

print("\nFILES BY FOLDER")
print("-" * 70)
print(df.groupby("folder").size().to_string())

print("\nUNIQUE / AUGMENTED BREAKDOWN")
print("-" * 70)
print(
    df.groupby(["folder", "is_unique", "is_augmented"])
      .size()
      .to_string()
)

print("\nUNIQUE VS AUGMENTED TOTAL")
print("-" * 70)
print(f"Unique/base-style files: {df['is_unique'].sum():,}")
print(f"Augmented files: {df['is_augmented'].sum():,}")

print("\nMISSING IMAGE IDs")
print("-" * 70)
missing_ids = df["image_id"].isna().sum()
print(f"Files without recognizable ID_<number>: {missing_ids:,}")

print("\nNUMBER OF DISTINCT IMAGE IDs BY FOLDER")
print("-" * 70)
print(
    df.groupby("folder")["image_id"]
      .nunique()
      .to_string()
)

print("\nFILENAME LABEL PATTERNS")
print("-" * 70)

for folder in FOLDERS:
    print(f"\n[{folder}]")

    counts = (
        df.loc[df["folder"] == folder, "filename_label_text"]
        .value_counts()
        .head(20)
    )

    for label, count in counts.items():
        print(f"{count:6d}  {label}")

print("\nIDs APPEARING IN MULTIPLE FOLDERS")
print("-" * 70)

id_folder_counts = (
    df.groupby("image_id")["folder"]
      .nunique()
)

multi_folder_ids = id_folder_counts[id_folder_counts > 1]

print(f"IDs appearing in >1 folder: {len(multi_folder_ids):,}")

if len(multi_folder_ids) > 0:
    for image_id in multi_folder_ids.index[:50]:
        folders = sorted(
            df.loc[df["image_id"] == image_id, "folder"].unique()
        )
        print(f"ID_{image_id}: {folders}")

print("\nAUGMENTATION COUNTS PER IMAGE ID")
print("-" * 70)

base_counts = (
    df.groupby(["folder", "image_id"])
      .agg(
          total_files=("filename", "size"),
          augmented_files=("is_augmented", "sum"),
          unique_files=("is_unique", "sum"),
      )
      .reset_index()
)

print("Summary:")
print(base_counts["augmented_files"].describe().to_string())

print("\nTOP 30 IDs BY NUMBER OF AUGMENTED FILES")
print("-" * 70)

top_aug = base_counts.sort_values(
    "augmented_files",
    ascending=False
).head(30)

print(top_aug.to_string(index=False))

# Save complete metadata
csv_path = OUTPUT_DIR / "tbihemoct_filename_metadata.csv"
df.to_csv(csv_path, index=False)

# Save per-ID summary
id_summary_path = OUTPUT_DIR / "tbihemoct_id_summary.csv"
base_counts.to_csv(id_summary_path, index=False)

print("\nOUTPUT FILES")
print("-" * 70)
print(csv_path)
print(id_summary_path)

print("\nAUDIT COMPLETE")