from pathlib import Path
import re
import pandas as pd
from collections import defaultdict

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

        # Extract ID as a STRING, not a number.
        match = re.match(r"^ID_(\d+)_", name)

        if match:
            image_id = match.group(1)
        else:
            image_id = None

        # Augmentation marker
        aug_match = re.search(r"_aug_(\d+)\.png$", name)
        is_augmented = aug_match is not None
        augmentation_id = aug_match.group(1) if aug_match else None

        # Remove ID
        stem = path.stem

        if image_id is not None:
            remainder = stem[len(f"ID_{image_id}_"):]
        else:
            remainder = stem

        # Remove augmentation suffix
        remainder = re.sub(r"_aug_\d+$", "", remainder)

        # Determine UNIQUE marker
        is_unique = "_UNIQUE" in remainder

        # Remove UNIQUE marker
        label_text = remainder.replace("_UNIQUE", "").strip()

        rows.append({
            "folder": folder,
            "filename": name,
            "image_id": image_id,
            "is_augmented": is_augmented,
            "augmentation_id": augmentation_id,
            "is_unique": is_unique,
            "filename_label_text": label_text,
        })

df = pd.DataFrame(rows)

print("=" * 75)
print("TBI-HemoCT CORRECTED ID / FILENAME AUDIT")
print("=" * 75)

print(f"Total PNG files: {len(df):,}")

# ------------------------------------------------------------
# Missing / malformed IDs
# ------------------------------------------------------------

print("\nFILES WITHOUT VALID ID")
print("-" * 75)

missing = df[df["image_id"].isna()]

print(f"Count: {len(missing):,}")

if len(missing):
    for name in missing["filename"]:
        print(name)

# ------------------------------------------------------------
# Distinct IDs
# ------------------------------------------------------------

valid = df[df["image_id"].notna()].copy()

print("\nDISTINCT IDS")
print("-" * 75)

print(f"Distinct IDs across all folders: {valid['image_id'].nunique():,}")

for folder in FOLDERS:
    n = valid.loc[valid["folder"] == folder, "image_id"].nunique()
    print(f"{folder}: {n:,}")

# ------------------------------------------------------------
# Folder coverage per ID
# ------------------------------------------------------------

id_folder = (
    valid.groupby("image_id")["folder"]
    .nunique()
)

print("\nID COVERAGE ACROSS FOLDERS")
print("-" * 75)

for n in sorted(id_folder.unique()):
    count = (id_folder == n).sum()
    print(f"IDs appearing in exactly {n} folder(s): {count:,}")

all_six = id_folder[id_folder == len(FOLDERS)]

print(f"\nIDs appearing in ALL SIX folders: {len(all_six):,}")

# ------------------------------------------------------------
# IDs and their folders
# ------------------------------------------------------------

print("\nSAMPLE IDS WITH THEIR FOLDERS")
print("-" * 75)

for image_id in sorted(valid["image_id"].unique(), key=int)[:30]:
    folders = sorted(
        valid.loc[valid["image_id"] == image_id, "folder"].unique()
    )
    print(f"ID_{image_id}: {', '.join(folders)}")

# ------------------------------------------------------------
# Label combinations by folder
# ------------------------------------------------------------

print("\nLABEL PATTERNS BY FOLDER")
print("-" * 75)

for folder in FOLDERS:
    print(f"\n[{folder}]")

    subset = valid[valid["folder"] == folder]

    counts = (
        subset["filename_label_text"]
        .value_counts()
    )

    for label, count in counts.items():
        print(f"{count:6d}  {label}")

# ------------------------------------------------------------
# Label consistency for each ID
# ------------------------------------------------------------

print("\nIDs WITH MULTIPLE FILENAME LABEL PATTERNS")
print("-" * 75)

label_counts = (
    valid.groupby("image_id")["filename_label_text"]
    .nunique()
)

inconsistent = label_counts[label_counts > 1]

print(f"IDs with >1 filename label pattern: {len(inconsistent):,}")

for image_id in sorted(inconsistent.index, key=int)[:50]:
    subset = valid[valid["image_id"] == image_id]

    print(f"\nID_{image_id}")

    for _, row in subset.iterrows():
        print(
            f"  {row['folder']}: "
            f"{row['filename_label_text']} | "
            f"{row['filename']}"
        )

# ------------------------------------------------------------
# Augmentation statistics
# ------------------------------------------------------------

print("\nAUGMENTATION")
print("-" * 75)

print(f"Total augmented files: {df['is_augmented'].sum():,}")
print(f"Total non-augmented files: {(~df['is_augmented']).sum():,}")

print("\nAugmented files by folder:")

aug_by_folder = (
    df.groupby("folder")["is_augmented"]
    .sum()
)

for folder in FOLDERS:
    print(f"{folder}: {int(aug_by_folder.get(folder, 0)):,}")

# ------------------------------------------------------------
# Per-ID counts
# ------------------------------------------------------------

id_summary = (
    valid.groupby(["folder", "image_id"])
    .agg(
        total_files=("filename", "size"),
        augmented_files=("is_augmented", "sum"),
        unique_marked_files=("is_unique", "sum"),
        label_patterns=("filename_label_text", "nunique"),
    )
    .reset_index()
)

print("\nPER-ID FILE COUNT SUMMARY")
print("-" * 75)

print(id_summary["total_files"].describe().to_string())

print("\nMaximum files belonging to one ID within one folder:")

max_row = id_summary.loc[
    id_summary["total_files"].idxmax()
]

print(max_row.to_string())

# ------------------------------------------------------------
# Duplicate filenames
# ------------------------------------------------------------

print("\nDUPLICATE FILENAMES")
print("-" * 75)

duplicate_names = (
    df.groupby("filename")
    .size()
)

duplicate_names = duplicate_names[duplicate_names > 1]

print(f"Filename strings appearing more than once: {len(duplicate_names):,}")

if len(duplicate_names):
    print("\nFirst 30:")
    for name, count in duplicate_names.head(30).items():
        print(f"{count}x  {name}")

# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------

metadata_path = OUTPUT_DIR / "tbihemoct_filename_metadata_corrected.csv"
summary_path = OUTPUT_DIR / "tbihemoct_id_summary_corrected.csv"

df.to_csv(metadata_path, index=False)
id_summary.to_csv(summary_path, index=False)

print("\nOUTPUT FILES")
print("-" * 75)
print(metadata_path)
print(summary_path)

print("\nAUDIT COMPLETE")