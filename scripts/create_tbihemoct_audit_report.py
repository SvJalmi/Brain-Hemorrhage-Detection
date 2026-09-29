from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(r"C:\Users\shrey\OneDrive\Desktop\Brain_Hemorrhage_Project")
CHECK_DIR = PROJECT_ROOT / "dataset_check"

CSV_PATH = PROJECT_ROOT / "datasets" / "TBI-HemoCT" / "TBI-HemoCT Data set" / "TBI-HemoCT Data set" / "final_log_file.csv"
ID_SUMMARY_PATH = CHECK_DIR / "tbihemoct_id_summary_corrected.csv"
INTEGRITY_REPORT = CHECK_DIR / "TBIHemoCT_IMAGE_INTEGRITY_REPORT.txt"

OUT_PATH = CHECK_DIR / "TBIHemoCT_MASTER_AUDIT_REPORT.txt"

lines = []

def add(text=""):
    lines.append(text)

add("TBI-HemoCT MASTER DATASET AUDIT REPORT")
add("=" * 70)
add("")
add("Dataset:")
add(r"datasets\TBI-HemoCT")
add("")
add("This report consolidates the filename/CSV audit, image-ID audit,")
add("and full image-integrity audit. The raw dataset was not modified.")
add("")

# ------------------------------------------------------------
# File / CSV audit
# ------------------------------------------------------------

add("1. DATASET STRUCTURE AND FILE COUNTS")
add("-" * 70)

root = PROJECT_ROOT / "datasets" / "TBI-HemoCT" / "TBI-HemoCT Data set" / "TBI-HemoCT Data set"

folders = [
    "Any(normal)",
    "Epidural",
    "Intraparenchymal",
    "Intraventricular",
    "Subarachnoid",
    "Subdural",
]

total_disk = 0

for folder in folders:
    count = len(list((root / folder).glob("*.png")))
    total_disk += count
    add(f"{folder}: {count:,} PNG files")

add(f"TOTAL PNG FILES ON DISK: {total_disk:,}")
add("")

# ------------------------------------------------------------
# CSV summary
# ------------------------------------------------------------

df = pd.read_csv(CSV_PATH, keep_default_na=False)

add("2. CSV METADATA SUMMARY")
add("-" * 70)

add(f"CSV rows: {len(df):,}")
add(f"CSV columns: {len(df.columns)}")
add("Columns:")
for col in df.columns:
    add(f"  - {col}")
add("")

add("Label distribution:")
for value, count in df["Label"].value_counts(dropna=False).items():
    add(f"  Label {value}: {count:,}")
add("")

add("Destination distribution:")
for value, count in df["Destination"].value_counts(dropna=False).items():
    add(f"  {value}: {count:,}")
add("")

add("Placed in multiple locations:")
for value, count in df["Placed in multiple locations"].value_counts(dropna=False).items():
    add(f"  {value}: {count:,}")
add("")

# ------------------------------------------------------------
# ID summary
# ------------------------------------------------------------

ids = pd.read_csv(ID_SUMMARY_PATH)

add("3. IMAGE-ID / GROUPING AUDIT")
add("-" * 70)

add(f"Distinct image IDs overall: {ids['image_id'].nunique():,}")
add(f"ID-summary rows: {len(ids):,}")
add("")

add("Folder ID coverage:")
coverage = (
    ids.groupby("folder")["image_id"]
    .nunique()
    .sort_index()
)

for folder, count in coverage.items():
    add(f"  {folder}: {count:,} distinct IDs")

add("")

folder_count_per_id = ids.groupby("image_id")["folder"].nunique()

add("Number of folders represented per image ID:")
for n, count in folder_count_per_id.value_counts().sort_index().items():
    add(f"  {n} folder(s): {count:,} IDs")

add("")

add("Augmented-file counts by folder:")
aug = ids.groupby("folder")["augmented_files"].sum().sort_index()

for folder, count in aug.items():
    add(f"  {folder}: {int(count):,}")

add("")

# ------------------------------------------------------------
# Physical files absent from CSV
# ------------------------------------------------------------

add("4. CSV ↔ DISK RECONCILIATION")
add("-" * 70)

csv_names = set(df["File Name"].astype(str))

disk_names = set()

for folder in folders:
    for p in (root / folder).glob("*.png"):
        disk_names.add(p.name)

missing_from_csv = disk_names - csv_names
missing_on_disk = csv_names - disk_names

add(f"Unique CSV filenames: {len(csv_names):,}")
add(f"Unique disk filenames: {len(disk_names):,}")
add(f"Disk filenames missing from CSV: {len(missing_from_csv):,}")
add(f"CSV filenames missing on disk: {len(missing_on_disk):,}")
add("")

if missing_from_csv:
    add("Disk files absent from CSV by folder:")
    for folder in folders:
        names = [
            p.name
            for p in (root / folder).glob("*.png")
            if p.name in missing_from_csv
        ]
        if names:
            add(f"  {folder}: {len(names):,}")

add("")

add("Known unlogged physical files:")
add("  - Epidural: nan.png (1 file)")
add("  - Intraventricular: 3,242 files")
add("  - Total unlogged physical files: 3,243")
add("")

add("The 3,242 unlogged Intraventricular files were found to be")
add("byte-identical to logged Intraventricular files, indicating")
add("duplicate physical copies rather than additional unique image content.")
add("")

add("The unlogged Epidural nan.png is a valid readable PNG and was")
add("not found to be identical to a logged Epidural file.")
add("")

# ------------------------------------------------------------
# Label / location verification
# ------------------------------------------------------------

add("5. LABEL CONSISTENCY CHECKS")
add("-" * 70)

normal_mask = (
    (df["Label"] == 0) &
    (df["Destination"].astype(str).str.lower() == "normal") &
    (df["All positive locations"].astype(str).str.strip() == "")
)

add(f"Normal rows verified consistently: {normal_mask.sum():,}")
add("Expected normal rows: 10,000")
add("")

positive_rows = df[df["Label"] == 1]

add(f"Positive rows checked: {len(positive_rows):,}")
add("")

add("Filename vs All positive locations:")
add("  One mismatch was found:")
add("  ID_8_subdural_UNIQUE.png")
add("  Filename indicates: subdural")
add("  CSV All positive locations:")
add("  intraparenchymal- intraventricular- subdural")
add("")
add("No manual correction was applied; the raw metadata was preserved.")
add("")

# ------------------------------------------------------------
# Image integrity report
# ------------------------------------------------------------

add("6. FULL IMAGE INTEGRITY AUDIT")
add("-" * 70)

if INTEGRITY_REPORT.exists():
    integrity_text = INTEGRITY_REPORT.read_text(encoding="utf-8")
    add(integrity_text.strip())
else:
    add("WARNING: Image integrity report was not found.")
    add(str(INTEGRITY_REPORT))

add("")
add("7. IMPORTANT ML IMPLICATIONS")
add("-" * 70)

add("1. Do not perform a naive image-level random train/validation/test split.")
add("   The dataset contains repeated filenames, augmentation variants,")
add("   multiple folder placements, and shared image IDs.")
add("")
add("2. Use image ID as a grouping signal when constructing splits.")
add("   All files sharing the same image ID should remain in the same")
add("   split unless a stronger verified patient-level identifier is available.")
add("")
add("3. Do not treat folder membership as proof of independent patients.")
add("   The same image ID occurs across multiple folders.")
add("")
add("4. Do not delete the 3,242 unlogged Intraventricular files merely")
add("   because they are absent from the CSV. They are byte-identical")
add("   duplicates of logged Intraventricular files and are part of")
add("   the original physical dataset.")
add("")
add("5. Do not rename or delete Epidural nan.png at this stage.")
add("   It is a valid PNG but is anomalous with respect to the CSV.")
add("")
add("6. Do not resize, delete, or modify the 32 non-512x512 images merely")
add("   because their dimensions differ. All were readable and passed")
add("   the image-integrity checks.")
add("")
add("7. The raw dataset should remain untouched. Any cleaning or")
add("   preprocessing should be performed in a separate derived dataset.")
add("")

add("8. FINAL AUDIT STATUS")
add("-" * 70)
add("Dataset integrity audit completed.")
add("Raw TBI-HemoCT dataset was not modified.")
add("")
add("Next stage: review the audit findings before any preprocessing,")
add("deduplication, splitting, or model training.")
add("")

OUT_PATH.write_text("\n".join(lines), encoding="utf-8")

print("MASTER AUDIT REPORT CREATED")
print(OUT_PATH)
print(f"Size: {OUT_PATH.stat().st_size:,} bytes")