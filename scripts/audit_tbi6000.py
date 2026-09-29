from pathlib import Path
from collections import Counter
from PIL import Image

# ============================================================
# TBI-6000 IMAGE AUDIT
# ============================================================

ROOT = Path(
    r"C:\Users\shrey\OneDrive\Desktop\Brain_Hemorrhage_Project"
) / "datasets" / "TBI-6000"

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff"
}

print("=" * 70)
print("TBI-6000 IMAGE AUDIT")
print("=" * 70)

# ------------------------------------------------------------
# CHECK DATASET
# ------------------------------------------------------------

print("\nDataset location:")
print(ROOT)

if not ROOT.exists():

    print("\nERROR: Dataset folder not found.")
    print("Check the ROOT path in this script.")

    raise SystemExit

print("\nDataset folder: FOUND")

# ------------------------------------------------------------
# FIND CLASS FOLDERS
# ------------------------------------------------------------

class_folders = sorted(
    [
        folder
        for folder in ROOT.iterdir()
        if folder.is_dir()
    ]
)

print("\nClass folders:")

for folder in class_folders:

    print("  -", folder.name)

# ------------------------------------------------------------
# FIND ALL IMAGES
# ------------------------------------------------------------

print("\nSearching for images...")

all_images = []

for folder in class_folders:

    folder_images = [
        file
        for file in folder.rglob("*")
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print(
        f"{folder.name:25s}: "
        f"{len(folder_images)} images"
    )

    all_images.extend(folder_images)

print("\nTOTAL IMAGES:", len(all_images))

# ------------------------------------------------------------
# FILE EXTENSIONS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FILE EXTENSIONS")
print("=" * 70)

extension_counts = Counter(
    file.suffix.lower()
    for file in all_images
)

for extension, count in extension_counts.items():

    print(
        f"{extension:10s}: {count}"
    )

# ------------------------------------------------------------
# IMAGE DIMENSIONS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("IMAGE DIMENSIONS")
print("=" * 70)

dimension_counts = Counter()

corrupted_images = []

for image_path in all_images:

    try:

        with Image.open(image_path) as image:

            dimension_counts[image.size] += 1

    except Exception as error:

        corrupted_images.append(
            (
                image_path,
                str(error)
            )
        )

print("\nDimensions found:")

for dimension, count in dimension_counts.most_common():

    print(
        f"{dimension}: {count} images"
    )

# ------------------------------------------------------------
# CORRUPTED IMAGES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CORRUPTED / UNREADABLE IMAGES")
print("=" * 70)

print(
    "Total corrupted/unreadable:",
    len(corrupted_images)
)

if corrupted_images:

    for path, error in corrupted_images[:20]:

        print("\nFILE:")
        print(path)

        print("ERROR:")
        print(error)

else:

    print("No corrupted images found.")

# ------------------------------------------------------------
# EMPTY FILES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("EMPTY FILES")
print("=" * 70)

empty_files = []

for image_path in all_images:

    try:

        if image_path.stat().st_size == 0:

            empty_files.append(image_path)

    except Exception:
        pass

print(
    "Zero-byte image files:",
    len(empty_files)
)

for path in empty_files[:20]:

    print(path)

# ------------------------------------------------------------
# FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(
    "Total images:",
    len(all_images)
)

print(
    "Corrupted images:",
    len(corrupted_images)
)

print(
    "Zero-byte files:",
    len(empty_files)
)

print("\nDimension distribution:")

for dimension, count in dimension_counts.most_common():

    print(
        f"  {dimension}: {count}"
    )

print("\nAudit completed.")