from pathlib import Path
from PIL import Image
import hashlib
import csv
import re
from collections import defaultdict

ROOT = Path(r"datasets\TBI-6000")
OUTPUT = Path(r"dataset_check\tbi6000_duplicate_metadata.csv")


def calculate_hash(path):
    """Calculate SHA-256 hash of the complete PNG file."""
    sha256 = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


def get_base_id(filename):
    """
    Extract the original ID from filenames such as:

    ID_104_epidural_UNIQUE.png
    ID_104_epidural_UNIQUE_aug_1888.png
    """

    match = re.match(r"(ID_\d+)", filename)

    if match:
        return match.group(1)

    return ""


def is_augmented(filename):
    """Determine whether filename contains an augmentation suffix."""
    return "_aug_" in filename.lower()


# ---------------------------------------------------------
# Find all PNG files
# ---------------------------------------------------------

images = sorted(ROOT.rglob("*.png"))

print("=" * 80)
print("TBI-6000 DUPLICATE METADATA AUDIT")
print("=" * 80)

print(f"\nPNG files found: {len(images)}")


# ---------------------------------------------------------
# Calculate file hashes
# ---------------------------------------------------------

hash_to_files = defaultdict(list)

for i, path in enumerate(images, start=1):

    file_hash = calculate_hash(path)

    hash_to_files[file_hash].append(path)

    if i % 500 == 0:
        print(f"Processed: {i}/{len(images)}")


# ---------------------------------------------------------
# Keep only duplicate groups
# ---------------------------------------------------------

duplicate_groups = {
    h: files
    for h, files in hash_to_files.items()
    if len(files) > 1
}


print()
print(f"Unique file hashes: {len(hash_to_files)}")
print(f"Duplicate groups: {len(duplicate_groups)}")


# ---------------------------------------------------------
# Create metadata rows
# ---------------------------------------------------------

rows = []

for group_number, (file_hash, files) in enumerate(
    sorted(duplicate_groups.items()),
    start=1
):

    for path in sorted(files):

        class_name = path.parent.name

        with Image.open(path) as img:
            dimensions = img.size
            mode = img.mode

        filename = path.name

        rows.append({
            "duplicate_group": group_number,
            "class": class_name,
            "filename": filename,
            "base_id": get_base_id(filename),
            "augmented": is_augmented(filename),
            "dimensions": f"{dimensions[0]}x{dimensions[1]}",
            "mode": mode,
            "sha256": file_hash,
        })


# ---------------------------------------------------------
# Save CSV
# ---------------------------------------------------------

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

fieldnames = [
    "duplicate_group",
    "class",
    "filename",
    "base_id",
    "augmented",
    "dimensions",
    "mode",
    "sha256",
]

with open(
    OUTPUT,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(rows)


# ---------------------------------------------------------
# Print duplicate groups
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("DUPLICATE GROUPS")
print("=" * 80)

for group_number, (file_hash, files) in enumerate(
    sorted(duplicate_groups.items()),
    start=1
):

    print(f"\nGroup {group_number}:")

    for path in sorted(files):

        augmented = "AUGMENTED" if is_augmented(path.name) else "ORIGINAL"

        print(
            f"  [{augmented}] "
            f"{path.parent.name}\\{path.name}"
        )


print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"Total PNG files: {len(images)}")
print(f"Unique hashes: {len(hash_to_files)}")
print(f"Duplicate groups: {len(duplicate_groups)}")
print(f"Files in duplicate groups: {len(rows)}")

print("\nCSV report created:")
print(OUTPUT)

print("\nNo dataset files were modified.")