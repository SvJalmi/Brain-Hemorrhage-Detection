from pathlib import Path
from collections import defaultdict
import hashlib

ROOT = Path(
    r"C:\Users\shrey\OneDrive\OneDrive\Desktop\Brain_Hemorrhage_Project"
)

# Correct the path if needed
ROOT = Path(
    r"C:\Users\shrey\OneDrive\Desktop\Brain_Hemorrhage_Project"
) / "datasets" / "TBI-6000"

print("=" * 70)
print("TBI-6000 DUPLICATE CHECK")
print("=" * 70)

image_files = list(ROOT.rglob("*.png"))

print("\nPNG files found:", len(image_files))

hash_map = defaultdict(list)

for i, image_path in enumerate(image_files, start=1):

    try:

        with open(image_path, "rb") as f:

            file_hash = hashlib.sha256(
                f.read()
            ).hexdigest()

        hash_map[file_hash].append(image_path)

    except Exception as error:

        print("Could not read:", image_path)
        print(error)

duplicate_groups = [
    files
    for files in hash_map.values()
    if len(files) > 1
]

duplicate_image_count = sum(
    len(files)
    for files in duplicate_groups
)

print("\nUnique image hashes:", len(hash_map))

print(
    "Duplicate groups:",
    len(duplicate_groups)
)

print(
    "Images involved in duplicate groups:",
    duplicate_image_count
)

if duplicate_groups:

    print("\nDuplicate groups:\n")

    for number, group in enumerate(
        duplicate_groups,
        start=1
    ):

        print(f"Group {number}:")

        for path in group:

            print(
                " ",
                path.relative_to(ROOT)
            )

else:

    print("\nNo exact duplicate images found.")

print("\nAudit finished.")