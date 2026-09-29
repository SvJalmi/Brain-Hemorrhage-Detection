from pathlib import Path
import hashlib

SOURCE = Path(r"C:\Users\shrey\Downloads\MBH_Train_2025_voxel-label (1)")
PROJECT = Path(r"C:\Users\shrey\OneDrive\Desktop\Brain_Hemorrhage_Project\datasets\MBH-Seg25")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


source_files = {
    p.relative_to(SOURCE): p
    for p in SOURCE.rglob("*")
    if p.is_file()
}

project_files = {
    p.relative_to(PROJECT): p
    for p in PROJECT.rglob("*")
    if p.is_file()
}

print("SOURCE files :", len(source_files))
print("PROJECT files:", len(project_files))
print()

source_keys = set(source_files)
project_keys = set(project_files)

missing_in_project = sorted(source_keys - project_keys)
extra_in_project = sorted(project_keys - source_keys)

print("Missing in project:", len(missing_in_project))
print("Extra in project  :", len(extra_in_project))
print()

if missing_in_project:
    print("FILES MISSING IN PROJECT:")
    for p in missing_in_project:
        print("  ", p)

if extra_in_project:
    print("EXTRA FILES IN PROJECT:")
    for p in extra_in_project:
        print("  ", p)

common = sorted(source_keys & project_keys)

different_hash = []

print("Comparing SHA-256 hashes...")
for i, rel in enumerate(common, 1):
    source_hash = sha256(source_files[rel])
    project_hash = sha256(project_files[rel])

    if source_hash != project_hash:
        different_hash.append(rel)

    if i % 50 == 0 or i == len(common):
        print(f"Checked {i}/{len(common)} files")

print()
print("Files with different SHA-256:", len(different_hash))

if different_hash:
    print()
    print("HASH DIFFERENCES:")
    for p in different_hash:
        print("  ", p)

print()
if (
    len(source_files) == len(project_files)
    and not missing_in_project
    and not extra_in_project
    and not different_hash
):
    print("RESULT: SOURCE AND PROJECT COPIES ARE IDENTICAL.")
else:
    print("RESULT: DIFFERENCES FOUND. DO NOT MODIFY ANYTHING YET.")