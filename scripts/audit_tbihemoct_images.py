from pathlib import Path
from PIL import Image
import numpy as np
from collections import Counter

ROOT = Path(
    r"datasets\TBI-HemoCT\TBI-HemoCT Data set\TBI-HemoCT Data set"
)

files = sorted(ROOT.rglob("*.png"))

dimensions = Counter()
modes = Counter()
dtypes = Counter()

errors = []
zero_byte = []
blank = []
constant = []

global_min = 256
global_max = -1

print("=" * 60)
print("TBI-HemoCT FULL IMAGE INTEGRITY AUDIT")
print("=" * 60)
print(f"PNG files found: {len(files)}")
print()

for i, path in enumerate(files, 1):

    try:
        if path.stat().st_size == 0:
            zero_byte.append(str(path))
            continue

        with Image.open(path) as img:
            img.verify()

        with Image.open(path) as img:
            arr = np.asarray(img)

            dimensions[arr.shape] += 1
            modes[img.mode] += 1
            dtypes[str(arr.dtype)] += 1

            if arr.size == 0:
                blank.append(str(path))
                continue

            image_min = int(arr.min())
            image_max = int(arr.max())

            global_min = min(global_min, image_min)
            global_max = max(global_max, image_max)

            if np.all(arr == 0):
                blank.append(str(path))

            if np.all(arr == arr.flat[0]):
                constant.append(str(path))

    except Exception as e:
        errors.append((str(path), str(e)))

    if i % 5000 == 0:
        print(f"Processed {i}/{len(files)}")

report = Path(r"dataset_check\TBIHemoCT_IMAGE_INTEGRITY_REPORT.txt")

with report.open("w", encoding="utf-8") as f:

    f.write("TBI-HemoCT FULL IMAGE INTEGRITY AUDIT\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"PNG files found: {len(files)}\n\n")

    f.write("IMAGE DIMENSIONS\n")
    f.write("-" * 40 + "\n")
    for key, value in sorted(dimensions.items()):
        f.write(f"{key}: {value}\n")
    f.write("\n")

    f.write("IMAGE MODES\n")
    f.write("-" * 40 + "\n")
    for key, value in sorted(modes.items()):
        f.write(f"{key}: {value}\n")
    f.write("\n")

    f.write("DATA TYPES\n")
    f.write("-" * 40 + "\n")
    for key, value in sorted(dtypes.items()):
        f.write(f"{key}: {value}\n")
    f.write("\n")

    f.write("GLOBAL PIXEL RANGE\n")
    f.write("-" * 40 + "\n")
    f.write(f"Minimum: {global_min}\n")
    f.write(f"Maximum: {global_max}\n\n")

    f.write("ZERO-BYTE FILES\n")
    f.write("-" * 40 + "\n")
    f.write(f"Total: {len(zero_byte)}\n")
    for path in zero_byte:
        f.write(f"{path}\n")
    f.write("\n")

    f.write("BLANK IMAGES (ALL PIXELS ZERO)\n")
    f.write("-" * 40 + "\n")
    f.write(f"Total: {len(blank)}\n")
    for path in blank:
        f.write(f"{path}\n")
    f.write("\n")

    f.write("CONSTANT-VALUE IMAGES\n")
    f.write("-" * 40 + "\n")
    f.write(f"Total: {len(constant)}\n")
    for path in constant:
        f.write(f"{path}\n")
    f.write("\n")

    f.write("READ / PROCESSING ERRORS\n")
    f.write("-" * 40 + "\n")
    f.write(f"Total: {len(errors)}\n")
    for path, error in errors:
        f.write(f"{path} | {error}\n")
    f.write("\n")

print()
print("=" * 60)
print("AUDIT COMPLETE")
print("=" * 60)
print(f"Report saved to: {report}")
print()
print("IMAGE DIMENSIONS:")
for key, value in sorted(dimensions.items()):
    print(f"  {key}: {value}")

print()
print("IMAGE MODES:")
for key, value in sorted(modes.items()):
    print(f"  {key}: {value}")

print()
print("DATA TYPES:")
for key, value in sorted(dtypes.items()):
    print(f"  {key}: {value}")

print()
print("GLOBAL PIXEL RANGE:")
print(f"  Minimum: {global_min}")
print(f"  Maximum: {global_max}")

print()
print(f"ZERO-BYTE FILES: {len(zero_byte)}")
print(f"BLANK IMAGES: {len(blank)}")
print(f"CONSTANT-VALUE IMAGES: {len(constant)}")
print(f"READ / PROCESSING ERRORS: {len(errors)}")

print()
print("NON-512x512 FILES")
print("-" * 60)

for path in files:
    try:
        with Image.open(path) as img:
            if img.size != (512, 512):
                print(f"{img.size}: {path}")
    except Exception:
        pass