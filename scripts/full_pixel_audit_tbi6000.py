from pathlib import Path
from PIL import Image
from collections import Counter
import numpy as np

ROOT = Path(r"datasets\TBI-6000")

print("=" * 80)
print("TBI-6000 FULL PIXEL AUDIT")
print("=" * 80)

# Find all PNG files
images = sorted(ROOT.rglob("*.png"))

print(f"\nPNG files found: {len(images)}")

# Counters
mode_counter = Counter()
dtype_counter = Counter()
size_counter = Counter()

blank_images = []
constant_images = []
read_errors = []

global_min = 255
global_max = 0

# Process every image
for i, image_path in enumerate(images, start=1):

    try:
        with Image.open(image_path) as img:

            # Image representation
            mode_counter[img.mode] += 1
            size_counter[img.size] += 1

            # Convert to numpy array
            arr = np.array(img)

            dtype_counter[str(arr.dtype)] += 1

            # Pixel range
            image_min = int(arr.min())
            image_max = int(arr.max())

            global_min = min(global_min, image_min)
            global_max = max(global_max, image_max)

            # Check for completely constant image
            if image_min == image_max:
                constant_images.append(
                    (str(image_path), image_min)
                )

            # Check for completely blank image
            if np.all(arr == 0):
                blank_images.append(str(image_path))

    except Exception as e:
        read_errors.append(
            (str(image_path), str(e))
        )

    # Progress every 500 images
    if i % 500 == 0:
        print(f"Processed: {i}/{len(images)}")

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

print("\nIMAGE MODES")
for mode, count in mode_counter.items():
    print(f"{mode}: {count}")

print("\nDATA TYPES")
for dtype, count in dtype_counter.items():
    print(f"{dtype}: {count}")

print("\nIMAGE DIMENSIONS")
for size, count in size_counter.items():
    print(f"{size}: {count}")

print("\nGLOBAL PIXEL RANGE")
print(f"Minimum pixel value across dataset: {global_min}")
print(f"Maximum pixel value across dataset: {global_max}")

print("\nCOMPLETELY BLANK IMAGES")
print(f"Total: {len(blank_images)}")

if blank_images:
    for path in blank_images:
        print(f"  {path}")

print("\nCONSTANT-VALUE IMAGES")
print(f"Total: {len(constant_images)}")

if constant_images:
    for path, value in constant_images:
        print(f"  {path} -> value={value}")

print("\nREAD / PROCESSING ERRORS")
print(f"Total: {len(read_errors)}")

if read_errors:
    for path, error in read_errors:
        print(f"  {path}")
        print(f"    Error: {error}")

print("\n" + "=" * 80)
print("FULL PIXEL AUDIT COMPLETE")
print("=" * 80)