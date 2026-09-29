from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path(r"datasets\TBI-6000")

pairs = [
    (
        "subarachnoid",
        "ID_1254_subarachnoid_UNIQUE.png",
        "ID_373_subarachnoid_UNIQUE.png",
    ),
    (
        "intraparenchymal",
        "ID_1603_intraparenchymal_UNIQUE.png",
        "ID_487_intraparenchymal_UNIQUE.png",
    ),
    (
        "intraventricular",
        "ID_110_intraventricular_UNIQUE.png",
        "ID_1556_intraventricular_UNIQUE.png",
    ),
    (
        "intraventricular",
        "ID_1651_intraventricular_UNIQUE.png",
        "ID_233_intraventricular_UNIQUE.png",
    ),
]

print("=" * 80)
print("TBI-6000 PIXEL-LEVEL DUPLICATE VERIFICATION")
print("=" * 80)

for class_name, file1, file2 in pairs:

    path1 = ROOT / class_name / file1
    path2 = ROOT / class_name / file2

    print("\n" + "-" * 80)
    print(f"CLASS: {class_name}")
    print(f"IMAGE 1: {file1}")
    print(f"IMAGE 2: {file2}")
    print("-" * 80)

    try:
        with Image.open(path1) as img1:
            arr1 = np.array(img1)

        with Image.open(path2) as img2:
            arr2 = np.array(img2)

        print(f"Image 1 shape: {arr1.shape}")
        print(f"Image 2 shape: {arr2.shape}")
        print(f"Image 1 dtype: {arr1.dtype}")
        print(f"Image 2 dtype: {arr2.dtype}")

        if arr1.shape != arr2.shape:
            print("PIXEL IDENTICAL: NO")
            print("Reason: Different array shapes.")
            continue

        identical = np.array_equal(arr1, arr2)

        print(f"PIXEL IDENTICAL: {'YES' if identical else 'NO'}")

        if identical:
            print("Every pixel value is exactly the same.")
        else:
            difference = np.abs(
                arr1.astype(np.int16) - arr2.astype(np.int16)
            )

            different_pixels = np.count_nonzero(difference)

            print(f"Different pixels: {different_pixels}")
            print(f"Maximum pixel difference: {difference.max()}")

    except Exception as e:
        print(f"ERROR: {e}")

print("\n" + "=" * 80)
print("PIXEL-LEVEL VERIFICATION COMPLETE")
print("=" * 80)