from pathlib import Path
from collections import Counter

ROOT = Path(r"datasets\TBI-6000")

classes = [
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "normal",
    "subarachnoid",
    "subdural",
]

print("=" * 80)
print("TBI-6000 ORIGINAL vs AUGMENTED IMAGE BREAKDOWN")
print("=" * 80)

grand_total = 0
grand_original = 0
grand_augmented = 0

for class_name in classes:

    class_dir = ROOT / class_name
    files = sorted(class_dir.glob("*.png"))

    original = [
        f for f in files
        if "_aug_" not in f.name.lower()
    ]

    augmented = [
        f for f in files
        if "_aug_" in f.name.lower()
    ]

    print()
    print("-" * 80)
    print(f"CLASS: {class_name}")
    print("-" * 80)

    print(f"Total images:       {len(files)}")
    print(f"Original images:    {len(original)}")
    print(f"Augmented images:   {len(augmented)}")

    if len(files) > 0:
        print(
            f"Original percentage:  "
            f"{len(original) / len(files) * 100:.2f}%"
        )

        print(
            f"Augmented percentage: "
            f"{len(augmented) / len(files) * 100:.2f}%"
        )

    grand_total += len(files)
    grand_original += len(original)
    grand_augmented += len(augmented)


print()
print("=" * 80)
print("DATASET TOTAL")
print("=" * 80)

print(f"Total images:       {grand_total}")
print(f"Original images:    {grand_original}")
print(f"Augmented images:   {grand_augmented}")

print(
    f"Original percentage:  "
    f"{grand_original / grand_total * 100:.2f}%"
)

print(
    f"Augmented percentage: "
    f"{grand_augmented / grand_total * 100:.2f}%"
)

print()
print("=" * 80)
print("BREAKDOWN COMPLETE")
print("=" * 80)