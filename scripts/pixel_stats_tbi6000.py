from pathlib import Path
from PIL import Image
import numpy as np

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
print("TBI-6000 PIXEL-INTENSITY STATISTICS")
print("=" * 80)

for class_name in classes:

    class_dir = ROOT / class_name
    images = sorted(class_dir.glob("*.png"))[:10]

    print()
    print("-" * 80)
    print(f"CLASS: {class_name}")
    print("-" * 80)

    for image_path in images:

        with Image.open(image_path) as img:
            arr = np.array(img)

            print(
                f"{image_path.name} | "
                f"mode={img.mode} | "
                f"size={img.size} | "
                f"dtype={arr.dtype} | "
                f"min={arr.min()} | "
                f"max={arr.max()} | "
                f"mean={arr.mean():.2f}"
            )

print()
print("=" * 80)
print("PIXEL STATISTICS CHECK COMPLETE")
print("=" * 80)