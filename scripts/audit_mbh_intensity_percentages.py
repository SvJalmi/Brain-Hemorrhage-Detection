import nibabel as nib
import numpy as np
from pathlib import Path

root = Path(r"C:\Users\shrey\Downloads\MBH_Train_2025_voxel-label (1)")

results = []

for case_dir in sorted(root.iterdir()):
    if not case_dir.is_dir():
        continue

    image_path = case_dir / "image.nii.gz"

    if not image_path.exists():
        continue

    data = np.asanyarray(nib.load(str(image_path)).dataobj)

    total = data.size
    below = int(np.sum(data < -2048))
    above = int(np.sum(data > 3071))

    below_pct = 100.0 * below / total
    above_pct = 100.0 * above / total

    results.append(
        (
            case_dir.name,
            data.shape,
            float(data.min()),
            float(data.max()),
            below,
            below_pct,
            above,
            above_pct,
        )
    )

print("Cases checked:", len(results))
print()

print("TOP 10 CASES BY PERCENTAGE BELOW -2048")
print("-" * 90)

for r in sorted(results, key=lambda x: x[5], reverse=True)[:10]:
    print(
        r[0],
        "| shape:", r[1],
        "| min:", r[2],
        "| max:", r[3],
        "| below -2048:", r[4],
        "| percent:", round(r[5], 4)
    )

print()
print("TOP 10 CASES BY PERCENTAGE ABOVE 3071")
print("-" * 90)

for r in sorted(results, key=lambda x: x[7], reverse=True)[:10]:
    print(
        r[0],
        "| shape:", r[1],
        "| min:", r[2],
        "| max:", r[3],
        "| above 3071:", r[6],
        "| percent:", round(r[7], 4)
    )