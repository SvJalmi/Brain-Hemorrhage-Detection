import nibabel as nib
import numpy as np
from pathlib import Path

root = Path(r"C:\Users\shrey\Downloads\MBH_Train_2025_voxel-label (1)")

case_count = 0
disagreement_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

for case_dir in sorted(root.iterdir()):
    if not case_dir.is_dir():
        continue

    annotation_files = sorted(case_dir.glob("label_annot_*.nii.gz"))

    if not annotation_files:
        continue

    case_count += 1

    label_sets = []

    for path in annotation_files:
        data = np.asanyarray(nib.load(str(path)).dataobj)

        labels = set(
            int(x)
            for x in np.unique(data)
            if 1 <= x <= 5
        )

        label_sets.append(labels)

    for label in range(1, 6):
        presence = [label in labels for labels in label_sets]

        if len(set(presence)) > 1:
            disagreement_counts[label] += 1

print("Cases checked:", case_count)
print()
print("Subtype presence disagreement:")
print("Label 1 (epidural):", disagreement_counts[1])
print("Label 2 (intraparenchymal):", disagreement_counts[2])
print("Label 3 (intraventricular):", disagreement_counts[3])
print("Label 4 (subarachnoid):", disagreement_counts[4])
print("Label 5 (subdural):", disagreement_counts[5])