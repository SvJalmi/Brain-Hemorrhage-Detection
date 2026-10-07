from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "preprocessing"
    / "segmentation"
    / "mbh_seg25_split.csv"
)

LABELS = {
    1: "epidural",
    2: "intraparenchymal",
    3: "intraventricular",
    4: "subarachnoid",
    5: "subdural",
}

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("MBH-SEG25 SUBTYPE PRESENCE BY SPLIT")
print("=" * 70)

for split in ["train", "validation", "test"]:

    split_df = df[df["split"] == split]

    counts = {
        name: 0
        for name in LABELS.values()
    }

    for _, row in split_df.iterrows():

        found_labels = set()

        for i in range(1, 5):

            path = row[f"annot_{i}_path"]

            if not isinstance(path, str) or not path:
                continue

            mask = nib.load(path).get_fdata()

            found_labels.update(
                np.unique(mask).astype(int).tolist()
            )

        for label_value, label_name in LABELS.items():

            if label_value in found_labels:
                counts[label_name] += 1

    print(f"\n{split.upper()} ({len(split_df)} cases)")

    for name, count in counts.items():
        print(f"{name}: {count}")

print("\n" + "=" * 70)
print("STATUS: SUBTYPE SPLIT CHECK COMPLETE")
print("=" * 70)