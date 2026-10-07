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

OUTPUT_FILE = (
    PROJECT_ROOT
    / "dataset_check"
    / "MBH_SEG25_FINAL_SPLIT_AUDIT.txt"
)

df = pd.read_csv(INPUT_FILE)

LABELS = {
    1: "epidural",
    2: "intraparenchymal",
    3: "intraventricular",
    4: "subarachnoid",
    5: "subdural",
}

lines = []

def add(text=""):
    lines.append(str(text))


add("=" * 70)
add("MBH-SEG25 FINAL SPLIT AUDIT")
add("=" * 70)

add()
add(f"Total cases: {len(df)}")
add(f"Unique case IDs: {df['case_id'].nunique()}")

add()
add("SPLIT COUNTS")
add("-" * 70)

for split in ["train", "validation", "test"]:
    add(f"{split}: {int((df['split'] == split).sum())}")

add()
add("ANNOTATION COUNTS")
add("-" * 70)

annotation_counts = pd.crosstab(
    df["split"],
    df["annotation_count"]
)

add(annotation_counts.to_string())

add()
add("SUBTYPE PRESENCE BY CASE")
add("-" * 70)

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

        for value, name in LABELS.items():

            if value in found_labels:
                counts[name] += 1

    add()
    add(f"{split.upper()} ({len(split_df)} cases)")

    for name, count in counts.items():
        add(f"{name}: {count}")

add()
add("VOLUME DIMENSIONS")
add("-" * 70)

shape_counts = (
    df[["shape_x", "shape_y", "shape_z"]]
    .value_counts()
)

add(shape_counts.to_string())

add()
add("VOXEL SPACING")
add("-" * 70)

add(
    "XY spacing range: "
    f"{df['spacing_x'].min():.6f} "
    f"to {df['spacing_x'].max():.6f} mm"
)

add(
    "Z spacing range: "
    f"{df['spacing_z'].min():.6f} "
    f"to {df['spacing_z'].max():.6f} mm"
)

add()
add("FILE EXISTENCE")
add("-" * 70)

missing_images = 0
missing_annotations = 0

for _, row in df.iterrows():

    if not Path(row["image_path"]).exists():
        missing_images += 1

    for i in range(1, 5):

        path = row[f"annot_{i}_path"]

        if isinstance(path, str) and path:
            if not Path(path).exists():
                missing_annotations += 1

add(f"Missing CT images: {missing_images}")
add(f"Missing annotation files: {missing_annotations}")

add()
add("SPLIT LEAKAGE CHECK")
add("-" * 70)

case_split_counts = df.groupby("case_id")["split"].nunique()

leakage_cases = int((case_split_counts > 1).sum())

add(f"Cases appearing in multiple splits: {leakage_cases}")

add()
add("=" * 70)
add("FINAL STATUS")
add("=" * 70)

if (
    len(df) == 192
    and df["case_id"].nunique() == 192
    and missing_images == 0
    and missing_annotations == 0
    and leakage_cases == 0
):
    add("MBH-SEG25 FINAL SPLIT AUDIT: PASSED")
else:
    add("MBH-SEG25 FINAL SPLIT AUDIT: CHECK REQUIRED")

add("=" * 70)

OUTPUT_FILE.write_text(
    "\n".join(lines),
    encoding="utf-8"
)

print("\n".join(lines))
print()
print("Saved:")
print(OUTPUT_FILE)