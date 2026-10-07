from pathlib import Path

import nibabel as nib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = PROJECT_ROOT / "datasets" / "MBH-Seg25"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "preprocessing"
    / "segmentation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "mbh_seg25_manifest.csv"


print("=" * 70)
print("MBH-SEG25 MANIFEST PREPARATION")
print("=" * 70)

cases = sorted(
    [
        p
        for p in DATASET_ROOT.iterdir()
        if p.is_dir() and (p / "image.nii.gz").exists()
    ]
)

print(f"\nCases found: {len(cases)}")

if len(cases) == 0:
    raise RuntimeError("No MBH-Seg25 cases found.")


records = []

for case_dir in cases:

    case_id = case_dir.name

    image_path = case_dir / "image.nii.gz"

    annotation_paths = sorted(
        case_dir.glob("label_annot_*.nii.gz")
    )

    if not annotation_paths:
        raise RuntimeError(
            f"No annotations found for case: {case_id}"
        )

    image_nii = nib.load(str(image_path))

    shape = image_nii.shape

    spacing = image_nii.header.get_zooms()[:3]

    record = {
        "case_id": case_id,
        "image_path": str(image_path),
        "annotation_count": len(annotation_paths),
        "shape_x": shape[0],
        "shape_y": shape[1],
        "shape_z": shape[2],
        "spacing_x": float(spacing[0]),
        "spacing_y": float(spacing[1]),
        "spacing_z": float(spacing[2]),
    }

    for i in range(1, 5):
        annotation = case_dir / f"label_annot_{i}.nii.gz"

        if annotation.exists():
            record[f"annot_{i}_path"] = str(annotation)
        else:
            record[f"annot_{i}_path"] = ""

    records.append(record)


df = pd.DataFrame(records)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nManifest columns:")
print(df.columns.tolist())

print("\nAnnotation-count distribution:")
print(
    df["annotation_count"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nShape distribution:")
print(
    df[
        ["shape_x", "shape_y", "shape_z"]
    ]
    .value_counts()
    .to_string()
)

print("\nManifest rows:", len(df))

print("\nSaved:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("STATUS: MBH-SEG25 MANIFEST PREPARATION COMPLETE")
print("=" * 70)