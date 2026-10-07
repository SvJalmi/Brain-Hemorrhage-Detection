from pathlib import Path
import shutil
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLIT_CSV = PROJECT_ROOT / "preprocessing" / "segmentation" / "mbh_seg25_split.csv"
OUTPUT_ROOT = PROJECT_ROOT / "nnunet" / "nnUNet_raw" / "Dataset501_MBHSeg25"
IMAGES_TR = OUTPUT_ROOT / "imagesTr"
LABELS_TR = OUTPUT_ROOT / "labelsTr"


def main():
    df = pd.read_csv(SPLIT_CSV)
    train = df[df["split"] == "train"].copy()

    if len(train) != 134:
        raise RuntimeError(f"Expected 134 training cases, found {len(train)}")
    if train["case_id"].nunique() != 134:
        raise RuntimeError("Training case IDs are not unique")

    IMAGES_TR.mkdir(parents=True, exist_ok=True)
    LABELS_TR.mkdir(parents=True, exist_ok=True)

    for index, row in enumerate(train.itertuples(index=False), start=1):
        case_name = f"case_{index:04d}"
        image_src = Path(row.image_path)
        label_src = Path(row.annot_3_path)

        if not image_src.exists():
            raise FileNotFoundError(f"Missing image: {image_src}")
        if not label_src.exists():
            raise FileNotFoundError(f"Missing annotation: {label_src}")

        image_dst = IMAGES_TR / f"{case_name}_0000.nii.gz"
        label_dst = LABELS_TR / f"{case_name}.nii.gz"

        shutil.copy2(image_src, image_dst)
        shutil.copy2(label_src, label_dst)

    image_files = sorted(IMAGES_TR.glob("*_0000.nii.gz"))
    label_files = sorted(LABELS_TR.glob("*.nii.gz"))

    if len(image_files) != 134:
        raise RuntimeError(f"Expected 134 training images, found {len(image_files)}")
    if len(label_files) != 134:
        raise RuntimeError(f"Expected 134 training labels, found {len(label_files)}")

    image_cases = {p.name.replace("_0000.nii.gz", "") for p in image_files}
    label_cases = {p.name.replace(".nii.gz", "") for p in label_files}

    if image_cases != label_cases:
        raise RuntimeError("Image/label case names do not match")

    print("MBH nnU-Net training dataset prepared successfully")
    print("Training cases:", len(train))
    print("Images copied:", len(image_files))
    print("Labels copied:", len(label_files))
    print("Reference annotation: annot_3")
    print("Original MBH-Seg25 files were not modified.")


if __name__ == "__main__":
    main()
