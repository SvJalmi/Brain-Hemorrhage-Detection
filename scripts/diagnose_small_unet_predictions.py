import sys
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.small_unet import SmallUNet, IMAGE_SIZE

CHECKPOINT = PROJECT_ROOT / "models" / "segmentation" / "small_unet_mbh_seg25_best.pt"
SPLIT_CSV = PROJECT_ROOT / "preprocessing" / "segmentation" / "mbh_seg25_split.csv"

CLASS_NAMES = [
    "background",
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
]


def main():
    torch.set_num_threads(12)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    model = SmallUNet()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    df = pd.read_csv(SPLIT_CSV)
    df = df[df["split"] == "test"].reset_index(drop=True)

    predicted_counts = np.zeros(6, dtype=np.int64)
    target_counts = np.zeros(6, dtype=np.int64)

    with torch.no_grad():
        for case_index, row in df.iterrows():

            image = nib.load(
                str(row["image_path"])
            ).get_fdata(dtype=np.float32)

            target = nib.load(
                str(row["annot_3_path"])
            ).get_fdata(dtype=np.float32).astype(np.uint8)

            for z in range(image.shape[2]):

                image_slice = torch.from_numpy(
                    image[:, :, z].copy()
                ).unsqueeze(0).unsqueeze(0)

                image_slice = F.interpolate(
                    image_slice,
                    size=(IMAGE_SIZE, IMAGE_SIZE),
                    mode="bilinear",
                    align_corners=False,
                )

                image_slice = torch.clamp(
                    image_slice,
                    -1000,
                    1000,
                )

                image_slice = (
                    image_slice + 1000.0
                ) / 2000.0

                logits = model(image_slice)

                pred = torch.argmax(
                    logits,
                    dim=1,
                ).squeeze(0).numpy().astype(np.uint8)

                target_slice = torch.from_numpy(
                    target[:, :, z].copy()
                ).unsqueeze(0).unsqueeze(0).float()

                target_slice = F.interpolate(
                    target_slice,
                    size=(IMAGE_SIZE, IMAGE_SIZE),
                    mode="nearest",
                ).squeeze().numpy().astype(np.uint8)

                for class_id in range(6):
                    predicted_counts[class_id] += int(
                        (pred == class_id).sum()
                    )

                    target_counts[class_id] += int(
                        (target_slice == class_id).sum()
                    )

            print(
                f"Processed {case_index + 1}/{len(df)} test cases"
            )

    predicted_total = predicted_counts.sum()
    target_total = target_counts.sum()

    print("\nPrediction distribution:")

    for class_id, name in enumerate(CLASS_NAMES):
        pct = (
            100.0
            * predicted_counts[class_id]
            / predicted_total
        )

        print(
            f"{class_id} {name}: "
            f"{predicted_counts[class_id]} pixels "
            f"({pct:.2f}%)"
        )

    print("\nReference distribution:")

    for class_id, name in enumerate(CLASS_NAMES):
        pct = (
            100.0
            * target_counts[class_id]
            / target_total
        )

        print(
            f"{class_id} {name}: "
            f"{target_counts[class_id]} pixels "
            f"({pct:.2f}%)"
        )


if __name__ == "__main__":
    main()