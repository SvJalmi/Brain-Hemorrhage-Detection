from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = PROJECT_ROOT / "datasets" / "MBH-Seg25"

SPLIT_CSV = (
    PROJECT_ROOT
    / "preprocessing"
    / "segmentation"
    / "mbh_seg25_split.csv"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "segmentation"
    / "small_unet_mbh_seg25_imbalance_aware_best.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "segmentation"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "small_unet_imbalance_aware_test_metrics.csv"
)

CASE_OUTPUT_FILE = (
    OUTPUT_DIR
    / "small_unet_imbalance_aware_test_case_metrics.csv"
)

IMAGE_SIZE = 256
NUM_CLASSES = 6

CLASS_NAMES = [
    "background",
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
]


# ============================================================
# EXACT MODEL ARCHITECTURE USED DURING TRAINING
# ============================================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SmallUNet(nn.Module):

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.enc1 = DoubleConv(1, 16)
        self.enc2 = DoubleConv(16, 32)
        self.enc3 = DoubleConv(32, 64)

        self.bottleneck = DoubleConv(64, 96)

        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(
            96,
            64,
            kernel_size=2,
            stride=2,
        )

        self.dec3 = DoubleConv(128, 64)

        self.up2 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2,
        )

        self.dec2 = DoubleConv(64, 32)

        self.up1 = nn.ConvTranspose2d(
            32,
            16,
            kernel_size=2,
            stride=2,
        )

        self.dec1 = DoubleConv(32, 16)

        # IMPORTANT:
        # Training checkpoint uses "final", not "out".
        self.final = nn.Conv2d(
            16,
            num_classes,
            kernel_size=1,
        )

    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        b = self.bottleneck(
            self.pool(e3)
        )

        d3 = self.up3(b)
        d3 = torch.cat(
            [d3, e3],
            dim=1,
        )
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat(
            [d2, e2],
            dim=1,
        )
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat(
            [d1, e1],
            dim=1,
        )
        d1 = self.dec1(d1)

        return self.final(d1)


# ============================================================
# PREPROCESSING
# EXACTLY MATCHES IMBALANCE-AWARE TRAINING
# ============================================================

def preprocess_image_slice(image_slice):

    lower = np.percentile(
        image_slice,
        1,
    )

    upper = np.percentile(
        image_slice,
        99,
    )

    if upper > lower:

        image_slice = np.clip(
            image_slice,
            lower,
            upper,
        )

        image_slice = (
            image_slice - lower
        ) / (
            upper - lower
        )

    else:

        image_slice = np.zeros_like(
            image_slice,
            dtype=np.float32,
        )

    image_tensor = torch.from_numpy(
        image_slice.astype(np.float32)
    ).float()

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = F.interpolate(
        image_tensor.unsqueeze(0),
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    return image_tensor


def preprocess_mask_slice(mask_slice):

    mask_tensor = torch.from_numpy(
        mask_slice.astype(np.int64)
    ).long()

    mask_tensor = F.interpolate(
        mask_tensor.unsqueeze(0)
        .unsqueeze(0)
        .float(),
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="nearest",
    ).squeeze(0).squeeze(0).long()

    return mask_tensor


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(confusion):

    rows = []

    for class_idx, class_name in enumerate(CLASS_NAMES):

        tp = int(
            confusion[class_idx, class_idx]
        )

        fp = int(
            confusion[:, class_idx].sum()
            - tp
        )

        fn = int(
            confusion[class_idx, :].sum()
            - tp
        )

        dice_denominator = (
            2 * tp + fp + fn
        )

        iou_denominator = (
            tp + fp + fn
        )

        dice = (
            (2 * tp) / dice_denominator
            if dice_denominator > 0
            else 1.0
        )

        iou = (
            tp / iou_denominator
            if iou_denominator > 0
            else 1.0
        )

        sensitivity = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        rows.append(
            {
                "class": class_name,
                "dice": dice,
                "iou": iou,
                "sensitivity": sensitivity,
                "precision": precision,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )

    return rows


# ============================================================
# MAIN
# ============================================================

def main():

    torch.set_num_threads(12)

    print("=" * 70)
    print(
        "MBH-Seg25 — IMBALANCE-AWARE 2D U-NET TEST EVALUATION"
    )
    print("=" * 70)

    device = torch.device("cpu")

    print()
    print("Device:", device)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset directory not found:\n{DATASET_ROOT}"
        )

    if not SPLIT_CSV.exists():
        raise FileNotFoundError(
            f"Split CSV not found:\n{SPLIT_CSV}"
        )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT_PATH}"
        )

    # --------------------------------------------------------
    # Test cases
    # --------------------------------------------------------

    split_df = pd.read_csv(
        SPLIT_CSV
    )

    test_df = split_df[
        split_df["split"].astype(str).str.lower()
        == "test"
    ].copy()

    test_case_ids = (
        test_df["case_id"]
        .astype(str)
        .tolist()
    )

    print()
    print(
        "Test cases:",
        len(test_case_ids),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = SmallUNet(
        num_classes=NUM_CLASSES
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint[
            "model_state_dict"
        ]
    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.eval()

    print()
    print("Checkpoint loaded:")
    print(CHECKPOINT_PATH)

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    confusion = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64,
    )

    case_rows = []

    total_slices = 0

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EVALUATING TEST CASES")
    print("=" * 70)

    with torch.no_grad():

        for case_number, case_id in enumerate(
            test_case_ids,
            start=1,
        ):

            case_dir = (
                DATASET_ROOT / case_id
            )

            image_path = (
                case_dir / "image.nii.gz"
            )

            label_path = (
                case_dir / "label_annot_3.nii.gz"
            )

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Missing image:\n{image_path}"
                )

            if not label_path.exists():
                raise FileNotFoundError(
                    f"Missing reference annotation:\n{label_path}"
                )

            image = nib.load(
                str(image_path)
            ).get_fdata(
                dtype=np.float32
            )

            label = nib.load(
                str(label_path)
            ).get_fdata(
                dtype=np.float32
            )

            label = np.rint(
                label
            ).astype(np.int64)

            if image.shape != label.shape:
                raise RuntimeError(
                    f"Shape mismatch in {case_id}: "
                    f"image={image.shape}, "
                    f"label={label.shape}"
                )

            case_confusion = np.zeros(
                (NUM_CLASSES, NUM_CLASSES),
                dtype=np.int64,
            )

            # ------------------------------------------------
            # Every axial slice
            # ------------------------------------------------

            for z in range(
                image.shape[2]
            ):

                image_slice = (
                    image[:, :, z]
                )

                label_slice = (
                    label[:, :, z]
                )

                image_tensor = (
                    preprocess_image_slice(
                        image_slice
                    )
                )

                label_tensor = (
                    preprocess_mask_slice(
                        label_slice
                    )
                )

                image_tensor = (
                    image_tensor
                    .unsqueeze(0)
                    .to(device)
                )

                logits = model(
                    image_tensor
                )

                prediction = torch.argmax(
                    logits,
                    dim=1,
                )

                prediction = (
                    prediction
                    .squeeze(0)
                    .cpu()
                    .numpy()
                    .astype(np.int64)
                )

                target = (
                    label_tensor
                    .cpu()
                    .numpy()
                    .astype(np.int64)
                )

                true_flat = target.ravel()
                pred_flat = prediction.ravel()

                for true_class, pred_class in zip(
                    true_flat,
                    pred_flat,
                ):

                    if (
                        0 <= true_class < NUM_CLASSES
                        and
                        0 <= pred_class < NUM_CLASSES
                    ):

                        confusion[
                            true_class,
                            pred_class
                        ] += 1

                        case_confusion[
                            true_class,
                            pred_class
                        ] += 1

                total_slices += 1

            # ------------------------------------------------
            # Case-level metrics
            # ------------------------------------------------

            case_metrics = calculate_metrics(
                case_confusion
            )

            foreground_dice = np.mean(
                [
                    row["dice"]
                    for row in case_metrics[1:]
                ]
            )

            case_rows.append(
                {
                    "case_id": case_id,
                    "foreground_macro_dice":
                        foreground_dice,
                }
            )

            print(
                f"[{case_number:02d}/"
                f"{len(test_case_ids)}] "
                f"{case_id} — "
                f"Foreground Dice: "
                f"{foreground_dice:.4f}"
            )

    # --------------------------------------------------------
    # Final results
    # --------------------------------------------------------

    rows = calculate_metrics(
        confusion
    )

    print()
    print("=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print()
    print(
        f"{'Class':<20}"
        f"{'Dice':>10}"
        f"{'IoU':>10}"
        f"{'Sensitivity':>15}"
        f"{'Precision':>12}"
    )

    for row in rows:

        print(
            f"{row['class']:<20}"
            f"{row['dice']:>10.4f}"
            f"{row['iou']:>10.4f}"
            f"{row['sensitivity']:>15.4f}"
            f"{row['precision']:>12.4f}"
        )

    # --------------------------------------------------------
    # Foreground macro averages
    # --------------------------------------------------------

    foreground_rows = rows[1:]

    mean_dice = np.mean(
        [
            row["dice"]
            for row in foreground_rows
        ]
    )

    mean_iou = np.mean(
        [
            row["iou"]
            for row in foreground_rows
        ]
    )

    mean_sensitivity = np.mean(
        [
            row["sensitivity"]
            for row in foreground_rows
        ]
    )

    mean_precision = np.mean(
        [
            row["precision"]
            for row in foreground_rows
        ]
    )

    print()
    print("-" * 70)
    print("FOREGROUND MACRO AVERAGE")
    print("-" * 70)

    print(
        f"Mean Dice:        {mean_dice:.4f}"
    )

    print(
        f"Mean IoU:         {mean_iou:.4f}"
    )

    print(
        f"Mean Sensitivity: {mean_sensitivity:.4f}"
    )

    print(
        f"Mean Precision:   {mean_precision:.4f}"
    )

    print()
    print(
        "Total test slices:",
        total_slices,
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(rows).to_csv(
        OUTPUT_FILE,
        index=False,
    )

    pd.DataFrame(case_rows).to_csv(
        CASE_OUTPUT_FILE,
        index=False,
    )

    print()
    print("Metrics saved:")
    print(OUTPUT_FILE)

    print()
    print("Case metrics saved:")
    print(CASE_OUTPUT_FILE)

    print()
    print("=" * 70)
    print("TEST EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()