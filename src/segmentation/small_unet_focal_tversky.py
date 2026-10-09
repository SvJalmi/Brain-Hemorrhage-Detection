from pathlib import Path
import random
import csv

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# MBH-Seg25 — FOCAL-TVERSKY 2D U-NET
# Foreground-aware sampling + Focal-Tversky loss
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = PROJECT_ROOT / "datasets" / "MBH-Seg25"
SPLIT_CSV = (
    PROJECT_ROOT
    / "preprocessing"
    / "segmentation"
    / "mbh_seg25_split.csv"
)

MODEL_DIR = PROJECT_ROOT / "models" / "segmentation"

CHECKPOINT_PATH = (
    MODEL_DIR
    / "small_unet_mbh_seg25_focal_tversky_best.pt"
)

IMAGE_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 1e-3
NUM_WORKERS = 0

NUM_CLASSES = 6
SEED = 42

# Tversky parameters:
# beta > alpha gives more penalty to false negatives,
# which is useful for small hemorrhage detection.
ALPHA = 0.3
BETA = 0.7
GAMMA = 0.75

CLASS_NAMES = [
    "background",
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DATASET
# ============================================================

class MBHSliceDataset(Dataset):

    def __init__(
        self,
        case_dirs,
        foreground_probability=0.75,
    ):

        self.samples = []

        for case_dir in case_dirs:

            case_dir = Path(case_dir)

            image_path = (
                case_dir / "image.nii.gz"
            )

            label_path = (
                case_dir / "label_annot_3.nii.gz"
            )

            if not image_path.exists():
                continue

            if not label_path.exists():
                continue

            image = nib.load(
                str(image_path)
            )

            label = nib.load(
                str(label_path)
            )

            image_data = image.get_fdata(
                dtype=np.float32
            )

            label_data = label.get_fdata(
                dtype=np.float32
            )

            label_data = np.rint(
                label_data
            ).astype(np.int64)

            depth = image_data.shape[2]

            foreground_slices = []

            for z in range(depth):

                if np.any(
                    label_data[:, :, z] > 0
                ):
                    foreground_slices.append(z)

            background_slices = [
                z
                for z in range(depth)
                if z not in foreground_slices
            ]

            selected_foreground = (
                foreground_slices
            )

            selected_background = []

            if len(background_slices) > 0:

                n_background = max(
                    1,
                    int(
                        len(foreground_slices)
                        * (1 - foreground_probability)
                        / foreground_probability
                    ),
                )

                n_background = min(
                    n_background,
                    len(background_slices),
                )

                selected_background = random.sample(
                    background_slices,
                    n_background,
                )

            selected_slices = (
                selected_foreground
                + selected_background
            )

            for z in selected_slices:

                self.samples.append(
                    (
                        str(image_path),
                        str(label_path),
                        z,
                    )
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, label_path, z = (
            self.samples[index]
        )

        image = nib.load(
            image_path
        ).get_fdata(
            dtype=np.float32
        )

        label = nib.load(
            label_path
        ).get_fdata(
            dtype=np.float32
        )

        label = np.rint(
            label
        ).astype(np.int64)

        image_slice = image[:, :, z]
        label_slice = label[:, :, z]

        # ----------------------------------------------------
        # Robust per-slice normalization
        # ----------------------------------------------------

        lower = np.percentile(
            image_slice,
            1
        )

        upper = np.percentile(
            image_slice,
            99
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
            image_slice
        ).float()

        label_tensor = torch.from_numpy(
            label_slice
        ).long()

        image_tensor = image_tensor.unsqueeze(0)

        image_tensor = F.interpolate(
            image_tensor.unsqueeze(0),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        label_tensor = F.interpolate(
            label_tensor.unsqueeze(0)
            .unsqueeze(0)
            .float(),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="nearest",
        ).squeeze(0).squeeze(0).long()

        return image_tensor, label_tensor


# ============================================================
# U-NET
# Exact same architecture as previous experiments
# ============================================================

class DoubleConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SmallUNet(nn.Module):

    def __init__(
        self,
        num_classes=NUM_CLASSES,
    ):

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

        self.dec3 = DoubleConv(
            128,
            64,
        )

        self.up2 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2,
        )

        self.dec2 = DoubleConv(
            64,
            32,
        )

        self.up1 = nn.ConvTranspose2d(
            32,
            16,
            kernel_size=2,
            stride=2,
        )

        self.dec1 = DoubleConv(
            32,
            16,
        )

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
# FOCAL-TVERSKY LOSS
# ============================================================

def focal_tversky_loss(
    logits,
    targets,
    alpha=ALPHA,
    beta=BETA,
    gamma=GAMMA,
    smooth=1e-6,
):

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    total_loss = 0.0
    valid_classes = 0

    # Exclude background.
    for class_idx in range(
        1,
        NUM_CLASSES,
    ):

        prediction = probabilities[
            :, class_idx
        ]

        target = (
            targets == class_idx
        ).float()

        prediction_flat = prediction.reshape(
            prediction.shape[0],
            -1,
        )

        target_flat = target.reshape(
            target.shape[0],
            -1,
        )

        true_positive = (
            prediction_flat
            * target_flat
        ).sum(dim=1)

        false_positive = (
            prediction_flat
            * (1.0 - target_flat)
        ).sum(dim=1)

        false_negative = (
            (1.0 - prediction_flat)
            * target_flat
        ).sum(dim=1)

        tversky = (
            true_positive + smooth
        ) / (
            true_positive
            + alpha * false_positive
            + beta * false_negative
            + smooth
        )

        valid = (
            target_flat.sum(dim=1)
            > 0
        )

        if valid.any():

            loss = (
                1.0
                - tversky[valid]
            ).pow(gamma)

            total_loss += loss.mean()

            valid_classes += 1

    if valid_classes == 0:

        return torch.tensor(
            0.0,
            device=logits.device,
            requires_grad=True,
        )

    return (
        total_loss
        / valid_classes
    )


# ============================================================
# COMBINED LOSS
# Focal-Tversky + Cross-Entropy
# ============================================================

def combined_loss(
    logits,
    targets,
):

    ce = F.cross_entropy(
        logits,
        targets,
    )

    focal_tversky = focal_tversky_loss(
        logits,
        targets,
    )

    return (
        0.5 * ce
        + 0.5 * focal_tversky
    )


# ============================================================
# LOAD CASES
# ============================================================

def load_cases(split_name):

    cases = []

    with open(
        SPLIT_CSV,
        "r",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            if row["split"] == split_name:

                cases.append(
                    DATASET_ROOT
                    / row["case_id"]
                )

    return cases


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print(
    "MBH-Seg25 — FOCAL-TVERSKY 2D U-NET"
)
print("=" * 70)

print(
    f"\nDevice: {device}"
)

train_cases = load_cases(
    "train"
)

val_cases = load_cases(
    "validation"
)

print(
    f"\nTraining cases: {len(train_cases)}"
)

print(
    f"Validation cases: {len(val_cases)}"
)

if len(train_cases) == 0:
    raise RuntimeError(
        "No training cases found."
    )

if len(val_cases) == 0:
    raise RuntimeError(
        "No validation cases found."
    )


# ------------------------------------------------------------
# Datasets
# ------------------------------------------------------------

print(
    "\nBuilding foreground-aware training dataset..."
)

train_dataset = MBHSliceDataset(
    train_cases,
    foreground_probability=0.75,
)

print(
    "Building validation dataset..."
)

val_dataset = MBHSliceDataset(
    val_cases,
    foreground_probability=0.75,
)

print(
    f"Training slices: {len(train_dataset)}"
)

print(
    f"Validation slices: {len(val_dataset)}"
)


# ------------------------------------------------------------
# DataLoaders
# ------------------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

model = SmallUNet(
    num_classes=NUM_CLASSES
).to(device)


# ------------------------------------------------------------
# Optimizer
# ------------------------------------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

best_val_loss = float("inf")

print(
    "\n" + "=" * 70
)

print(
    "STARTING FOCAL-TVERSKY TRAINING"
)

print(
    "=" * 70
)

for epoch in range(
    1,
    EPOCHS + 1,
):

    model.train()

    train_loss_total = 0.0

    for images, masks in train_loader:

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        logits = model(images)

        loss = combined_loss(
            logits,
            masks,
        )

        loss.backward()

        optimizer.step()

        train_loss_total += (
            loss.item()
            * images.size(0)
        )

    train_loss = (
        train_loss_total
        / len(train_dataset)
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    val_loss_total = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)

            loss = combined_loss(
                logits,
                masks,
            )

            val_loss_total += (
                loss.item()
                * images.size(0)
            )

    val_loss = (
        val_loss_total
        / len(val_dataset)
    )

    print(
        f"\nEpoch {epoch}/{EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Val Loss:   {val_loss:.4f}"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "alpha": ALPHA,
                "beta": BETA,
                "gamma": GAMMA,
            },
            CHECKPOINT_PATH,
        )

        print(
            "\nBest checkpoint saved:"
        )

        print(
            CHECKPOINT_PATH
        )


# ============================================================
# COMPLETE
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "FOCAL-TVERSKY TRAINING COMPLETE"
)

print(
    "=" * 70
)

print(
    f"\nBest validation loss: "
    f"{best_val_loss:.4f}"
)

print(
    f"Checkpoint: "
    f"{CHECKPOINT_PATH}"
)

print(
    "\nNext step: evaluate this model "
    "on the untouched MBH-Seg25 test cases."
)

print(
    "=" * 70
)