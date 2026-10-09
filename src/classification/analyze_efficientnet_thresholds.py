import sys
import os

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
)

from sklearn.metrics import f1_score


# ============================================================
# PROJECT PATH
# ============================================================

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            ".."
        )
    )
)

from src.classification.dataset import TBIHemoCTDataset
from src.classification.transforms import eval_transform


# ============================================================
# CONFIGURATION
# ============================================================

VAL_CSV = r"preprocessing\classification\prepared\validation.csv"

CHECKPOINT_PATH = (
    r"models\classification"
    r"\best_efficientnet_b0_frozen_tbihemoct.pth"
)

BATCH_SIZE = 32
NUM_WORKERS = 0
NUM_CLASSES = 5

LABEL_NAMES = [
    "epidural",
    "subdural",
    "subarachnoid",
    "intraparenchymal",
    "intraventricular",
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("C1-FROZEN — VALIDATION THRESHOLD ANALYSIS")
print("EfficientNet-B0")
print("=" * 70)

print()
print("Device:", DEVICE)


# ============================================================
# DATASET
# ============================================================

print()
print("Loading validation dataset...")

val_dataset = TBIHemoCTDataset(
    VAL_CSV,
    transform=eval_transform,
)

print(
    "Validation samples:",
    len(val_dataset)
)


# ============================================================
# DATALOADER
# ============================================================

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)


# ============================================================
# MODEL
# ============================================================

print()
print("Loading EfficientNet-B0...")

weights = EfficientNet_B0_Weights.DEFAULT

model = efficientnet_b0(
    weights=weights
)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    NUM_CLASSES
)

model = model.to(DEVICE)


# ============================================================
# LOAD C1 CHECKPOINT
# ============================================================

print()
print("Loading C1 checkpoint...")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print(
    "Checkpoint epoch:",
    checkpoint["epoch"]
)

print(
    "Validation Macro-F1 during training:",
    f"{checkpoint['val_macro_f1']:.4f}"
)


# ============================================================
# GET VALIDATION PROBABILITIES
# ============================================================

model.eval()

all_labels = []
all_probabilities = []


print()
print("Generating validation probabilities...")


with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(DEVICE)

        logits = model(images)

        probabilities = torch.sigmoid(
            logits
        )

        all_labels.append(
            labels.numpy()
        )

        all_probabilities.append(
            probabilities.cpu().numpy()
        )


all_labels = np.concatenate(
    all_labels,
    axis=0
)

all_probabilities = np.concatenate(
    all_probabilities,
    axis=0
)


# ============================================================
# THRESHOLD SEARCH
# ============================================================

thresholds = np.arange(
    0.05,
    0.96,
    0.01
)

best_thresholds = []
best_f1_scores = []


print()
print("=" * 70)
print("VALIDATION THRESHOLD OPTIMIZATION")
print("=" * 70)


for class_index, class_name in enumerate(
    LABEL_NAMES
):

    y_true = all_labels[:, class_index]

    y_prob = all_probabilities[:, class_index]


    best_threshold = 0.5
    best_f1 = -1.0


    for threshold in thresholds:

        y_pred = (
            y_prob >= threshold
        ).astype(np.int32)


        score = f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )


        if score > best_f1:

            best_f1 = score
            best_threshold = threshold


    best_thresholds.append(
        float(best_threshold)
    )

    best_f1_scores.append(
        float(best_f1)
    )


    default_predictions = (
        y_prob >= 0.5
    ).astype(np.int32)


    default_f1 = f1_score(
        y_true,
        default_predictions,
        zero_division=0,
    )


    print()
    print(
        f"{class_name:18s}"
    )

    print(
        f"  Default threshold: 0.50"
    )

    print(
        f"  Default F1:       {default_f1:.4f}"
    )

    print(
        f"  Best threshold:    {best_threshold:.2f}"
    )

    print(
        f"  Best validation F1: {best_f1:.4f}"
    )


# ============================================================
# SAVE THRESHOLDS
# ============================================================

OUTPUT_PATH = (
    r"models\classification"
    r"\c1_validation_optimal_thresholds.npy"
)

np.save(
    OUTPUT_PATH,
    np.array(
        best_thresholds,
        dtype=np.float32
    )
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("THRESHOLD ANALYSIS COMPLETE")
print("=" * 70)

print()
print("Class-specific thresholds:")

for name, threshold in zip(
    LABEL_NAMES,
    best_thresholds
):

    print(
        f"  {name:18s}: {threshold:.2f}"
    )


print()
print(
    "Saved thresholds:",
    OUTPUT_PATH
)

print()
print(
    "IMPORTANT: These thresholds were selected using"
)

print(
    "validation data only."
)

print(
    "The test set has NOT been used."
)