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

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


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

TEST_CSV = r"preprocessing\classification\prepared\test.csv"

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
print("TBI-HemoCT — EXPERIMENT C1-FROZEN TEST EVALUATION")
print("EfficientNet-B0 Frozen-Backbone")
print("=" * 70)

print()
print("Device:", DEVICE)

print(
    "Checkpoint:",
    CHECKPOINT_PATH
)


# ============================================================
# TEST DATASET
# ============================================================

print()
print("Loading test dataset...")

test_dataset = TBIHemoCTDataset(
    TEST_CSV,
    transform=eval_transform,
)

print(
    "Test samples:",
    len(test_dataset)
)


# ============================================================
# TEST DATALOADER
# ============================================================

test_loader = DataLoader(
    test_dataset,
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


# ============================================================
# FREEZE FEATURE EXTRACTOR
# ============================================================

for parameter in model.features.parameters():
    parameter.requires_grad = False


# ============================================================
# CLASSIFICATION HEAD
# ============================================================

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    NUM_CLASSES
)


model = model.to(DEVICE)


# ============================================================
# LOAD BEST CHECKPOINT
# ============================================================

print()
print("Loading best validation checkpoint...")

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
    "Validation Macro-F1:",
    f"{checkpoint['val_macro_f1']:.4f}"
)

print(
    "Validation Macro AUROC:",
    f"{checkpoint['val_macro_auroc']:.4f}"
)

print(
    "Validation Macro AUPRC:",
    f"{checkpoint['val_macro_auprc']:.4f}"
)


# ============================================================
# EVALUATION
# ============================================================

model.eval()

all_labels = []
all_probabilities = []


print()
print("Evaluating untouched test set...")


with torch.no_grad():

    for images, labels in test_loader:

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
# THRESHOLD
# ============================================================

THRESHOLD = 0.5

predictions = (
    all_probabilities >= THRESHOLD
).astype(np.int32)


# ============================================================
# OVERALL METRICS
# ============================================================

macro_f1 = f1_score(
    all_labels,
    predictions,
    average="macro",
    zero_division=0,
)

micro_f1 = f1_score(
    all_labels,
    predictions,
    average="micro",
    zero_division=0,
)

macro_precision = precision_score(
    all_labels,
    predictions,
    average="macro",
    zero_division=0,
)

macro_sensitivity = recall_score(
    all_labels,
    predictions,
    average="macro",
    zero_division=0,
)


# ============================================================
# SPECIFICITY
# ============================================================

specificities = []

for class_index in range(NUM_CLASSES):

    tn, fp, fn, tp = confusion_matrix(
        all_labels[:, class_index],
        predictions[:, class_index],
        labels=[0, 1],
    ).ravel()

    if (tn + fp) > 0:
        specificity = tn / (tn + fp)
    else:
        specificity = 0.0

    specificities.append(
        specificity
    )


macro_specificity = float(
    np.mean(specificities)
)


# ============================================================
# AUROC / AUPRC
# ============================================================

aurocs = []
auprcs = []

for class_index in range(NUM_CLASSES):

    y_true = all_labels[:, class_index]
    y_prob = all_probabilities[:, class_index]

    if len(np.unique(y_true)) == 2:

        auroc = roc_auc_score(
            y_true,
            y_prob,
        )

    else:

        auroc = np.nan


    if np.sum(y_true) > 0:

        auprc = average_precision_score(
            y_true,
            y_prob,
        )

    else:

        auprc = np.nan


    aurocs.append(auroc)
    auprcs.append(auprc)


macro_auroc = float(
    np.nanmean(aurocs)
)

macro_auprc = float(
    np.nanmean(auprcs)
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("C1-FROZEN TEST RESULTS")
print("=" * 70)

print(
    f"Macro-F1:          {macro_f1:.4f}"
)

print(
    f"Micro-F1:          {micro_f1:.4f}"
)

print(
    f"Macro Precision:   {macro_precision:.4f}"
)

print(
    f"Macro Sensitivity: {macro_sensitivity:.4f}"
)

print(
    f"Macro Specificity: {macro_specificity:.4f}"
)

print(
    f"Macro AUROC:       {macro_auroc:.4f}"
)

print(
    f"Macro AUPRC:       {macro_auprc:.4f}"
)


# ============================================================
# PER-CLASS RESULTS
# ============================================================

print()
print("Per-class test results:")

class_f1 = f1_score(
    all_labels,
    predictions,
    average=None,
    zero_division=0,
)

class_precision = precision_score(
    all_labels,
    predictions,
    average=None,
    zero_division=0,
)

class_sensitivity = recall_score(
    all_labels,
    predictions,
    average=None,
    zero_division=0,
)


for i, name in enumerate(LABEL_NAMES):

    print(
        f"  {name:18s}"
        f" F1={class_f1[i]:.4f}"
        f" Precision={class_precision[i]:.4f}"
        f" Sens={class_sensitivity[i]:.4f}"
        f" Spec={specificities[i]:.4f}"
        f" AUROC={aurocs[i]:.4f}"
        f" AUPRC={auprcs[i]:.4f}"
    )


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 70)
print("C1-FROZEN TEST EVALUATION COMPLETE")
print("=" * 70)