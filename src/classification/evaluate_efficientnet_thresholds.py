import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import models

from src.classification.dataset import TBIHemoCTDataset
from src.classification.transforms import eval_transform


# ============================================================
# C1 — TEST EVALUATION WITH VALIDATION-OPTIMIZED THRESHOLDS
# ============================================================

CSV_PATH = "preprocessing/classification/prepared/test.csv"
CHECKPOINT_PATH = "models/classification/best_efficientnet_b0_frozen_tbihemoct.pth"
THRESHOLD_PATH = "models/classification/c1_validation_optimal_thresholds.npy"

BATCH_SIZE = 32
NUM_WORKERS = 0
NUM_CLASSES = 5

CLASS_NAMES = [
    "epidural",
    "subdural",
    "subarachnoid",
    "intraparenchymal",
    "intraventricular",
]


print("=" * 70)
print("C1-FROZEN — TEST EVALUATION WITH OPTIMIZED THRESHOLDS")
print("EfficientNet-B0")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")


# ------------------------------------------------------------
# Load test dataset
# ------------------------------------------------------------

print("\nLoading test dataset...")

test_dataset = TBIHemoCTDataset(
    csv_path=CSV_PATH,
    transform=eval_transform,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)

print(f"Test samples: {len(test_dataset)}")


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

print("\nLoading EfficientNet-B0...")

model = models.efficientnet_b0(weights=None)

model.classifier[1] = torch.nn.Linear(
    model.classifier[1].in_features,
    NUM_CLASSES,
)

model = model.to(device)


print("\nLoading C1 checkpoint...")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False,
)

model.load_state_dict(checkpoint["model_state_dict"])

print(f"Checkpoint epoch: {checkpoint['epoch']}")


# ------------------------------------------------------------
# Load validation-optimized thresholds
# ------------------------------------------------------------

print("\nLoading validation-optimized thresholds...")

thresholds = np.load(THRESHOLD_PATH)

print("\nClass-specific thresholds:")

for class_name, threshold in zip(CLASS_NAMES, thresholds):
    print(f"  {class_name:18s}: {threshold:.2f}")


# ------------------------------------------------------------
# Generate test probabilities
# ------------------------------------------------------------

print("\nGenerating test probabilities...")

model.eval()

all_probs = []
all_targets = []

with torch.no_grad():

    for images, targets in test_loader:

        images = images.to(device)

        outputs = model(images)

        probabilities = torch.sigmoid(outputs)

        all_probs.append(probabilities.cpu().numpy())
        all_targets.append(targets.numpy())


y_prob = np.concatenate(all_probs, axis=0)
y_true = np.concatenate(all_targets, axis=0)


# ------------------------------------------------------------
# Thresholded predictions
# ------------------------------------------------------------

y_pred = np.zeros_like(y_prob, dtype=int)

for class_idx in range(NUM_CLASSES):

    y_pred[:, class_idx] = (
        y_prob[:, class_idx] >= thresholds[class_idx]
    ).astype(int)


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)


print("\n" + "=" * 70)
print("TEST RESULTS — OPTIMIZED THRESHOLDS")
print("=" * 70)


f1_macro = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)

f1_micro = f1_score(
    y_true,
    y_pred,
    average="micro",
    zero_division=0,
)

precision_macro = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)

sensitivity_macro = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)


print(f"\nMacro-F1:          {f1_macro:.4f}")
print(f"Micro-F1:          {f1_micro:.4f}")
print(f"Macro Precision:   {precision_macro:.4f}")
print(f"Macro Sensitivity: {sensitivity_macro:.4f}")


print("\n" + "-" * 70)
print("PER-CLASS RESULTS")
print("-" * 70)


for class_idx, class_name in enumerate(CLASS_NAMES):

    y_true_class = y_true[:, class_idx]
    y_pred_class = y_pred[:, class_idx]
    y_prob_class = y_prob[:, class_idx]

    f1 = f1_score(
        y_true_class,
        y_pred_class,
        zero_division=0,
    )

    precision = precision_score(
        y_true_class,
        y_pred_class,
        zero_division=0,
    )

    sensitivity = recall_score(
        y_true_class,
        y_pred_class,
        zero_division=0,
    )

    specificity = (
        ((y_true_class == 0) & (y_pred_class == 0)).sum()
        / (y_true_class == 0).sum()
    )

    try:
        auroc = roc_auc_score(
            y_true_class,
            y_prob_class,
        )
    except ValueError:
        auroc = float("nan")

    try:
        auprc = average_precision_score(
            y_true_class,
            y_prob_class,
        )
    except ValueError:
        auprc = float("nan")

    print(f"\n{class_name}")
    print(f"  Threshold:   {thresholds[class_idx]:.2f}")
    print(f"  F1:          {f1:.4f}")
    print(f"  Precision:   {precision:.4f}")
    print(f"  Sensitivity: {sensitivity:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  AUROC:       {auroc:.4f}")
    print(f"  AUPRC:       {auprc:.4f}")


print("\n" + "=" * 70)
print("IMPORTANT")
print("=" * 70)

print(
    "\nThresholds were optimized using the validation set only."
)

print(
    "The test set was used only once for final evaluation."
)

print(
    "\nThis provides a leakage-safe comparison of "
    "default vs validation-optimized thresholds."
)

print("\n" + "=" * 70)