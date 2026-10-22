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

from src.classification.transforms import (
    train_transform,
    eval_transform,
)


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_CSV = r"preprocessing\classification\prepared\train.csv"
VAL_CSV = r"preprocessing\classification\prepared\validation.csv"

BATCH_SIZE = 16
NUM_WORKERS = 0

LEARNING_RATE = 1e-4
EPOCHS = 1

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


MODEL_DIR = r"models\classification"

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


CHECKPOINT_PATH = os.path.join(
    MODEL_DIR,
    "best_efficientnet_b0_tbihemoct.pth"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("TBI-HemoCT — EXPERIMENT C1")
print("EfficientNet-B0 Single-Window Baseline")
print("=" * 70)

print()
print("Device:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# DATASET
# ============================================================

print()
print("Loading training dataset...")

train_dataset = TBIHemoCTDataset(
    TRAIN_CSV,
    transform=train_transform,
)

print(
    "Training samples:",
    len(train_dataset)
)


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
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)


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


# Replace ImageNet classifier
# with five hemorrhage subtype outputs.

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    NUM_CLASSES
)


model = model.to(DEVICE)


# ============================================================
# LOSS
# ============================================================

criterion = nn.BCEWithLogitsLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    labels,
    probabilities,
    threshold=0.5,
):

    predictions = (
        probabilities >= threshold
    ).astype(np.int32)


    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )


    micro_f1 = f1_score(
        labels,
        predictions,
        average="micro",
        zero_division=0,
    )


    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    macro_precision = precision_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )


    # --------------------------------------------------------
    # Sensitivity / Recall
    # --------------------------------------------------------

    macro_sensitivity = recall_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )


    # --------------------------------------------------------
    # Specificity
    # --------------------------------------------------------

    specificities = []


    for class_index in range(NUM_CLASSES):

        tn, fp, fn, tp = confusion_matrix(
            labels[:, class_index],
            predictions[:, class_index],
            labels=[0, 1],
        ).ravel()


        if (tn + fp) > 0:

            specificity = (
                tn / (tn + fp)
            )

        else:

            specificity = 0.0


        specificities.append(
            specificity
        )


    macro_specificity = float(
        np.mean(specificities)
    )


    # --------------------------------------------------------
    # AUROC and AUPRC
    # --------------------------------------------------------

    aurocs = []
    auprcs = []


    for class_index in range(NUM_CLASSES):

        y_true = labels[:, class_index]

        y_prob = probabilities[:, class_index]


        # AUROC requires both classes.

        if len(np.unique(y_true)) == 2:

            auroc = roc_auc_score(
                y_true,
                y_prob,
            )

        else:

            auroc = np.nan


        # Average precision / PR-AUC.

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


    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_sensitivity": macro_sensitivity,
        "macro_specificity": macro_specificity,
        "macro_auroc": macro_auroc,
        "macro_auprc": macro_auprc,
        "class_f1": f1_score(
            labels,
            predictions,
            average=None,
            zero_division=0,
        ),
        "class_precision": precision_score(
            labels,
            predictions,
            average=None,
            zero_division=0,
        ),
        "class_sensitivity": recall_score(
            labels,
            predictions,
            average=None,
            zero_division=0,
        ),
        "class_specificity": specificities,
        "class_auroc": aurocs,
        "class_auprc": auprcs,
    }


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0


    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)


        optimizer.zero_grad()


        logits = model(images)


        loss = criterion(
            logits,
            labels
        )


        loss.backward()

        optimizer.step()


        running_loss += (
            loss.item()
            * images.size(0)
        )


    return (
        running_loss
        / len(train_dataset)
    )


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def validate():

    model.eval()

    running_loss = 0.0

    all_labels = []
    all_probabilities = []


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)


            logits = model(images)


            loss = criterion(
                logits,
                labels
            )


            running_loss += (
                loss.item()
                * images.size(0)
            )


            probabilities = torch.sigmoid(
                logits
            )


            all_labels.append(
                labels.cpu().numpy()
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


    metrics = calculate_metrics(
        all_labels,
        all_probabilities,
        threshold=0.5,
    )


    val_loss = (
        running_loss
        / len(val_dataset)
    )


    return (
        val_loss,
        metrics,
    )


# ============================================================
# TRAINING LOOP
# ============================================================

best_macro_f1 = -1.0


print()
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)


for epoch in range(
    1,
    EPOCHS + 1
):


    train_loss = train_one_epoch()


    val_loss, metrics = validate()


    print()
    print(
        f"Epoch {epoch}/{EPOCHS}"
    )


    print(
        f"Train Loss: "
        f"{train_loss:.4f}"
    )


    print(
        f"Validation Loss: "
        f"{val_loss:.4f}"
    )


    print(
        f"Macro-F1: "
        f"{metrics['macro_f1']:.4f}"
    )


    print(
        f"Micro-F1: "
        f"{metrics['micro_f1']:.4f}"
    )


    print(
        f"Macro Precision: "
        f"{metrics['macro_precision']:.4f}"
    )


    print(
        f"Macro Sensitivity: "
        f"{metrics['macro_sensitivity']:.4f}"
    )


    print(
        f"Macro Specificity: "
        f"{metrics['macro_specificity']:.4f}"
    )


    print(
        f"Macro AUROC: "
        f"{metrics['macro_auroc']:.4f}"
    )


    print(
        f"Macro AUPRC: "
        f"{metrics['macro_auprc']:.4f}"
    )


    print()
    print("Per-class results:")


    for i, name in enumerate(
        LABEL_NAMES
    ):

        print(
            f"  {name:18s}"
            f" F1={metrics['class_f1'][i]:.4f}"
            f" Sens={metrics['class_sensitivity'][i]:.4f}"
            f" Spec={metrics['class_specificity'][i]:.4f}"
            f" AUROC={metrics['class_auroc'][i]:.4f}"
            f" AUPRC={metrics['class_auprc'][i]:.4f}"
        )


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if (
        metrics["macro_f1"]
        > best_macro_f1
    ):

        best_macro_f1 = (
            metrics["macro_f1"]
        )


        torch.save(
            {
                "epoch": epoch,
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "val_macro_f1":
                    metrics["macro_f1"],
                "val_macro_auroc":
                    metrics["macro_auroc"],
                "val_macro_auprc":
                    metrics["macro_auprc"],
            },
            CHECKPOINT_PATH,
        )


        print()
        print(
            "Saved best model:",
            CHECKPOINT_PATH
        )


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 70)
print("EXPERIMENT C1 COMPLETE")
print("=" * 70)


print(
    f"Best Validation Macro-F1: "
    f"{best_macro_f1:.4f}"
)


print()
print(
    "Checkpoint:",
    CHECKPOINT_PATH
)