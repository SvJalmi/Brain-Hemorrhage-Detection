import sys
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights

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
from src.classification.transforms import train_transform


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_CSV = r"preprocessing\classification\prepared\train.csv"
VAL_CSV = r"preprocessing\classification\prepared\validation.csv"

BATCH_SIZE = 32
NUM_WORKERS = 0

LEARNING_RATE = 1e-4
EPOCHS = 3

NUM_CLASSES = 5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_DIR = r"models\classification"

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# DATASET
# ============================================================

print("=" * 70)
print("TBI-HemoCT BASELINE CLASSIFICATION")
print("=" * 70)

print()
print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


print()
print("Loading training dataset...")

train_dataset = TBIHemoCTDataset(
    TRAIN_CSV,
    transform=train_transform,
)

print("Training samples:", len(train_dataset))


print()
print("Loading validation dataset...")

val_dataset = TBIHemoCTDataset(
    VAL_CSV,
    transform=train_transform,
)

print("Validation samples:", len(val_dataset))


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
print("Loading ResNet-18...")

weights = ResNet18_Weights.DEFAULT

model = resnet18(weights=weights)

# Replace ImageNet's 1000-class output
# with our 5 hemorrhage labels.

model.fc = nn.Linear(
    model.fc.in_features,
    NUM_CLASSES
)

model = model.to(DEVICE)


# ============================================================
# LOSS AND OPTIMIZER
# ============================================================

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


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
            loss.item() * images.size(0)
        )

    return running_loss / len(train_dataset)


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def validate():

    model.eval()

    running_loss = 0.0

    all_labels = []
    all_predictions = []

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
                loss.item() * images.size(0)
            )

            probabilities = torch.sigmoid(logits)

            predictions = (
                probabilities >= 0.5
            ).float()

            all_labels.append(
                labels.cpu()
            )

            all_predictions.append(
                predictions.cpu()
            )

    all_labels = torch.cat(all_labels).numpy()
    all_predictions = torch.cat(all_predictions).numpy()

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    micro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="micro",
        zero_division=0,
    )

    val_loss = running_loss / len(val_dataset)

    return val_loss, macro_f1, micro_f1


# ============================================================
# TRAINING LOOP
# ============================================================

best_macro_f1 = -1.0

print()
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)

for epoch in range(1, EPOCHS + 1):

    train_loss = train_one_epoch()

    val_loss, macro_f1, micro_f1 = validate()

    print()
    print(
        f"Epoch {epoch}/{EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Validation Loss: {val_loss:.4f}"
    )

    print(
        f"Validation Macro-F1: {macro_f1:.4f}"
    )

    print(
        f"Validation Micro-F1: {micro_f1:.4f}"
    )

    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if macro_f1 > best_macro_f1:

        best_macro_f1 = macro_f1

        checkpoint_path = os.path.join(
            MODEL_DIR,
            "best_resnet18_tbihemoct.pth"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": macro_f1,
                "val_micro_f1": micro_f1,
            },
            checkpoint_path,
        )

        print(
            "Saved best model:",
            checkpoint_path
        )


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 70)
print("BASELINE TRAINING COMPLETE")
print("=" * 70)

print(
    f"Best Validation Macro-F1: {best_macro_f1:.4f}"
)

print()
print(
    "Model directory:",
    MODEL_DIR
)