import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.dataset import TBIHemoCTDataset
from src.classification.transforms import eval_transform

TEST_CSV = PROJECT_ROOT / "preprocessing/classification/prepared/test.csv"
CHECKPOINT_PATH = PROJECT_ROOT / "models/classification/best_efficientnet_b0_frozen_tbihemoct.pth"
THRESHOLD_PATH = PROJECT_ROOT / "models/classification/c1_validation_optimal_thresholds.npy"
OUTPUT_DIR = PROJECT_ROOT / "results/classification/c1_frozen_test"

CLASS_NAMES = [
    "epidural",
    "subdural",
    "subarachnoid",
    "intraparenchymal",
    "intraventricular",
]

BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    thresholds = np.load(THRESHOLD_PATH)
    if thresholds.shape != (len(CLASS_NAMES),):
        raise ValueError(f"Expected 5 thresholds; received shape {thresholds.shape}")
    if not np.isfinite(thresholds).all() or not ((thresholds >= 0) & (thresholds <= 1)).all():
        raise ValueError("Thresholds must be finite values between 0 and 1.")

    print("Device:", DEVICE)
    print("Loading test set...")
    dataset = TBIHemoCTDataset(str(TEST_CSV), transform=eval_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print("Test samples:", len(dataset))

    print("Loading checkpoint...")
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    all_labels = []
    all_probabilities = []

    print("Evaluating test set...")
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(DEVICE))
            all_probabilities.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.numpy())

    y_true = np.concatenate(all_labels, axis=0).astype(int)
    y_prob = np.concatenate(all_probabilities, axis=0)

    if len(y_true) != len(dataset):
        raise RuntimeError("Prediction count does not match test-set size.")

    threshold_sets = {
        "default_0.5": np.full(len(CLASS_NAMES), 0.5),
        "validation_selected": thresholds,
    }

    overall_rows = []
    per_class_rows = []

    for setting, setting_thresholds in threshold_sets.items():
        y_pred = (y_prob >= setting_thresholds.reshape(1, -1)).astype(int)

        overall_rows.append({
            "setting": setting,
            "test_samples": len(y_true),
            "checkpoint_epoch": checkpoint.get("epoch", np.nan),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
            "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "macro_sensitivity": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "macro_specificity": np.mean([
                confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1]).ravel()[0]
                / max(1, confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1]).ravel()[0]
                      + confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1]).ravel()[1])
                for i in range(len(CLASS_NAMES))
            ]),
            "macro_auroc": np.nanmean([
                roc_auc_score(y_true[:, i], y_prob[:, i])
                for i in range(len(CLASS_NAMES))
                if len(np.unique(y_true[:, i])) == 2
            ]),
            "macro_auprc": np.nanmean([
                average_precision_score(y_true[:, i], y_prob[:, i])
                for i in range(len(CLASS_NAMES))
                if np.sum(y_true[:, i]) > 0
            ]),
        })

        for i, name in enumerate(CLASS_NAMES):
            tn, fp, fn, tp = confusion_matrix(
                y_true[:, i], y_pred[:, i], labels=[0, 1]
            ).ravel()

            per_class_rows.append({
                "setting": setting,
                "class": name,
                "threshold": float(setting_thresholds[i]),
                "f1": f1_score(y_true[:, i], y_pred[:, i], zero_division=0),
                "precision": precision_score(y_true[:, i], y_pred[:, i], zero_division=0),
                "sensitivity": recall_score(y_true[:, i], y_pred[:, i], zero_division=0),
                "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
                "auroc": roc_auc_score(y_true[:, i], y_prob[:, i])
                if len(np.unique(y_true[:, i])) == 2 else np.nan,
                "auprc": average_precision_score(y_true[:, i], y_prob[:, i])
                if np.sum(y_true[:, i]) > 0 else np.nan,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            })

    pd.DataFrame(overall_rows).to_csv(OUTPUT_DIR / "overall_metrics.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(OUTPUT_DIR / "per_class_metrics.csv", index=False)

    pd.DataFrame({
        "class": CLASS_NAMES,
        "validation_selected_threshold": thresholds,
    }).to_csv(OUTPUT_DIR / "thresholds.csv", index=False)

    print("\nSaved:")
    print(OUTPUT_DIR / "overall_metrics.csv")
    print(OUTPUT_DIR / "per_class_metrics.csv")
    print(OUTPUT_DIR / "thresholds.csv")
    print("\nOverall metrics:")
    print(pd.DataFrame(overall_rows).to_string(index=False))
    print("\nReport generation complete.")


if __name__ == "__main__":
    main()