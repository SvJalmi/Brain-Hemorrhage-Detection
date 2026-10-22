import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import csv
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    accuracy_score, confusion_matrix,
)

from src.classification.transforms import eval_transform


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "TBI-6000"
CHECKPOINT_PATH = (
    PROJECT_ROOT / "models" / "classification" /
    "best_efficientnet_b0_c2_full_finetune_tbihemoct.pth"
)
OUTPUT_DIR = PROJECT_ROOT / "results" / "classification" / "c3_test"

# These indices must match the C2 training label order exactly.
MODEL_CLASSES = [
    "epidural",
    "subdural",
    "subarachnoid",
    "intraparenchymal",
    "intraventricular",
]
DATASET_CLASSES = [
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "normal",
    "subarachnoid",
    "subdural",
]

BATCH_SIZE = 32
NUM_WORKERS = 0
GPU_ID = 1


class TBI6000FolderDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        for class_name in DATASET_CLASSES:
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing class directory: {class_dir}"
                )

            for path in sorted(class_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in {
                    ".png", ".jpg", ".jpeg", ".bmp"
                }:
                    self.samples.append((path, class_name))

        if not self.samples:
            raise RuntimeError(f"No images found under {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as im:
            image = im.convert("L").convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label, str(path)


def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def main():
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(f"Dataset not found: {DATASET_ROOT}")
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    if torch.cuda.is_available():
        if GPU_ID >= torch.cuda.device_count():
            raise RuntimeError(
                f"GPU {GPU_ID} requested, but only "
                f"{torch.cuda.device_count()} CUDA device(s) are visible."
            )
        device = torch.device(f"cuda:{GPU_ID}")
    else:
        device = torch.device("cpu")

    print("=" * 72)
    print("C3 — CROSS-DATASET EVALUATION: C2 ON TBI-6000")
    print("=" * 72)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Dataset: {DATASET_ROOT}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    dataset = TBI6000FolderDataset(DATASET_ROOT, eval_transform)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    counts = defaultdict(int)
    for _, label in dataset.samples:
        counts[label] += 1
    print("\nDataset counts:")
    for name in DATASET_CLASSES:
        print(f"  {name:20s} {counts[name]:5d}")
    print(f"  {'TOTAL':20s} {len(dataset):5d}")

    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, len(MODEL_CLASSES)
    )
    checkpoint = torch.load(
        CHECKPOINT_PATH, map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"\nLoaded C2 checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")

    all_labels = []
    all_probs = []
    all_paths = []

    print("\nRunning inference...")
    with torch.inference_mode():
        for batch_index, (images, labels, paths) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_labels.extend(labels)
            all_probs.append(probs)
            all_paths.extend(paths)

            if batch_index % 25 == 0 or batch_index == len(loader):
                print(f"  Batch {batch_index}/{len(loader)}")

    probs = np.concatenate(all_probs, axis=0)
    y_true = np.asarray(all_labels)
    predicted_index = probs.argmax(axis=1)
    predicted_class = [MODEL_CLASSES[i] for i in predicted_index]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save image-level probabilities and the highest-scoring subtype.
    prediction_path = OUTPUT_DIR / "image_predictions.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["image_path", "true_class", "top_scoring_hemorrhage_class"]
            + [f"prob_{name}" for name in MODEL_CLASSES]
        )
        for path, label, pred, row_probs in zip(
            all_paths, y_true, predicted_class, probs
        ):
            writer.writerow([path, label, pred] + [f"{x:.8f}" for x in row_probs])

    # Cross-dataset class-wise results are computed only for the five
    # corresponding hemorrhage folders. Normal cases are assessed separately.
    class_rows = []
    for class_index, class_name in enumerate(MODEL_CLASSES):
        mask = y_true == class_name
        binary_true = mask.astype(int)
        binary_pred = probs[:, class_index] >= 0.5

        class_rows.append({
            "class": class_name,
            "support_positive": int(mask.sum()),
            "support_total": int(len(y_true)),
            "precision": precision_score(
                binary_true, binary_pred, zero_division=0
            ),
            "sensitivity": recall_score(
                binary_true, binary_pred, zero_division=0
            ),
            "specificity": (
                float(confusion_matrix(
                    binary_true, binary_pred, labels=[0, 1]
                )[0, 0] /
                max(1, confusion_matrix(
                    binary_true, binary_pred, labels=[0, 1]
                )[0, :].sum()))
            ),
            "f1": f1_score(binary_true, binary_pred, zero_division=0),
            "auroc": safe_auc(binary_true, probs[:, class_index]),
            "auprc": (
                float(average_precision_score(
                    binary_true, probs[:, class_index]
                ))
                if binary_true.sum() > 0 else float("nan")
            ),
        })

    class_csv = OUTPUT_DIR / "per_class_metrics.csv"
    with class_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(class_rows[0].keys()))
        writer.writeheader()
        writer.writerows(class_rows)

    # Normal images have no corresponding normal output in C2.
    # Report their score behavior without treating them as a sixth model class.
    normal_mask = y_true == "normal"
    normal_summary = {
        "dataset": "TBI-6000",
        "total_samples": int(len(y_true)),
        "normal_samples": int(normal_mask.sum()),
        "normal_cases_max_probability_ge_0_5": int(
            (probs[normal_mask].max(axis=1) >= 0.5).sum()
        ),
        "normal_mean_max_hemorrhage_probability": (
            float(probs[normal_mask].max(axis=1).mean())
            if normal_mask.any() else float("nan")
        ),
        "threshold": 0.5,
        "note": (
            "C2 has five independent hemorrhage sigmoid outputs and no normal "
            "output. Normal cases are reported separately; this is not a "
            "six-class classification evaluation."
        ),
    }
    summary_csv = OUTPUT_DIR / "overall_metrics.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(normal_summary.keys()))
        writer.writeheader()
        writer.writerow(normal_summary)

    print("\nPer-class metrics (threshold = 0.5):")
    for row in class_rows:
        print(
            f"{row['class']:20s} "
            f"F1={row['f1']:.4f} "
            f"Precision={row['precision']:.4f} "
            f"Sensitivity={row['sensitivity']:.4f} "
            f"AUROC={row['auroc']:.4f} "
            f"AUPRC={row['auprc']:.4f}"
        )

    print("\nNormal-case assessment:")
    print(f"  Normal samples: {normal_summary['normal_samples']}")
    print(
        "  Mean maximum hemorrhage probability on normal images: "
        f"{normal_summary['normal_mean_max_hemorrhage_probability']:.4f}"
    )
    print("\nSaved:")
    print(f"  {class_csv}")
    print(f"  {summary_csv}")
    print(f"  {prediction_path}")
    print("\nNote: This evaluates the existing C2 model without retraining or threshold tuning.")


if __name__ == "__main__":
    main()
