from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

SEED = 42
LABELS = ["epidural", "subdural", "subarachnoid",
          "intraparenchymal", "intraventricular"]

LABELS_CSV = Path("preprocessing/classification/tbihemoct_labels.csv")

IMG_SIZE = 224
MEAN = (0.485, 0.456, 0.406)   # ImageNet stats, because we use pretrained weights
STD = (0.229, 0.224, 0.225)


def get_transforms(train, img_size=IMG_SIZE):
    """Train gets light geometric and intensity augmentation; eval gets none."""
    if train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Affine(
                rotate=(-10, 10),
                scale=(0.95, 1.05),
                translate_percent=(-0.03, 0.03),
                p=0.7,
            ),
            A.RandomBrightnessContrast(0.1, 0.1, p=0.5),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


class TBIDataset(Dataset):
    def __init__(self, split, csv_path=LABELS_CSV, transform=None):
        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No rows for split '{split}'")
        self.paths = self.df["image_path"].tolist()
        self.labels = self.df[LABELS].to_numpy(dtype=np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.paths[idx]
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        image = np.stack([gray, gray, gray], axis=-1)   # 1 channel -> 3 for pretrained nets
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        label = torch.from_numpy(self.labels[idx])
        return image, label, path


def compute_pos_weight(split="train", csv_path=LABELS_CSV):
    """neg/pos ratio per class, from the TRAIN split only."""
    df = pd.read_csv(csv_path)
    df = df[df["split"] == split]
    pos = df[LABELS].sum().to_numpy(dtype=np.float32)
    neg = len(df) - pos
    return torch.tensor(neg / pos, dtype=torch.float32)


def build_dataloaders(batch_size=32, num_workers=4, img_size=IMG_SIZE):
    train_ds = TBIDataset("train", transform=get_transforms(True, img_size))
    val_ds = TBIDataset("validation", transform=get_transforms(False, img_size))
    test_ds = TBIDataset("test", transform=get_transforms(False, img_size))

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Smoke test. The __main__ guard is required on Windows when num_workers > 0.
    train_loader, val_loader, test_loader = build_dataloaders(batch_size=16, num_workers=2)

    print("Batches  train/val/test:",
          len(train_loader), len(val_loader), len(test_loader))

    images, labels, paths = next(iter(train_loader))
    print("Image batch:", tuple(images.shape), images.dtype)
    print("Label batch:", tuple(labels.shape), labels.dtype)
    print("Image value range: %.2f to %.2f" % (images.min(), images.max()))
    print("Positive rate in this batch per class:", labels.mean(dim=0).tolist())
    print("pos_weight (train):",
          dict(zip(LABELS, [round(v, 2) for v in compute_pos_weight().tolist()])))

    # Save a grid of augmented training images for a visual check.
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    tiles = []
    for img in images[:8]:
        img = (img * std + mean).clamp(0, 1)
        tiles.append((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    grid = np.concatenate(
        [np.concatenate(tiles[:4], axis=1), np.concatenate(tiles[4:8], axis=1)], axis=0
    )
    cv2.imwrite("sample_batch.png", cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print("Saved sample_batch.png (open it and check the images look like brain CT)")