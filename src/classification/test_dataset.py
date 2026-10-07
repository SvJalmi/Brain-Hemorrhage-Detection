import sys
import os

import torch
from torch.utils.data import DataLoader

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


CSV = r"preprocessing\classification\prepared\train.csv"


dataset = TBIHemoCTDataset(
    CSV,
    transform=train_transform,
)


print("=" * 70)
print("DATASET TEST")
print("=" * 70)

print("Dataset size:", len(dataset))


image, label = dataset[0]

print("Image shape:", image.shape)
print("Image dtype:", image.dtype)

print("Label:", label)
print("Label shape:", label.shape)

print()
print("Image min:", image.min().item())
print("Image max:", image.max().item())

print()
print("STATUS: DATASET LOADING SUCCESSFUL")