import os
import ast

import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


LABEL_COLS = [
    "epidural",
    "subdural",
    "subarachnoid",
    "intraparenchymal",
    "intraventricular",
]


class TBIHemoCTDataset(Dataset):

    def __init__(
        self,
        csv_path,
        transform=None,
    ):

        self.df = pd.read_csv(csv_path)
        self.transform = transform

        required = ["image_path"] + LABEL_COLS

        missing = [
            c for c in required
            if c not in self.df.columns
        ]

        if missing:
            raise RuntimeError(
                f"Missing columns: {missing}"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_path = row["image_path"]

        if not os.path.isfile(image_path):
            raise FileNotFoundError(image_path)

        # ----------------------------------------------------
        # Load CT PNG
        # ----------------------------------------------------

        image = Image.open(image_path).convert("L")

        # ----------------------------------------------------
        # Convert grayscale → 3 channels
        #
        # EfficientNet expects 3-channel input.
        # We initially duplicate the CT grayscale channel.
        # ----------------------------------------------------

        image = image.convert("RGB")

        # ----------------------------------------------------
        # Transform
        # ----------------------------------------------------

        if self.transform is not None:
            image = self.transform(image)

        # ----------------------------------------------------
        # Multi-label target
        # ----------------------------------------------------

        labels = torch.tensor(
            row[LABEL_COLS].astype(float).values,
            dtype=torch.float32,
        )

        return image, labels