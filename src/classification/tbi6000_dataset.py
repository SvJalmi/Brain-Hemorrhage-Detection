from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


CLASS_NAMES = [
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "normal",
    "subarachnoid",
    "subdural",
]

CLASS_TO_INDEX = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}


class TBI6000Dataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.csv_file = Path(csv_file)
        self.transform = transform

        self.data = pd.read_csv(self.csv_file)

        required_columns = ["full_path", "label"]

        missing = [
            column
            for column in required_columns
            if column not in self.data.columns
        ]

        if missing:
            raise RuntimeError(
                f"Missing columns: {missing}"
            )

        unknown_labels = set(self.data["label"]) - set(CLASS_TO_INDEX)

        if unknown_labels:
            raise RuntimeError(
                f"Unknown labels: {unknown_labels}"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        image_path = row["full_path"]
        label_name = row["label"]

        image = Image.open(image_path).convert("RGB")

        label = CLASS_TO_INDEX[label_name]

        if self.transform is not None:
            image = self.transform(image)

        return image, label