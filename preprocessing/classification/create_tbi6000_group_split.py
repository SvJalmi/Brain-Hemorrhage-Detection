import os
import re
import random
import pandas as pd
from collections import defaultdict

ROOT = r"datasets\TBI-6000"
OUT = r"preprocessing\classification\tbi6000_group_split.csv"

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

CLASSES = [
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "normal",
    "subarachnoid",
    "subdural",
]


def get_base_id(filename):
    m = re.match(r"(ID_\d+)", filename)
    return m.group(1) if m else None


def main():
    random.seed(SEED)

    rows = []

    print("=" * 70)
    print("TBI-6000 LEAKAGE-SAFE GROUP SPLIT")
    print("=" * 70)

    # ---------------------------------------------------------
    # Collect all files
    # ---------------------------------------------------------
    for cls in CLASSES:
        folder = os.path.join(ROOT, cls)

        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Missing class folder: {folder}")

        for filename in os.listdir(folder):
            if not filename.lower().endswith(".png"):
                continue

            base_id = get_base_id(filename)

            if base_id is None:
                raise ValueError(f"Could not extract ID from: {filename}")

            rows.append({
                "path": os.path.join(cls, filename),
                "label": cls,
                "base_id": base_id,
            })

    df = pd.DataFrame(rows)

    print(f"Total images: {len(df)}")
    print(f"Unique base IDs: {df['base_id'].nunique()}")

    # ---------------------------------------------------------
    # Build group-level class counts
    # ---------------------------------------------------------
    group_counts = (
        df.groupby(["base_id", "label"])
        .size()
        .unstack(fill_value=0)
    )

    for cls in CLASSES:
        if cls not in group_counts.columns:
            group_counts[cls] = 0

    group_counts = group_counts[CLASSES]

    # Shuffle groups deterministically
    groups = list(group_counts.index)
    random.shuffle(groups)

    # ---------------------------------------------------------
    # Target number of images per class
    # ---------------------------------------------------------
    total_class_counts = df["label"].value_counts().to_dict()

    targets = {
        "train": {
            cls: total_class_counts[cls] * TRAIN_RATIO
            for cls in CLASSES
        },
        "validation": {
            cls: total_class_counts[cls] * VAL_RATIO
            for cls in CLASSES
        },
        "test": {
            cls: total_class_counts[cls] * TEST_RATIO
            for cls in CLASSES
        },
    }

    current = {
        "train": {cls: 0 for cls in CLASSES},
        "validation": {cls: 0 for cls in CLASSES},
        "test": {cls: 0 for cls in CLASSES},
    }

    group_split = {}

    # ---------------------------------------------------------
    # Greedy assignment
    # ---------------------------------------------------------
    for group in groups:
        counts = group_counts.loc[group]

        best_split = None
        best_score = float("inf")

        for split in ["train", "validation", "test"]:

            score = 0.0

            for cls in CLASSES:
                target = targets[split][cls]
                after = current[split][cls] + counts[cls]

                # Squared normalized deviation
                if target > 0:
                    score += ((after - target) / target) ** 2

            # Slight penalty for putting too much into a split
            total_target = sum(targets[split].values())
            total_after = sum(current[split].values()) + counts.sum()

            if total_target > 0:
                score += 0.1 * ((total_after - total_target) / total_target) ** 2

            if score < best_score:
                best_score = score
                best_split = split

        group_split[group] = best_split

        for cls in CLASSES:
            current[best_split][cls] += int(counts[cls])

    # ---------------------------------------------------------
    # Apply split labels
    # ---------------------------------------------------------
    df["split"] = df["base_id"].map(group_split)

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    df[["path", "label", "base_id", "split"]].to_csv(
        OUT,
        index=False
    )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------
    print()
    print("=" * 70)
    print("IMAGE COUNTS BY SPLIT")
    print("=" * 70)

    print(df["split"].value_counts())

    print()
    print("=" * 70)
    print("CLASS × SPLIT")
    print("=" * 70)

    print(pd.crosstab(df["label"], df["split"]))

    print()
    print("=" * 70)
    print("GROUP LEAKAGE CHECK")
    print("=" * 70)

    leakage = df.groupby("base_id")["split"].nunique()

    print(
        "Base IDs appearing in multiple splits:",
        int((leakage > 1).sum())
    )

    print()
    print("Saved:")
    print(OUT)

    print()
    if (leakage > 1).sum() == 0:
        print("STATUS: SUCCESS - NO GROUP LEAKAGE")
    else:
        print("STATUS: ERROR - GROUP LEAKAGE FOUND")


if __name__ == "__main__":
    main()