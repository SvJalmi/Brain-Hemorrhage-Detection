from pathlib import Path
import csv

from sklearn.model_selection import train_test_split

from preprocessing.common.config import TBI6000_DIR


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "preprocessing"
    / "classification"
)

OUTPUT_FILE = OUTPUT_DIR / "tbi6000_split.csv"


# ============================================================
# COLLECT DATASET
# ============================================================

def collect_images():
    records = []

    for class_dir in sorted(TBI6000_DIR.iterdir()):

        if not class_dir.is_dir():
            continue

        label = class_dir.name

        for image_path in sorted(class_dir.iterdir()):

            if image_path.suffix.lower() not in {
                ".png",
                ".jpg",
                ".jpeg",
            }:
                continue

            records.append({
                "path": str(image_path.relative_to(TBI6000_DIR)),
                "label": label,
            })

    return records


# ============================================================
# MAIN
# ============================================================

def main():

    records = collect_images()

    print("Total images:", len(records))

    if len(records) != 6000:
        raise ValueError(
            f"Expected 6000 images, found {len(records)}"
        )

    paths = [r["path"] for r in records]
    labels = [r["label"] for r in records]

    # --------------------------------------------------------
    # First split:
    # 70% train
    # 30% temporary
    # --------------------------------------------------------

    train_paths, temp_paths, train_labels, temp_labels = (
        train_test_split(
            paths,
            labels,
            test_size=(VALIDATION_RATIO + TEST_RATIO),
            stratify=labels,
            random_state=RANDOM_SEED,
        )
    )

    # --------------------------------------------------------
    # Second split:
    # 15% validation
    # 15% test
    # --------------------------------------------------------

    relative_test_ratio = TEST_RATIO / (
        VALIDATION_RATIO + TEST_RATIO
    )

    val_paths, test_paths, val_labels, test_labels = (
        train_test_split(
            temp_paths,
            temp_labels,
            test_size=relative_test_ratio,
            stratify=temp_labels,
            random_state=RANDOM_SEED,
        )
    )

    rows = []

    for path, label in zip(
        train_paths,
        train_labels,
    ):
        rows.append({
            "path": path,
            "label": label,
            "split": "train",
        })

    for path, label in zip(
        val_paths,
        val_labels,
    ):
        rows.append({
            "path": path,
            "label": label,
            "split": "validation",
        })

    for path, label in zip(
        test_paths,
        test_labels,
    ):
        rows.append({
            "path": path,
            "label": label,
            "split": "test",
        })

    rows.sort(key=lambda x: x["path"])

    # --------------------------------------------------------
    # Save split
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "label",
                "split",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()
    print("Split created successfully.")
    print("Output:", OUTPUT_FILE)
    print()

    for split_name in [
        "train",
        "validation",
        "test",
    ]:

        split_rows = [
            r for r in rows
            if r["split"] == split_name
        ]

        print(
            split_name,
            ":",
            len(split_rows),
        )

        label_counts = {}

        for r in split_rows:
            label_counts[r["label"]] = (
                label_counts.get(r["label"], 0) + 1
            )

        for label in sorted(label_counts):
            print(
                "   ",
                label,
                ":",
                label_counts[label],
            )


if __name__ == "__main__":
    main()