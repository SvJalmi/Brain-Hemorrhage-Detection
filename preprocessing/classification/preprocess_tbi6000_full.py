from pathlib import Path
import csv

import numpy as np
from PIL import Image

from preprocessing.common.config import (
    TBI6000_DIR,
    CLASSIFICATION_IMAGE_SIZE,
)


# ============================================================
# TBI-6000 FULL PREPROCESSING
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_FILE = (
    PROJECT_ROOT
    / "preprocessing"
    / "classification"
    / "tbi6000_split.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "preprocessed_data"
    / "TBI-6000"
)

RANDOM_SEED = 42


# ============================================================
# PREPROCESS ONE IMAGE
# ============================================================

def preprocess_image(image_path: Path) -> np.ndarray:

    image = Image.open(image_path).convert("L")

    image = image.resize(
        CLASSIFICATION_IMAGE_SIZE,
        Image.Resampling.BILINEAR,
    )

    array = np.asarray(
        image,
        dtype=np.float32,
    )

    # Normalize 8-bit CT representation to [0, 1]
    array /= 255.0

    return array


# ============================================================
# PROCESS DATASET
# ============================================================

def main():

    print("=" * 60)
    print("TBI-6000 FULL PREPROCESSING")
    print("=" * 60)

    print("Dataset:")
    print(TBI6000_DIR)

    print()
    print("Split file:")
    print(SPLIT_FILE)

    print()
    print("Output:")
    print(OUTPUT_DIR)

    # --------------------------------------------------------
    # Verify split file
    # --------------------------------------------------------

    if not SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"Split file not found: {SPLIT_FILE}"
        )

    # --------------------------------------------------------
    # Read split CSV
    # --------------------------------------------------------

    with open(
        SPLIT_FILE,
        "r",
        newline="",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(f)
        records = list(reader)

    print()
    print("Records in split file:", len(records))

    if len(records) != 6000:
        raise ValueError(
            f"Expected 6000 records, found {len(records)}"
        )

    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    for split in [
        "train",
        "validation",
        "test",
    ]:
        (OUTPUT_DIR / split).mkdir(
            parents=True,
            exist_ok=True,
        )

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    processed = 0
    failed = []

    metadata_rows = []

    for index, record in enumerate(records, start=1):

        relative_path = Path(record["path"])
        label = record["label"]
        split = record["split"]

        input_path = TBI6000_DIR / relative_path

        try:

            if not input_path.exists():
                raise FileNotFoundError(
                    f"Input image does not exist: {input_path}"
                )

            array = preprocess_image(input_path)

            # ------------------------------------------------
            # Preserve original filename
            # ------------------------------------------------

            output_name = (
                relative_path.stem + ".npy"
            )

            output_path = (
                OUTPUT_DIR
                / split
                / label
                / output_name
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            np.save(
                output_path,
                array,
            )

            metadata_rows.append({
                "processed_path": str(
                    output_path.relative_to(PROJECT_ROOT)
                ),
                "original_path": str(
                    relative_path
                ),
                "label": label,
                "split": split,
                "height": array.shape[0],
                "width": array.shape[1],
                "dtype": str(array.dtype),
                "min": float(array.min()),
                "max": float(array.max()),
            })

            processed += 1

            # Progress every 100 images
            if processed % 100 == 0:
                print(
                    f"Processed {processed}/6000"
                )

        except Exception as exc:

            failed.append({
                "path": str(relative_path),
                "error": str(exc),
            })

            print(
                f"FAILED: {relative_path}"
            )
            print(
                f"Reason: {exc}"
            )

    # --------------------------------------------------------
    # Save preprocessing metadata
    # --------------------------------------------------------

    metadata_file = (
        OUTPUT_DIR
        / "preprocessing_metadata.csv"
    )

    with open(
        metadata_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        fieldnames = [
            "processed_path",
            "original_path",
            "label",
            "split",
            "height",
            "width",
            "dtype",
            "min",
            "max",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(metadata_rows)

    # --------------------------------------------------------
    # Save failure report
    # --------------------------------------------------------

    failure_file = (
        OUTPUT_DIR
        / "preprocessing_failures.csv"
    )

    with open(
        failure_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "error",
            ],
        )

        writer.writeheader()
        writer.writerows(failed)

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)

    print("Expected:", 6000)
    print("Processed:", processed)
    print("Failed:", len(failed))

    print()
    print("Metadata:")
    print(metadata_file)

    print()
    print("Failure report:")
    print(failure_file)

    if processed == 6000 and not failed:
        print()
        print("STATUS: SUCCESS")
    else:
        print()
        print("STATUS: CHECK FAILURES")


if __name__ == "__main__":
    main()