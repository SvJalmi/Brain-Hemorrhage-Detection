from pathlib import Path
from PIL import Image
import numpy as np

from preprocessing.common.config import (
    TBI6000_DIR,
    CLASSIFICATION_IMAGE_SIZE,
)

# ============================================================
# TEST PREPROCESSING ONLY
# ============================================================

# We deliberately process only ONE image first.
# Once this is verified, we will process the full dataset.

OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "preprocessing"
    / "classification"
    / "test_output"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_image(image_path: Path) -> None:
    print("Input:", image_path)

    # Load image as grayscale
    image = Image.open(image_path).convert("L")

    print("Original size:", image.size)
    print("Original mode:", image.mode)

    # Resize using high-quality interpolation
    image = image.resize(
        CLASSIFICATION_IMAGE_SIZE,
        Image.Resampling.BILINEAR,
    )

    # Convert to float32
    array = np.asarray(image, dtype=np.float32)

    # Normalize to [0, 1]
    array = array / 255.0

    print("Preprocessed shape:", array.shape)
    print("Data type:", array.dtype)
    print("Minimum:", float(array.min()))
    print("Maximum:", float(array.max()))

    # Save as NumPy array for this test
    output_path = OUTPUT_DIR / "sample.npy"
    np.save(output_path, array)

    print("Saved:", output_path)


def find_first_image() -> Path:
    extensions = {".png", ".jpg", ".jpeg"}

    images = [
        p
        for p in TBI6000_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    ]

    if not images:
        raise FileNotFoundError(
            f"No classification images found in: {TBI6000_DIR}"
        )

    images.sort()
    return images[0]


if __name__ == "__main__":
    print("=== TBI-6000 TEST PREPROCESSING ===")
    print("Dataset:", TBI6000_DIR)
    print()

    sample = find_first_image()

    print("Selected sample:", sample)
    print()

    preprocess_image(sample)

    print()
    print("=== TEST COMPLETE ===")