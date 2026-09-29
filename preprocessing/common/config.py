from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASETS_DIR = PROJECT_ROOT / "datasets"
PREPROCESSING_DIR = PROJECT_ROOT / "preprocessing"

TBI6000_DIR = DATASETS_DIR / "TBI-6000"
TBIHEMOCT_DIR = DATASETS_DIR / "TBI-HemoCT"
MBH_SEG25_DIR = DATASETS_DIR / "MBH-Seg25"


# ============================================================
# CLASSIFICATION SETTINGS
# ============================================================

CLASSIFICATION_IMAGE_SIZE = (224, 224)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ============================================================
# SEGMENTATION SETTINGS
# ============================================================

SEGMENTATION_TARGET_SPACING = (1.0, 1.0, 1.0)


# ============================================================
# DATA SPLIT
# ============================================================

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


if abs(TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO - 1.0) > 1e-6:
    raise ValueError(
        "Train/validation/test ratios must sum to 1.0"
    )