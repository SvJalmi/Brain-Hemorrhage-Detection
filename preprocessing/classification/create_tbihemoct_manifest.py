from pathlib import Path
import re
import pandas as pd


# ============================================================
# TBI-HemoCT MANIFEST CREATION
# ============================================================

SEED = 42

# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TBIHEMOCT_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "TBI-HemoCT"
    / "TBI-HemoCT Data set"
    / "TBI-HemoCT Data set"
)

RAW_CSV = TBIHEMOCT_DIR / "final_log_file.csv"

SPLIT_CSV = (
    PROJECT_ROOT
    / "preprocessing"
    / "classification"
    / "tbihemoct_base_split.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "preprocessing"
    / "classification"
)

OUTPUT_CSV = OUTPUT_DIR / "tbihemoct_manifest.csv"

UNRESOLVED_CSV = OUTPUT_DIR / "tbihemoct_unresolved_base_images.csv"
MAX_UNRESOLVED_FRACTION = 0.001  # abort if more than 0.1% unresolved


# ------------------------------------------------------------
# Expected dataset folders
# ------------------------------------------------------------

EXPECTED_FOLDERS = [
    "Any(normal)",
    "Epidural",
    "Intraparenchymal",
    "Intraventricular",
    "Subarachnoid",
    "Subdural",
]


# ------------------------------------------------------------
# Class mapping
# ------------------------------------------------------------

CLASS_MAP = {
    "Any(normal)": "normal",
    "Epidural": "epidural",
    "Intraparenchymal": "intraparenchymal",
    "Intraventricular": "intraventricular",
    "Subarachnoid": "subarachnoid",
    "Subdural": "subdural",
}


LABELS = [
    "epidural",
    "subdural",
    "subarachnoid",
    "intraparenchymal",
    "intraventricular",
]


# ============================================================
# Utility functions
# ============================================================

def extract_image_id(filename):
    """
    Extract numeric image ID.

    Examples:

        ID_123_normal_UNIQUE.png
        ID_123_subdural_UNIQUE.png
        ID_123_subdural_aug_7.png

    -> 123
    """

    match = re.search(
        r"ID_(\d+)",
        str(filename),
    )

    if match:
        return int(match.group(1))

    return None


def remove_augmentation_suffix(filename):
    """
    Convert an augmented filename into its base filename.

    Example:

        ID_123_subdural_aug_7.png

    becomes:

        ID_123_subdural.png
    """

    return re.sub(
        r"_aug_\d+(?=\.png$)",
        "",
        str(filename),
        flags=re.IGNORECASE,
    )


def normalize_filename(filename):
    """
    Normalize filename for reliable comparison.
    """

    return Path(
        str(filename)
    ).name.strip().lower()


# ============================================================
# Dataset discovery
# ============================================================

def validate_dataset_root():
    """
    Validate TBI-HemoCT dataset directory.
    """

    print()
    print("Checking dataset root...")

    if not TBIHEMOCT_DIR.exists():
        raise FileNotFoundError(
            f"TBI-HemoCT dataset directory not found:\n"
            f"{TBIHEMOCT_DIR}"
        )

    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"Raw CSV not found:\n"
            f"{RAW_CSV}"
        )

    print("Dataset root exists:", TBIHEMOCT_DIR)
    print("Raw CSV exists:", RAW_CSV)


def print_dataset_folders():
    """
    Print top-level folders.
    """

    print()
    print("Top-level dataset contents:")

    for item in sorted(
        TBIHEMOCT_DIR.iterdir(),
        key=lambda x: x.name.lower(),
    ):
        if item.is_dir():
            print("  [DIR ]", item.name)
        else:
            print("  [FILE]", item.name)


# ============================================================
# Build actual file index
# ============================================================

def build_file_index():
    """
    Recursively scan the complete TBI-HemoCT dataset.

    Creates mappings:

        normalized filename -> actual disk path

    and

        normalized base filename -> list of actual paths

    This is necessary because images are stored inside
    class-specific folders rather than directly in the dataset root.
    """

    print()
    print("=" * 70)
    print("BUILDING IMAGE FILE INDEX")
    print("=" * 70)

    print()
    print("Searching recursively for PNG files...")

    png_files = []

    for path in TBIHEMOCT_DIR.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() != ".png":
            continue

        # Do not treat the CSV or anything else as an image.
        png_files.append(path)

    print()
    print("PNG files found:", len(png_files))

    if len(png_files) == 0:
        raise RuntimeError(
            "No PNG files were found inside the TBI-HemoCT dataset."
        )

    # --------------------------------------------------------
    # Exact filename index
    # --------------------------------------------------------

    filename_index = {}

    # --------------------------------------------------------
    # Base filename index
    # --------------------------------------------------------

    base_index = {}

    duplicate_exact = []

    duplicate_base = []

    for path in png_files:

        filename = normalize_filename(
            path.name
        )

        base_filename = normalize_filename(
            remove_augmentation_suffix(
                path.name
            )
        )

        # ----------------------------------------------------
        # Exact filename
        # ----------------------------------------------------

        if filename in filename_index:

            duplicate_exact.append(
                (
                    filename,
                    filename_index[filename],
                    path,
                )
            )

        else:

            filename_index[filename] = path

        # ----------------------------------------------------
        # Base filename
        # ----------------------------------------------------

        base_index.setdefault(
            base_filename,
            [],
        ).append(path)

    # --------------------------------------------------------
    # Report duplicates
    # --------------------------------------------------------

    print()
    print("Unique exact filenames:", len(filename_index))
    print("Unique base filenames:", len(base_index))

    print()
    print(
        "Duplicate exact filenames:",
        len(duplicate_exact),
    )

    duplicate_base_count = sum(
        1
        for paths in base_index.values()
        if len(paths) > 1
    )

    print(
        "Base filenames with multiple disk files:",
        duplicate_base_count,
    )

    # --------------------------------------------------------
    # Show a few base duplicates
    # --------------------------------------------------------

    if duplicate_base_count > 0:

        print()
        print("Examples of base filenames with multiple files:")

        shown = 0

        for base_name, paths in base_index.items():

            if len(paths) <= 1:
                continue

            print()
            print(" ", base_name)

            for path in paths[:10]:
                print("    ", path)

            shown += 1

            if shown >= 10:
                break

    return filename_index, base_index


# ============================================================
# Load split CSV
# ============================================================

def load_split_csv():
    """
    Load tbihemoct_base_split.csv.

    The split CSV contains one row per base image.

    Required columns:

        image_id
        base_file
        Destination
        Label
        class_name
        signature
        split
    """

    print()
    print("=" * 70)
    print("LOADING SPLIT CSV")
    print("=" * 70)

    if not SPLIT_CSV.exists():

        raise FileNotFoundError(
            f"Split CSV not found:\n"
            f"{SPLIT_CSV}"
        )

    df = pd.read_csv(
        SPLIT_CSV
    )

    required_columns = {
        "image_id",
        "base_file",
        "Destination",
        "Label",
        "class_name",
        "signature",
        "split",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Split CSV is missing required columns:\n"
            + "\n".join(
                sorted(missing)
            )
        )

    print()
    print("Split CSV columns:")

    for column in df.columns:
        print(" ", column)

    print()
    print("Split CSV rows:", len(df))

    print(
        "Split CSV unique base files:",
        df["base_file"].nunique(),
    )

    print(
        "Split CSV unique image IDs:",
        df["image_id"].nunique(),
    )

    # --------------------------------------------------------
    # Validate duplicate base files
    # --------------------------------------------------------

    duplicate_base = (
        df["base_file"]
        .duplicated(keep=False)
    )

    if duplicate_base.any():

        duplicates = (
            df.loc[
                duplicate_base,
                "base_file",
            ]
            .drop_duplicates()
            .tolist()
        )

        print()
        print(
            "WARNING: duplicate base filenames in split CSV:"
        )

        for item in duplicates[:20]:
            print(" ", item)

        raise RuntimeError(
            "Split CSV contains duplicate base filenames."
        )

    # --------------------------------------------------------
    # Validate image IDs
    #
    # Multiple base images per ID are EXPECTED.
    # Therefore we do NOT require image_id to be unique.
    # --------------------------------------------------------

    split_per_id = (
        df.groupby("image_id")["split"]
        .nunique()
    )

    leaked_ids = split_per_id[
        split_per_id > 1
    ]

    if len(leaked_ids) > 0:

        print()
        print(
            "DATA LEAKAGE DETECTED:"
        )

        print(
            leaked_ids.head(20)
        )

        raise RuntimeError(
            "An image ID appears in multiple splits."
        )

    print()
    print(
        "Image IDs appearing in multiple splits:",
        int(
            (
                df.groupby("image_id")["split"]
                .nunique() > 1
            ).sum()
        ),
    )

    return df


# ============================================================
# Load raw metadata
# ============================================================

def load_raw_csv():
    """
    Load original TBI-HemoCT metadata.
    """

    print()
    print("=" * 70)
    print("LOADING RAW METADATA")
    print("=" * 70)

    df = pd.read_csv(
        RAW_CSV
    )

    print()
    print("Raw CSV rows:", len(df))
    print(
        "Raw unique filenames:",
        df["File Name"].nunique(),
    )

    # --------------------------------------------------------
    # Extract image IDs
    # --------------------------------------------------------

    df["image_id"] = (
        df["File Name"]
        .apply(extract_image_id)
    )

    missing_ids = (
        df["image_id"]
        .isna()
        .sum()
    )

    if missing_ids > 0:

        raise ValueError(
            f"{missing_ids} rows have no valid image ID."
        )

    df["image_id"] = (
        df["image_id"]
        .astype(int)
    )

    # --------------------------------------------------------
    # Create base filename
    # --------------------------------------------------------

    df["base_file"] = (
        df["File Name"]
        .apply(
            remove_augmentation_suffix
        )
    )

    print()
    print(
        "Unique base images:",
        df["base_file"].nunique(),
    )

    print(
        "Unique image IDs:",
        df["image_id"].nunique(),
    )

    return df


# ============================================================
# Attach actual image paths
# ============================================================

def resolve_base_image_path(
    base_filename,
    base_index,
):
    """
    Resolve a base filename to the actual PNG file.

    Preference:

        1. exact base filename
        2. unique matching base filename

    Augmented files are deliberately NOT selected.
    """

    normalized = normalize_filename(
        base_filename
    )

    candidates = (
        base_index.get(
            normalized,
            [],
        )
    )

    if len(candidates) == 0:

        return None

    # --------------------------------------------------------
    # Prefer an exact base image.
    #
    # Example:
    #
    # ID_123_subdural.png
    #
    # instead of:
    #
    # ID_123_subdural_aug_7.png
    # --------------------------------------------------------

    exact = [
        path
        for path in candidates
        if normalize_filename(path.name)
        == normalized
    ]

    if len(exact) == 1:

        return exact[0]

    if len(exact) > 1:

        # If identical filenames somehow exist in different
        # folders, this is ambiguous.
        return None

    # --------------------------------------------------------
    # No exact file.
    #
    # The base index may contain augmented variants.
    # We deliberately avoid silently choosing one.
    # --------------------------------------------------------

    non_augmented = [
        path
        for path in candidates
        if "_aug_" not in path.name.lower()
    ]

    if len(non_augmented) == 1:

        return non_augmented[0]

    return None


def attach_image_paths(
    split_df,
    base_index,
):
    """
    Attach actual image paths to the split manifest.
    """

    print()
    print("=" * 70)
    print("RESOLVING IMAGE PATHS")
    print("=" * 70)

    manifest = split_df.copy()

    resolved_paths = []

    missing = []

    ambiguous = []

    for _, row in manifest.iterrows():

        base_file = row["base_file"]

        path = resolve_base_image_path(
            base_file,
            base_index,
        )

        if path is None:

            candidates = base_index.get(
                normalize_filename(
                    base_file
                ),
                [],
            )

            if len(candidates) == 0:

                missing.append(
                    (
                        int(row["image_id"]),
                        base_file,
                    )
                )

            else:

                ambiguous.append(
                    (
                        int(row["image_id"]),
                        base_file,
                        candidates,
                    )
                )

            resolved_paths.append(None)

        else:

            resolved_paths.append(
                str(
                    path.resolve()
                )
            )

    manifest["image_path"] = (
        resolved_paths
    )

    # --------------------------------------------------------
    # Missing files
    # --------------------------------------------------------

    print()
    print(
        "Resolved image paths:",
        manifest["image_path"]
        .notna()
        .sum(),
    )

    print(
        "Missing base images:",
        len(missing),
    )

    print(
        "Ambiguous base images:",
        len(ambiguous),
    )

    # --------------------------------------------------------
    # Show missing examples
    # --------------------------------------------------------

    if len(missing) > 0:

        print()
        print("Examples of missing files:")

        for image_id, base_file in missing[:20]:

            print(
                f"  ID {image_id}: {base_file}"
            )

    # --------------------------------------------------------
    # Show ambiguous examples
    # --------------------------------------------------------

    if len(ambiguous) > 0:

        print()
        print("Examples of ambiguous files:")

        for (
            image_id,
            base_file,
            candidates,
        ) in ambiguous[:10]:

            print(
                f"  ID {image_id}: {base_file}"
            )

            for path in candidates:
                print(
                    "      ",
                    path,
                )

    if missing or ambiguous:

        unresolved = (
            [(i, f, "missing") for i, f in missing]
            + [(i, f, "only_augmented_or_ambiguous") for i, f, _ in ambiguous]
        )

        pd.DataFrame(
            unresolved,
            columns=["image_id", "base_file", "reason"],
        ).to_csv(UNRESOLVED_CSV, index=False)

        fraction = len(unresolved) / len(manifest)

        print()
        print(
            f"Unresolved base images: {len(unresolved)} "
            f"({fraction:.4%}) -> saved to {UNRESOLVED_CSV}"
        )

        if fraction > MAX_UNRESOLVED_FRACTION:
            raise FileNotFoundError(
                "Too many base images could not be resolved. "
                "Check the dataset path / extraction."
            )

        manifest = (
            manifest[manifest["image_path"].notna()]
            .reset_index(drop=True)
        )

        print("Rows dropped from manifest:", len(unresolved))

    return manifest


# ============================================================
# Attach raw metadata
# ============================================================

def attach_raw_metadata(
    manifest,
    raw_df,
):
    """
    Attach useful raw CSV metadata.

    Since the raw CSV can contain multiple rows per image ID,
    metadata is attached at base-file level.
    """

    print()
    print("=" * 70)
    print("ATTACHING RAW METADATA")
    print("=" * 70)

    metadata_columns = [
        "File Name",
        "Destination",
        "Label",
        "Placed in multiple locations",
        "All positive locations",
    ]

    available = [
        column
        for column in metadata_columns
        if column in raw_df.columns
    ]

    raw_meta = (
        raw_df[
            [
                "base_file"
            ]
            + available
        ]
        .drop_duplicates(
            subset=["base_file"]
        )
    )

    # --------------------------------------------------------
    # Make sure each base file has exactly one metadata row.
    # --------------------------------------------------------

    if (
        raw_meta["base_file"]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Raw metadata contains duplicate base-file records."
        )

    manifest = manifest.merge(
        raw_meta,
        on="base_file",
        how="left",
        validate="one_to_one",
        suffixes=(
            "",
            "_raw",
        ),
    )

    print(
        "Metadata rows attached:",
        len(raw_meta),
    )

    return manifest


# ============================================================
# Validate manifest
# ============================================================

def validate_manifest(
    manifest,
):
    """
    Perform final manifest validation.
    """

    print()
    print("=" * 70)
    print("VALIDATING MANIFEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required = [
        "image_id",
        "base_file",
        "class_name",
        "split",
        "image_path",
    ]

    missing_columns = [
        column
        for column in required
        if column not in manifest.columns
    ]

    if missing_columns:

        raise RuntimeError(
            "Manifest missing columns:\n"
            + "\n".join(
                missing_columns
            )
        )

    # --------------------------------------------------------
    # Empty paths
    # --------------------------------------------------------

    empty_paths = (
        manifest["image_path"]
        .isna()
        .sum()
    )

    if empty_paths > 0:

        raise RuntimeError(
            f"Manifest contains {empty_paths} empty image paths."
        )

    # --------------------------------------------------------
    # Check every file exists
    # --------------------------------------------------------

    nonexistent = []

    for path in manifest["image_path"]:

        if not Path(path).is_file():

            nonexistent.append(
                path
            )

    print()
    print(
        "Nonexistent image paths:",
        len(nonexistent),
    )

    if nonexistent:

        print()
        print("Examples:")

        for path in nonexistent[:20]:
            print(" ", path)

        raise RuntimeError(
            "Manifest contains nonexistent image paths."
        )

    # --------------------------------------------------------
    # Duplicate base files
    # --------------------------------------------------------

    duplicate_base = (
        manifest["base_file"]
        .duplicated()
        .sum()
    )

    print(
        "Duplicate base files:",
        duplicate_base,
    )

    if duplicate_base > 0:

        raise RuntimeError(
            "Manifest contains duplicate base files."
        )

    # --------------------------------------------------------
    # Image ID split leakage
    # --------------------------------------------------------

    id_split_counts = (
        manifest
        .groupby("image_id")["split"]
        .nunique()
    )

    leaked_ids = (
        id_split_counts[
            id_split_counts > 1
        ]
    )

    print(
        "IDs appearing in multiple splits:",
        len(leaked_ids),
    )

    if len(leaked_ids) > 0:

        raise RuntimeError(
            "Image-ID leakage detected."
        )

    # --------------------------------------------------------
    # Check split values
    # --------------------------------------------------------

    valid_splits = {
        "train",
        "validation",
        "test",
    }

    actual_splits = set(
        manifest["split"]
        .dropna()
        .unique()
    )

    invalid_splits = (
        actual_splits
        - valid_splits
    )

    if invalid_splits:

        raise ValueError(
            "Invalid split values:\n"
            + str(
                sorted(
                    invalid_splits
                )
            )
        )

    # --------------------------------------------------------
    # Check class values
    # --------------------------------------------------------

    valid_classes = {
        "normal",
        "epidural",
        "subdural",
        "subarachnoid",
        "intraparenchymal",
        "intraventricular",
    }

    actual_classes = set(
        manifest["class_name"]
        .dropna()
        .unique()
    )

    invalid_classes = (
        actual_classes
        - valid_classes
    )

    if invalid_classes:

        raise ValueError(
            "Invalid class values:\n"
            + str(
                sorted(
                    invalid_classes
                )
            )
        )

    print()
    print("Manifest validation: PASSED")


# ============================================================
# Print final summary
# ============================================================

def print_summary(
    manifest,
):
    """
    Print final manifest statistics.
    """

    print()
    print("=" * 70)
    print("TBI-HemoCT MANIFEST SUMMARY")
    print("=" * 70)

    print()
    print(
        "Total manifest rows:",
        len(manifest),
    )

    print(
        "Unique base images:",
        manifest["base_file"].nunique(),
    )

    print(
        "Unique image IDs:",
        manifest["image_id"].nunique(),
    )

    print()
    print("Images by split:")

    print(
        manifest["split"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Images by class:")

    print(
        manifest["class_name"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Class distribution by split:")

    table = pd.crosstab(
        manifest["split"],
        manifest["class_name"],
    )

    print(
        table.to_string()
    )

    print()
    print("Image IDs by split:")

    id_table = (
        manifest
        .groupby("split")["image_id"]
        .nunique()
        .sort_index()
    )

    print(
        id_table.to_string()
    )

    print()
    print(
        "Augmented files included in manifest:"
    )

    augmented_count = (
        manifest["image_path"]
        .astype(str)
        .str.contains(
            r"_aug_\d+\.png$",
            case=False,
            regex=True,
        )
        .sum()
    )

    print(
        augmented_count
    )

    print()
    print("=" * 70)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("TBI-HemoCT MANIFEST CREATION")
    print("=" * 70)

    print()
    print("Dataset:")
    print(TBIHEMOCT_DIR)

    print()
    print("Raw CSV:")
    print(RAW_CSV)

    print()
    print("Split CSV:")
    print(SPLIT_CSV)

    # --------------------------------------------------------
    # Validate directories
    # --------------------------------------------------------

    validate_dataset_root()

    print_dataset_folders()

    # --------------------------------------------------------
    # Load raw CSV
    # --------------------------------------------------------

    raw_df = load_raw_csv()

    # --------------------------------------------------------
    # Load leakage-safe split
    # --------------------------------------------------------

    split_df = load_split_csv()

    # --------------------------------------------------------
    # Build recursive file index
    # --------------------------------------------------------

    filename_index, base_index = (
        build_file_index()
    )

    # filename_index is retained for debugging/future use.
    _ = filename_index

    # --------------------------------------------------------
    # Resolve base image paths
    # --------------------------------------------------------

    manifest = attach_image_paths(
        split_df,
        base_index,
    )

    # --------------------------------------------------------
    # Attach original metadata
    # --------------------------------------------------------

    manifest = attach_raw_metadata(
        manifest,
        raw_df,
    )

    # --------------------------------------------------------
    # Sort deterministically
    # --------------------------------------------------------

    manifest = manifest.sort_values(
        [
            "split",
            "image_id",
            "class_name",
            "base_file",
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_manifest(
        manifest
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_summary(
        manifest
    )

    print()
    print("Saved:")
    print(OUTPUT_CSV)

    print()
    print("STATUS: SUCCESS")


if __name__ == "__main__":
    main()