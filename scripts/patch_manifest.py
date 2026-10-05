from pathlib import Path

target = Path("preprocessing/classification/create_tbihemoct_manifest.py")
src = target.read_text(encoding="utf-8")

# ---- Edit 1: add constants after OUTPUT_CSV ----
old1 = 'OUTPUT_CSV = OUTPUT_DIR / "tbihemoct_manifest.csv"\n'
new1 = (
    old1
    + '\nUNRESOLVED_CSV = OUTPUT_DIR / "tbihemoct_unresolved_base_images.csv"\n'
    + 'MAX_UNRESOLVED_FRACTION = 0.001  # abort if more than 0.1% unresolved\n'
)
assert src.count(old1) == 1, "Edit 1 anchor not found exactly once"
src = src.replace(old1, new1)

# ---- Edit 2: replace the hard failure in attach_image_paths ----
old2 = '''    if missing or ambiguous:

        raise FileNotFoundError(
            "Some base images could not be resolved "
            "uniquely on disk."
        )

    return manifest
'''
new2 = '''    if missing or ambiguous:

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
'''
assert src.count(old2) == 1, "Edit 2 anchor not found exactly once"
src = src.replace(old2, new2)

target.write_text(src, encoding="utf-8")
print("Patch applied successfully.")