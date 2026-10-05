from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
LABELS = ["epidural", "subdural", "subarachnoid",
          "intraparenchymal", "intraventricular"]

DIR = Path("preprocessing/classification")
IN_CSV = DIR / "tbihemoct_manifest_hashed.csv"
OUT_CSV = DIR / "tbihemoct_labels.csv"

MIN_COMBO_COUNT = 20  # rarer label combinations are pooled for stratification


def parse_positive(value, class_name):
    """Return the set of positive labels for one manifest row."""
    if pd.isna(value) or str(value).strip() == "":
        if class_name == "normal":
            return frozenset()
        return frozenset([class_name])  # fallback: use the folder name
    tokens = {t.strip().lower() for t in str(value).split("-") if t.strip()}
    unknown = tokens - set(LABELS)
    if unknown:
        raise ValueError(f"Unknown label tokens: {unknown} in '{value}'")
    return frozenset(tokens)


def main():
    df = pd.read_csv(IN_CSV)
    print("Manifest rows:", len(df))
    print("Unique image files (md5):", df["md5"].nunique())

    fallback_rows = int(
        (df["All positive locations"].isna() & (df["class_name"] != "normal")).sum()
    )
    df["_labels"] = [
        parse_positive(v, c)
        for v, c in zip(df["All positive locations"], df["class_name"])
    ]

    rows = []
    conflicts = 0
    for md5, g in df.groupby("md5", sort=True):
        label_sets = set(g["_labels"])
        if len(label_sets) > 1:
            conflicts += 1
        union = frozenset().union(*label_sets)
        first = g.sort_values("image_path").iloc[0]
        row = {
            "md5": md5,
            "image_path": first["image_path"],
            "n_source_copies": len(g),
        }
        for name in LABELS:
            row[name] = int(name in union)
        rows.append(row)

    out = pd.DataFrame(rows)
    out["label_str"] = out[LABELS].apply(
        lambda r: "+".join(n for n in LABELS if r[n]) or "normal", axis=1
    )

    print("Rows using folder-name fallback label:", fallback_rows)
    print("Files whose copies disagree on labels:", conflicts)
    print("Unique images after collapsing:", len(out))

    # ---- stratified 70/15/15 split on label combination ----
    counts = out["label_str"].value_counts()
    key = out["label_str"].where(out["label_str"].map(counts) >= MIN_COMBO_COUNT, "rare")
    if (key == "rare").sum() < 10:
        key[key == "rare"] = counts.index[0]

    idx = out.index.to_numpy()
    train_idx, temp_idx = train_test_split(
        idx, test_size=0.30, stratify=key, random_state=SEED
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=key.loc[temp_idx], random_state=SEED
    )

    out["split"] = ""
    out.loc[train_idx, "split"] = "train"
    out.loc[val_idx, "split"] = "validation"
    out.loc[test_idx, "split"] = "test"

    assert out["md5"].is_unique
    assert (out["split"] != "").all()

    print("\nImages per split:")
    print(out["split"].value_counts().to_string())

    print("\nPositives per class per split:")
    print(out.groupby("split")[LABELS].sum().to_string())

    out["is_normal"] = (out[LABELS].sum(axis=1) == 0).astype(int)
    print("\nNormal images per split:")
    print(out.groupby("split")["is_normal"].sum().to_string())

    out.to_csv(OUT_CSV, index=False)
    print("\nSaved:", OUT_CSV)


if __name__ == "__main__":
    main()