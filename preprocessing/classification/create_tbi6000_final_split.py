import os
import re
import hashlib
import random
import pandas as pd

ROOT = r"datasets\TBI-6000"
OUT = r"preprocessing\classification\tbi6000_final_split.csv"

SEED = 42

CLASSES = [
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "normal",
    "subarachnoid",
    "subdural",
]


def base_id(filename):
    m = re.match(r"(ID_\d+)", filename)
    if not m:
        raise ValueError(f"Cannot extract ID: {filename}")
    return m.group(1)


def file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(parent, a, b):
    ra = find(parent, a)
    rb = find(parent, b)

    if ra != rb:
        parent[rb] = ra


def main():
    random.seed(SEED)

    rows = []

    print("=" * 70)
    print("TBI-6000 FINAL LEAKAGE-SAFE SPLIT")
    print("=" * 70)

    # ---------------------------------------------------------
    # Collect images
    # ---------------------------------------------------------
    for cls in CLASSES:
        folder = os.path.join(ROOT, cls)

        for filename in os.listdir(folder):
            if not filename.lower().endswith(".png"):
                continue

            path = os.path.join(folder, filename)

            rows.append({
                "path": os.path.join(cls, filename),
                "label": cls,
                "base_id": base_id(filename),
                "file_hash": file_hash(path),
            })

    df = pd.DataFrame(rows)

    print("Total images:", len(df))
    print("Unique base IDs:", df["base_id"].nunique())
    print("Unique pixel hashes:", df["file_hash"].nunique())

    # ---------------------------------------------------------
    # Build connected components
    #
    # Same base_id OR same pixel hash = same leakage group
    # ---------------------------------------------------------
    parent = {i: i for i in range(len(df))}

    base_owner = {}
    hash_owner = {}

    for i, row in df.iterrows():

        b = row["base_id"]
        h = row["file_hash"]

        if b in base_owner:
            union(parent, i, base_owner[b])
        else:
            base_owner[b] = i

        if h in hash_owner:
            union(parent, i, hash_owner[h])
        else:
            hash_owner[h] = i

    df["group_id"] = [
        find(parent, i)
        for i in range(len(df))
    ]

    print("Leakage groups:", df["group_id"].nunique())

    # ---------------------------------------------------------
    # Group-level class counts
    # ---------------------------------------------------------
    group_counts = (
        df.groupby(["group_id", "label"])
        .size()
        .unstack(fill_value=0)
    )

    for cls in CLASSES:
        if cls not in group_counts.columns:
            group_counts[cls] = 0

    group_counts = group_counts[CLASSES]

    groups = list(group_counts.index)
    random.shuffle(groups)

    total = df["label"].value_counts().to_dict()

    targets = {
        "train": {c: total[c] * 0.70 for c in CLASSES},
        "validation": {c: total[c] * 0.15 for c in CLASSES},
        "test": {c: total[c] * 0.15 for c in CLASSES},
    }

    current = {
        s: {c: 0 for c in CLASSES}
        for s in ["train", "validation", "test"]
    }

    assignments = {}

    # ---------------------------------------------------------
    # Greedy group assignment
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

                if target > 0:
                    score += ((after - target) / target) ** 2

            if score < best_score:
                best_score = score
                best_split = split

        assignments[group] = best_split

        for cls in CLASSES:
            current[best_split][cls] += int(counts[cls])

    df["split"] = df["group_id"].map(assignments)

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    df[[
        "path",
        "label",
        "base_id",
        "file_hash",
        "group_id",
        "split"
    ]].to_csv(OUT, index=False)

    # ---------------------------------------------------------
    # Reports
    # ---------------------------------------------------------
    print()
    print("=" * 70)
    print("IMAGE COUNTS")
    print("=" * 70)

    print(df["split"].value_counts())

    print()
    print("=" * 70)
    print("CLASS × SPLIT")
    print("=" * 70)

    print(pd.crosstab(df["label"], df["split"]))

    # ---------------------------------------------------------
    # Base ID leakage
    # ---------------------------------------------------------
    base_leak = (
        df.groupby("base_id")["split"]
        .nunique()
    )

    # ---------------------------------------------------------
    # Pixel duplicate leakage
    # ---------------------------------------------------------
    hash_leak = (
        df.groupby("file_hash")["split"]
        .nunique()
    )

    print()
    print("=" * 70)
    print("FINAL LEAKAGE CHECK")
    print("=" * 70)

    print(
        "Base IDs crossing splits:",
        int((base_leak > 1).sum())
    )

    print(
        "Pixel-duplicate groups crossing splits:",
        int((hash_leak > 1).sum())
    )

    print()
    print("Saved:")
    print(OUT)

    if (base_leak > 1).sum() == 0 and (hash_leak > 1).sum() == 0:
        print()
        print("STATUS: SUCCESS - FINAL SPLIT IS LEAKAGE-SAFE")
    else:
        print()
        print("STATUS: ERROR - LEAKAGE STILL EXISTS")


if __name__ == "__main__":
    main()