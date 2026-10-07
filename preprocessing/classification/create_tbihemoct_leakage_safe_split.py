import os
import re
import hashlib
import random
from collections import defaultdict

import pandas as pd


INPUT = r"preprocessing\classification\tbihemoct_labels.csv"
OUTPUT = r"preprocessing\classification\tbihemoct_final_leakage_safe.csv"

RANDOM_STATE = 42

TARGETS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}

LABEL_COLS = [
    "epidural",
    "subdural",
    "subarachnoid",
    "intraparenchymal",
    "intraventricular",
]


# ============================================================
# UNION-FIND
# ============================================================

class UnionFind:

    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):

        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return

        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


# ============================================================
# FILE HASH
# ============================================================

def md5_file(path):

    h = hashlib.md5()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


# ============================================================
# BASE ID
# ============================================================

def extract_base_id(path):

    match = re.search(r"ID_(\d+)", os.path.basename(path))

    if match:
        return "ID_" + match.group(1)

    return None


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("TBI-HemoCT FINAL LEAKAGE-SAFE SPLIT")
print("=" * 70)

df = pd.read_csv(INPUT)

print("Original rows:", len(df))


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required = [
    "image_path",
    "epidural",
    "subdural",
    "subarachnoid",
    "intraparenchymal",
    "intraventricular",
    "is_normal",
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise RuntimeError(f"Missing required columns: {missing}")


# ============================================================
# BASE IDs
# ============================================================

df["base_id"] = df["image_path"].apply(extract_base_id)

if df["base_id"].isna().any():
    raise RuntimeError("Some rows have invalid base IDs.")

print("Unique base IDs:", df["base_id"].nunique())


# ============================================================
# FILE HASHES
# ============================================================

print()
print("Calculating file hashes...")

hashes = []

for i, path in enumerate(df["image_path"]):

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    hashes.append(md5_file(path))

    if (i + 1) % 5000 == 0:
        print(f"  processed {i + 1}/{len(df)}")


df["file_hash"] = hashes

print("Unique file hashes:", df["file_hash"].nunique())


# ============================================================
# IDENTIFY CONFLICTING HASHES
# ============================================================

print()
print("Checking hash-label consistency...")

hash_label_variation = (
    df.groupby("file_hash")[LABEL_COLS]
    .nunique()
)

conflict_mask = (hash_label_variation > 1).any(axis=1)

conflicting_hashes = set(
    hash_label_variation.index[conflict_mask]
)

conflicting_rows = df[
    df["file_hash"].isin(conflicting_hashes)
].copy()

print("Conflicting hashes:", len(conflicting_hashes))
print("Rows affected:", len(conflicting_rows))
print(
    "Base IDs affected:",
    conflicting_rows["base_id"].nunique()
)


# ============================================================
# REMOVE CONFLICTING RECORDS
# ============================================================

if len(conflicting_rows) > 0:

    df = df[
        ~df["file_hash"].isin(conflicting_hashes)
    ].copy()

print()
print("Rows after conflict removal:", len(df))
print("Removed rows:", len(conflicting_rows))


# ============================================================
# UNION-FIND GROUPING
#
# Same base ID  -> same group
# Same file hash -> same group
# ============================================================

n = len(df)

uf = UnionFind(n)

base_to_indices = defaultdict(list)
hash_to_indices = defaultdict(list)

for i, row in df.reset_index(drop=True).iterrows():

    base_to_indices[row["base_id"]].append(i)
    hash_to_indices[row["file_hash"]].append(i)


print()
print("Connecting base-ID groups...")

for indices in base_to_indices.values():

    first = indices[0]

    for idx in indices[1:]:
        uf.union(first, idx)


print("Connecting identical-image groups...")

for indices in hash_to_indices.values():

    first = indices[0]

    for idx in indices[1:]:
        uf.union(first, idx)


# ============================================================
# COMPACT GROUP IDS
# ============================================================

root_to_group = {}
group_ids = []

for i in range(n):

    root = uf.find(i)

    if root not in root_to_group:
        root_to_group[root] = len(root_to_group)

    group_ids.append(root_to_group[root])


df = df.reset_index(drop=True)

df["group_id"] = group_ids

print()
print("Unique leakage groups:", df["group_id"].nunique())


# ============================================================
# GROUP STATISTICS
# ============================================================

groups = []

for group_id, g in df.groupby("group_id"):

    row = {
        "group_id": group_id,
        "n_rows": len(g),
        "n_base_ids": g["base_id"].nunique(),
    }

    for label in LABEL_COLS:
        row[label] = int(g[label].sum())

    groups.append(row)


groups_df = pd.DataFrame(groups)

print()
print("Largest leakage groups:")

print(
    groups_df
    .sort_values("n_rows", ascending=False)
    .head(15)
    .to_string(index=False)
)


# ============================================================
# GROUP ASSIGNMENT
#
# Large groups are assigned first.
# We select the split whose current proportion is furthest
# below its target.
# ============================================================

random.seed(RANDOM_STATE)

groups_list = groups_df.to_dict("records")

random.shuffle(groups_list)

total_rows = len(df)

target_rows = {
    split: total_rows * fraction
    for split, fraction in TARGETS.items()
}

split_rows = {
    "train": 0,
    "validation": 0,
    "test": 0,
}

group_split = {}

for group in sorted(
    groups_list,
    key=lambda x: x["n_rows"],
    reverse=True
):

    # Select split with largest deficit
    deficits = {
        split: target_rows[split] - split_rows[split]
        for split in TARGETS
    }

    best_split = max(
        deficits,
        key=deficits.get
    )

    group_split[
        group["group_id"]
    ] = best_split

    split_rows[best_split] += group["n_rows"]


df["split"] = df["group_id"].map(group_split)


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("=" * 70)
print("FINAL SPLIT")
print("=" * 70)

print()
print("Rows:")
print(df["split"].value_counts().sort_index())

print()
print("Percentages:")

print(
    (
        df["split"]
        .value_counts(normalize=True)
        .sort_index() * 100
    ).round(2)
)


print()
print("Unique base IDs:")

for split in ["train", "validation", "test"]:

    print(
        split,
        df.loc[
            df["split"] == split,
            "base_id"
        ].nunique()
    )


# ============================================================
# LEAKAGE CHECKS
# ============================================================

base_split_counts = (
    df.groupby("base_id")["split"]
    .nunique()
)

hash_split_counts = (
    df.groupby("file_hash")["split"]
    .nunique()
)

base_leakage = int(
    (base_split_counts > 1).sum()
)

hash_leakage = int(
    (hash_split_counts > 1).sum()
)

print()
print("BASE-ID GROUPS CROSSING SPLITS:", base_leakage)
print("FILE-HASH GROUPS CROSSING SPLITS:", hash_leakage)


# ============================================================
# CONFLICT CHECK
# ============================================================

remaining_variation = (
    df.groupby("file_hash")[LABEL_COLS]
    .nunique()
)

remaining_conflicts = int(
    (
        (remaining_variation > 1)
        .any(axis=1)
    ).sum()
)

print(
    "CONFLICTING HASHES REMAINING:",
    remaining_conflicts
)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print()
print("LABEL COUNTS:")

print(
    df.groupby("split")[LABEL_COLS]
    .sum()
)


print()
print("NORMAL COUNTS:")

print(
    df.groupby("split")["is_normal"]
    .sum()
)


# ============================================================
# FILE CHECK
# ============================================================

missing_files = int(
    (~df["image_path"].map(os.path.isfile))
    .sum()
)

print()
print("MISSING FILES:", missing_files)


# ============================================================
# FINAL VALIDATION
# ============================================================

if base_leakage != 0:
    raise RuntimeError(
        "Base-ID leakage detected!"
    )

if hash_leakage != 0:
    raise RuntimeError(
        "File-hash leakage detected!"
    )

if remaining_conflicts != 0:
    raise RuntimeError(
        "Conflicting labels remain!"
    )

if missing_files != 0:
    raise RuntimeError(
        "Missing image files detected!"
    )


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT,
    index=False
)

print()
print("Saved:")
print(OUTPUT)

print()
print("=" * 70)
print(
    "STATUS: SUCCESS - FINAL SPLIT IS LEAKAGE-SAFE"
)
print("=" * 70)