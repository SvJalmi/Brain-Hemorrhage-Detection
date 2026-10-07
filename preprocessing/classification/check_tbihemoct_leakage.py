import pandas as pd
import os
import hashlib
from pathlib import Path

CSV = r"preprocessing\classification\tbihemoct_labels.csv"

df = pd.read_csv(CSV)

print("=" * 70)
print("TBI-HemoCT LEAKAGE CHECK")
print("=" * 70)

print("Total rows:", len(df))
print()

required = [
    "md5",
    "image_path",
    "split",
    "label_str",
    "is_normal",
]

print("Required columns:")
for col in required:
    print(f"  {col}: {'YES' if col in df.columns else 'NO'}")

missing = [c for c in required if c not in df.columns]

if missing:
    raise RuntimeError(f"Missing columns: {missing}")

print()
print("SPLIT COUNTS")
print("=" * 70)
print(df["split"].value_counts().sort_index())

print()
print("LABEL COUNTS")
print("=" * 70)
print(df["label_str"].value_counts().head(20))

# ------------------------------------------------------------
# 1. MD5 leakage
# ------------------------------------------------------------

print()
print("MD5 / EXACT FILE DUPLICATE LEAKAGE")
print("=" * 70)

g = (
    df.groupby("md5")
      .agg(
          n=("md5", "size"),
          splits=("split", lambda x: ",".join(sorted(set(x)))),
          labels=("label_str", lambda x: ",".join(sorted(set(x))))
      )
)

duplicates = g[g["n"] > 1]

cross_split = duplicates[
    duplicates["splits"].str.contains(",", regex=False)
]

print("Duplicate MD5 groups:", len(duplicates))
print("Rows in duplicate groups:", int(duplicates["n"].sum()))
print("Duplicate groups crossing splits:", len(cross_split))

if len(cross_split) == 0:
    print("STATUS: PASS - no exact duplicate crosses train/validation/test")
else:
    print("STATUS: FAIL - exact duplicates cross splits")
    print()
    print(cross_split.to_string())

# ------------------------------------------------------------
# 2. Filename/base-ID leakage
# ------------------------------------------------------------

print()
print("BASE-ID / SOURCE-ID LEAKAGE")
print("=" * 70)

# Extract ID_12345 from filenames
df["base_id"] = (
    df["image_path"]
    .str.extract(r"(ID_\d+)", expand=False)
)

missing_ids = df["base_id"].isna().sum()

print("Rows without detected base ID:", missing_ids)

if missing_ids > 0:
    print("WARNING: some filenames do not contain ID_<number>")

id_groups = (
    df.groupby("base_id")["split"]
      .nunique()
)

cross_id = id_groups[id_groups > 1]

print("Unique base IDs:", df["base_id"].nunique())
print("Base IDs crossing splits:", len(cross_id))

if len(cross_id) == 0:
    print("STATUS: PASS - no base ID crosses splits")
else:
    print("STATUS: FAIL - base IDs cross splits")
    
    bad_ids = cross_id.index.tolist()

    print()
    print(
        df[df["base_id"].isin(bad_ids)]
        [["base_id", "image_path", "label_str", "split"]]
        .sort_values("base_id")
        .to_string(index=False)
    )

# ------------------------------------------------------------
# 3. Label consistency
# ------------------------------------------------------------

print()
print("LABEL CONSISTENCY")
print("=" * 70)

label_cols = [
    "epidural",
    "subdural",
    "subarachnoid",
    "intraparenchymal",
    "intraventricular",
]

print("Label columns present:", all(c in df.columns for c in label_cols))

if all(c in df.columns for c in label_cols):
    print()
    print("Positive counts:")
    for c in label_cols:
        print(f"  {c}: {int(df[c].sum())}")

# ------------------------------------------------------------
# 4. Final status
# ------------------------------------------------------------

print()
print("=" * 70)
print("FINAL TBI-HemoCT LEAKAGE STATUS")
print("=" * 70)

if len(cross_split) == 0 and len(cross_id) == 0:
    print("STATUS: PASS")
    print("No exact duplicate or base-ID leakage detected.")
else:
    print("STATUS: REVIEW REQUIRED")
    print("Potential leakage was detected.")

print()