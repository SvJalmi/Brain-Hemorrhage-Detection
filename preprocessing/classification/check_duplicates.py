import hashlib
import pandas as pd
from pathlib import Path

df = pd.read_csv("preprocessing/classification/tbihemoct_manifest.csv")
df["md5"] = [hashlib.md5(Path(p).read_bytes()).hexdigest() for p in df["image_path"]]

dup = df[df.duplicated("md5", keep=False)]
print("Rows sharing an identical file:", len(dup))

# Do identical files cross split boundaries? (this is real leakage)
cross = dup.groupby("md5")["split"].nunique()
print("Identical-file groups spanning >1 split:", int((cross > 1).sum()))

df.to_csv("preprocessing/classification/tbihemoct_manifest_hashed.csv", index=False)