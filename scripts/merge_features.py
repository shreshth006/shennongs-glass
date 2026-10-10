"""Merge Kaggle extraction outputs (one or more shards) into features/.

    python scripts/merge_features.py --src kaggle_out/cpu_* --config cpu
"""
import argparse
import json
import os
import tarfile

import numpy as np
import pandas as pd

CONFIGS = {
    # must match the constants used in kaggle/features/features.py
    "gpu": {"model": "dinov2_vitb14", "emb_crop": 224, "field": [448, 560], "n_cells": 32, "sites": [1, 2, 3, 4]},
    "cpu": {"model": "dinov2_vits14", "emb_crop": 112, "field": [224, 280], "n_cells": 16, "sites": [1, 3]},
}

ap = argparse.ArgumentParser()
ap.add_argument("--src", nargs="+", required=True)
ap.add_argument("--config", choices=CONFIGS, required=True)
ap.add_argument("--out", default="features")
ap.add_argument("--thumbs", default="artifacts")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
os.makedirs(a.thumbs, exist_ok=True)

metas, morphs, ec, ei = [], [], [], []
for d in sorted(a.src):
    if not os.path.exists(os.path.join(d, "meta.parquet")):
        print("skip (incomplete)", d)
        continue
    metas.append(pd.read_parquet(os.path.join(d, "meta.parquet")))
    morphs.append(pd.read_parquet(os.path.join(d, "morph.parquet")))
    ec.append(np.load(os.path.join(d, "emb_cell.npy")))
    ei.append(np.load(os.path.join(d, "emb_img.npy")))
    t = os.path.join(d, "thumbs.tar")
    if os.path.exists(t):
        with tarfile.open(t) as tf:
            tf.extractall(a.thumbs, filter="data")
    print("ok", d, len(metas[-1]))

meta = pd.concat(metas, ignore_index=True)
morph = pd.concat(morphs, ignore_index=True)
assert not meta.duplicated(["TableNumber", "ImageNumber"]).any()
meta.to_parquet(os.path.join(a.out, "meta.parquet"))
morph.to_parquet(os.path.join(a.out, "morph.parquet"))
np.save(os.path.join(a.out, "emb_cell.npy"), np.concatenate(ec))
np.save(os.path.join(a.out, "emb_img.npy"), np.concatenate(ei))
with open(os.path.join(a.out, "config.json"), "w") as fh:
    json.dump(CONFIGS[a.config], fh, indent=1)
print("merged", len(meta), "image sets ->", a.out)
