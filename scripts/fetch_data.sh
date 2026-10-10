#!/usr/bin/env bash
# Downloads everything needed to reproduce the results / run the demo.
# Requires a (free) Kaggle account: `kaggle auth login` once.
set -euo pipefail
cd "$(dirname "$0")/.."

# 1. BBBC021 metadata + MoA annotations (Broad Bioimage Benchmark Collection)
mkdir -p data/bbbc021
for f in BBBC021_v1_image.csv BBBC021_v1_compound.csv BBBC021_v1_moa.csv; do
  [ -f data/bbbc021/$f ] || curl -sSL -o data/bbbc021/$f https://data.broadinstitute.org/bbbc/BBBC021/$f
done

# 2. Pre-extracted features: outputs of the public Kaggle notebooks
#    kaggle/cpu_shards/shard_k (DINOv2 ViT-S, fields 1+3 of every well).
#    Shard 6 did not finish before the submission deadline; the reported
#    results use the other seven (every well still has >= 1 field).
for k in 0 1 2 3 4 5 7; do
  mkdir -p kaggle_out/cpu_$k
  [ -f kaggle_out/cpu_$k/meta.parquet ] || kaggle kernels output aerochrome/ai4s-feat-cpu-$k -p kaggle_out/cpu_$k
done
python scripts/merge_features.py --src kaggle_out/cpu_* --config cpu --out features --thumbs artifacts

# 3. Four example fields for the demo app (DMSO, taxol, colchicine, cytochalasin B)
python scripts/fetch_examples.py
