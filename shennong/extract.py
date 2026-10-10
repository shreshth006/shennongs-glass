"""Per-image feature extraction for new images (CPU or GPU).

Mirrors kaggle/features/features.py exactly, so that a field uploaded to the
demo app is described the same way as the 13,200 reference fields.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from skimage import feature, filters, measure, morphology, segmentation

CROP, HALF = 64, 2
# must match the configuration the reference features were extracted with
# (features/config.json, written by scripts/merge_features.py)
DEFAULT_CFG = {"model": "dinov2_vitb14", "emb_crop": 224, "field": [448, 560], "n_cells": 32}
NUC_PROPS = ["area", "perimeter", "eccentricity", "solidity", "major_axis_length", "minor_axis_length"]


def norm(img):
    lo, hi = np.percentile(img, (0.5, 99.8))
    return np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)


def segment(dapi, cyto):
    sm = filters.gaussian(dapi, 1)
    fg = sm > filters.threshold_otsu(sm)
    fg = morphology.remove_small_objects(fg, 30)
    dist = ndi.distance_transform_edt(fg)
    peaks = feature.peak_local_max(dist, min_distance=6, labels=measure.label(fg), exclude_border=False)
    markers = np.zeros(dist.shape, np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    nuc = segmentation.watershed(-dist, markers, mask=fg)
    csm = filters.gaussian(cyto, 2)
    cfg = csm > filters.threshold_otsu(csm) * 0.6
    cell = segmentation.expand_labels(nuc, 20) * (cfg | (nuc > 0))
    return nuc, cell


def process(dapi, tub, act, seed=0, cfg=DEFAULT_CFG):
    """Raw channel arrays -> (morph dict, crops, n_cells, field, half-res stack, nuclei labels)."""
    full = np.stack([norm(dapi), norm(tub), norm(act)])
    half = full[:, ::HALF, ::HALF]
    d, t, a = half
    nuc, cell = segment(d, np.maximum(t, a))
    nprops = measure.regionprops(nuc, intensity_image=d)
    cprops = {p.label: p for p in measure.regionprops(cell, intensity_image=np.stack([t, a], -1))}
    rows = []
    for p in nprops:
        if p.area < 30:
            continue
        c = cprops.get(p.label)
        r = {k: float(getattr(p, k)) for k in NUC_PROPS}
        r["dapi_mean"] = float(p.intensity_mean)
        r["dapi_int"] = float(p.intensity_mean * p.area)
        if c is not None:
            r["cell_area"] = float(c.area)
            r["tub_mean"], r["act_mean"] = map(float, c.intensity_mean)
            r["cell_nuc_ratio"] = c.area / p.area
            r["cell_ecc"] = float(c.eccentricity)
        rows.append(r)
    cells = pd.DataFrame(rows)
    morph = {"cell_count": float(len(cells))}
    for k in cells.columns:
        v = cells[k].dropna()
        morph[f"{k}_med"] = float(v.median()) if len(v) else np.nan
        morph[f"{k}_iqr"] = float(v.quantile(0.75) - v.quantile(0.25)) if len(v) else np.nan
    for j, nm in enumerate(("dapi", "tub", "act")):
        morph[f"{nm}_fg_frac"] = float((half[j] > filters.threshold_otsu(half[j])).mean())
        morph[f"{nm}_texture"] = float(np.abs(filters.sobel(half[j])).mean())

    h = CROP // 2
    pad = np.pad(half, ((0, 0), (h, h), (h, h)))
    cents = [tuple(map(int, p.centroid)) for p in nprops if p.area >= 30]
    rng = np.random.default_rng(seed)
    n_cells = cfg["n_cells"]
    if len(cents) > n_cells:
        cents = [cents[k] for k in rng.choice(len(cents), n_cells, replace=False)]
    crops = np.zeros((n_cells, 3, CROP, CROP), np.float32)
    for k, (y, x) in enumerate(cents):
        crops[k] = pad[:, y:y + CROP, x:x + CROP]
    from PIL import Image
    fh, fw = cfg["field"]
    field = np.stack([np.asarray(Image.fromarray(c).resize((fw, fh), Image.BOX)) for c in full])
    return morph, crops, len(cents), field, half, nuc


class Embedder:
    """DINOv2 ViT-B/14, used frozen (no fine-tuning on BBBC021)."""

    def __init__(self, cfg=DEFAULT_CFG, device=None):
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("facebookresearch/dinov2", cfg["model"], trust_repo=True).eval().to(self.device)
        self.D = self.model.embed_dim
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    @torch.no_grad()
    def _embed(self, x, size=None):
        x = torch.as_tensor(x, device=self.device).float()[:, None]
        if size:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        x = (x.repeat(1, 3, 1, 1) - self.mean) / self.std
        out = self.model.forward_features(x)
        return torch.cat([out["x_norm_clstoken"], out["x_norm_patchtokens"].mean(1)], 1).cpu().numpy()

    def __call__(self, crops, n_cells, field):
        D, ec, nc = self.D, self.cfg["emb_crop"], self.cfg["n_cells"]
        e = self._embed(crops.transpose(1, 0, 2, 3).reshape(-1, CROP, CROP), size=(ec, ec))[:, :D]
        e = e.reshape(3, nc, D)[:, :max(n_cells, 1)].mean(1).reshape(-1)
        ef = self._embed(field).reshape(-1)
        return e, ef
