"""Loading extracted features and turning them into normalised profiles.

Hierarchy: image (field) -> well (mean of 4 sites) -> treatment
(compound x concentration, median over replicate wells).

Normalisation is anchored on the DMSO (vehicle) wells of each plate, which is
what makes profiles comparable across plates and imaging weeks.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

MOA_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "bbbc021", "BBBC021_v1_moa.csv")

RENAME = {
    "Image_Metadata_Plate_DAPI": "plate",
    "Image_Metadata_Well_DAPI": "well",
    "Image_Metadata_Compound": "compound",
    "Image_Metadata_Concentration": "conc",
    "Replicate": "replicate",
}

# Human-readable descriptions of the classical features (used in explanations)
FEATURE_TEXT = {
    "cell_count": "cell count (proliferation / viability)",
    "area_med": "nuclear area",
    "perimeter_med": "nuclear perimeter",
    "eccentricity_med": "nuclear elongation",
    "solidity_med": "nuclear solidity (low = fragmented / lobed)",
    "major_axis_length_med": "nuclear length",
    "minor_axis_length_med": "nuclear width",
    "dapi_mean_med": "DNA stain intensity (chromatin condensation)",
    "dapi_int_med": "integrated DNA content (cell-cycle stage)",
    "cell_area_med": "cell spread area",
    "tub_mean_med": "tubulin intensity",
    "act_mean_med": "actin intensity",
    "cell_nuc_ratio_med": "cytoplasm-to-nucleus ratio",
    "cell_ecc_med": "cell elongation",
    "area_iqr": "variability of nuclear size",
    "dapi_int_iqr": "variability of DNA content",
    "cell_area_iqr": "variability of cell size",
    "dapi_fg_frac": "nuclear coverage of the field",
    "tub_fg_frac": "tubulin coverage of the field",
    "act_fg_frac": "actin coverage of the field",
    "dapi_texture": "DNA texture",
    "tub_texture": "microtubule texture / fibrousness",
    "act_texture": "actin texture / stress fibres",
}


@dataclass
class Features:
    meta: pd.DataFrame          # one row per image
    morph: pd.DataFrame         # classical features per image
    emb_cell: np.ndarray        # DINOv2 single-cell embedding per image
    emb_img: np.ndarray         # DINOv2 whole-field embedding per image


def load_features(d: str) -> Features:
    meta = pd.read_parquet(os.path.join(d, "meta.parquet")).rename(columns=RENAME)
    meta = meta.astype({c: object for c in meta.columns if meta[c].dtype.kind not in "biufc"})
    meta["batch"] = meta.plate.str.split("_").str[0]
    morph = pd.read_parquet(os.path.join(d, "morph.parquet"))
    keep = [c for c in FEATURE_TEXT if c in morph.columns] + sorted(c for c in morph.columns if c not in FEATURE_TEXT)
    morph = morph[keep].astype(float)
    return Features(meta, morph,
                    np.load(os.path.join(d, "emb_cell.npy")).astype(np.float32),
                    np.load(os.path.join(d, "emb_img.npy")).astype(np.float32))


def to_wells(f: Features):
    """Average the 4 fields of view of each well."""
    key = f.meta.plate + "|" + f.meta.well
    codes, uniq = pd.factorize(key)
    n = len(uniq)
    cnt = np.bincount(codes, minlength=n)[:, None]

    def agg(x):
        out = np.zeros((n, x.shape[1]), np.float64)
        np.add.at(out, codes, np.nan_to_num(x))
        return out / cnt

    wmeta = f.meta.groupby(codes).first().reset_index(drop=True)
    wmeta["n_fields"] = cnt[:, 0]
    morph = pd.DataFrame(agg(f.morph.values), columns=f.morph.columns)
    morph["cell_count"] = morph["cell_count"] * 4 / wmeta["n_fields"].values  # per-4-field count
    return wmeta, morph, agg(f.emb_cell), agg(f.emb_img)


def robust_z(x: np.ndarray, plates: np.ndarray, dmso: np.ndarray) -> np.ndarray:
    """Per-plate centring on the DMSO median, scaled by the pooled DMSO SD."""
    x = x.astype(np.float64).copy()
    for p in np.unique(plates):
        m = plates == p
        x[m] -= np.nanmedian(x[m & dmso], 0)
    sd = np.nanstd(x[dmso], 0)
    return x / np.where(sd > 1e-9, sd, 1.0)


class TVN:
    """Typical-variation normalisation: per-plate DMSO centring followed by
    PCA whitening fitted on DMSO wells only (Ando et al., 2017)."""

    def __init__(self, n_comp: int = 128, eps: float = 1e-3):
        self.n_comp, self.eps = n_comp, eps

    def fit_transform(self, x, plates, dmso):
        x = x.astype(np.float64).copy()
        self.plate_mu = {}
        for p in np.unique(plates):
            m = plates == p
            self.plate_mu[p] = x[m & dmso].mean(0)
            x[m] -= self.plate_mu[p]
        self.global_mu = np.mean(list(self.plate_mu.values()), 0)
        d = x[dmso]
        u, s, vt = np.linalg.svd(d - d.mean(0), full_matrices=False)
        k = min(self.n_comp, len(s))
        self.mu = d.mean(0)
        self.w = vt[:k].T / (s[:k] / np.sqrt(len(d) - 1) + self.eps)
        return (x - self.mu) @ self.w

    def transform_new(self, x):
        """For images without plate-matched controls (demo uploads): centre on
        the average DMSO profile across all plates."""
        return (x - self.global_mu - self.mu) @ self.w


def l2(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def build_profiles(f: Features, n_comp: int = 128):
    """Returns well-level metadata, a dict of well-level feature blocks and the
    fitted normalisers (needed by the demo app)."""
    wmeta, morph, ecell, eimg = to_wells(f)
    plates = wmeta.plate.values
    dmso = (wmeta.compound == "DMSO").values
    zc = robust_z(morph.values, plates, dmso)
    zc = np.clip(np.nan_to_num(zc), -20, 20)
    tvn_cell, tvn_img = TVN(n_comp), TVN(n_comp)
    tvn_act = tvn_cell  # activity uses the full whitened cell space (most sensitive at 1% FPR)
    blocks = {
        "classical": zc,
        "dino_cell": tvn_cell.fit_transform(ecell, plates, dmso),
        "dino_field": tvn_img.fit_transform(eimg, plates, dmso),
        # activity score space (= single-cell TVN; kept as its own key for clarity)
        "activity": tvn_act.fit_transform(ecell, plates, dmso),
    }
    blocks["fusion"] = np.hstack([l2(blocks["dino_cell"]), l2(blocks["dino_field"]), 0.7 * l2(blocks["classical"])])
    norms = {"tvn_cell": tvn_cell, "tvn_img": tvn_img, "tvn_act": tvn_act,
             "morph_cols": list(morph.columns),
             "morph_dmso_med": np.nanmedian(morph.values[dmso], 0),
             "morph_mad": np.nanstd(morph.values[dmso] - np.nanmedian(morph.values[dmso], 0), 0),
             "dmso_count_by_plate": wmeta[dmso].assign(c=morph.cell_count.values[dmso]).groupby("plate").c.median().to_dict()}
    return wmeta, morph, blocks, norms


def treatment_profiles(wmeta: pd.DataFrame, x: np.ndarray):
    """Median over replicate wells per (compound, concentration)."""
    g = wmeta.groupby(["compound", "conc"]).indices
    keys = sorted(g)
    prof = np.stack([np.median(x[g[k]], 0) for k in keys])
    tmeta = pd.DataFrame(keys, columns=["compound", "conc"])
    tmeta["n_wells"] = [len(g[k]) for k in keys]
    tmeta["batches"] = [frozenset(wmeta.batch.values[g[k]]) for k in keys]
    moa = pd.read_csv(MOA_CSV)
    tmeta = tmeta.merge(moa, left_on=["compound", "conc"], right_on=["compound", "concentration"], how="left").drop(columns="concentration")
    return tmeta, prof
