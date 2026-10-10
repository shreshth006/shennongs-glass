"""Human-readable explanations built on the classical (CellProfiler-style)
features: per-MoA morphological signatures and per-query narratives."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .profiles import FEATURE_TEXT

_BASE = {k[:-4]: v for k, v in FEATURE_TEXT.items() if k.endswith("_med")}


def feature_text(k: str) -> str:
    if k in FEATURE_TEXT:
        return FEATURE_TEXT[k]
    if k.endswith("_iqr") and k[:-4] in _BASE:
        return "variability of " + _BASE[k[:-4]]
    return k.replace("_", " ")


def moa_signatures(tm: pd.DataFrame, classical: np.ndarray, cols) -> pd.DataFrame:
    """Median robust-z of each classical feature per MoA (rows) -> heatmap."""
    df = pd.DataFrame(classical, columns=cols)
    df["moa"] = tm.moa.values
    return df.groupby("moa").median()


def describe(z: pd.Series, top: int = 4, thr: float = 1.5) -> list[str]:
    """Turn a robust-z profile into sentences, strongest deviations first."""
    z = z.dropna()
    z = z.reindex(z.abs().sort_values(ascending=False).index)
    out = []
    for k, v in z.head(top).items():
        if abs(v) < thr:
            break
        word = "increased" if v > 0 else "decreased"
        out.append(f"{feature_text(k)} {word} ({v:+.1f} SD vs. DMSO)")
    return out or ["no single morphological feature deviates strongly from DMSO"]
