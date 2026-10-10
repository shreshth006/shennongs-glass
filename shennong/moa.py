"""Mechanism-of-action (MoA) prediction, benchmark protocol and reliability.

Protocol follows Ljosa et al. (2013): 1-nearest-neighbour on treatment
profiles (cosine), with the query's own compound removed from the reference
set (NSC). NSCB additionally removes references imaged in the same batch, and
-- as is standard -- drops the two MoAs that only exist in a single batch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .profiles import l2

NSCB_EXCLUDE = {"Cholesterol-lowering", "Kinase inhibitors"}


def annotated(tmeta: pd.DataFrame, prof: np.ndarray):
    m = tmeta.moa.notna().values & (tmeta.compound != "DMSO").values
    return tmeta[m].reset_index(drop=True), prof[m]


def _masks(tm: pd.DataFrame, batch_aware: bool):
    comp = np.asarray(tm.compound, dtype=object)
    same = comp[:, None] == comp[None, :]
    if batch_aware:
        b = tm.batches.values
        share = np.array([[len(b[i] & b[j]) > 0 for j in range(len(b))] for i in range(len(b))])
        same = same | share
    return same


def class_scores(tm, prof, batch_aware=False, exclude_moa=None):
    """For every query treatment: max cosine similarity to the allowed
    references of each MoA. Returns (classes, S[n, n_classes])."""
    p = l2(prof)
    sim = p @ p.T
    banned = _masks(tm, batch_aware)
    classes = sorted(tm.moa.unique())
    S = np.full((len(tm), len(classes)), -np.inf)
    moa = tm.moa.values
    for ci, c in enumerate(classes):
        cols = moa == c
        s = np.where(banned[:, cols], -np.inf, sim[:, cols])
        S[:, ci] = s.max(1) if cols.any() else -np.inf
    if exclude_moa is not None:
        S[:, classes.index(exclude_moa)] = -np.inf
    return classes, S


def evaluate(tm, prof, batch_aware=False):
    if batch_aware:
        keep = ~tm.moa.isin(NSCB_EXCLUDE).values
        tm, prof = tm[keep].reset_index(drop=True), prof[keep]
    classes, S = class_scores(tm, prof, batch_aware)
    pred = np.array(classes)[S.argmax(1)]
    acc = float((pred == tm.moa.values).mean())
    # MoA-balanced accuracy (some MoAs have many more treatments)
    per = pd.Series(pred == tm.moa.values).groupby(tm.moa.values).mean()
    return {"accuracy": acc, "balanced_accuracy": float(per.mean()), "n": int(len(tm)),
            "pred": pred, "true": tm.moa.values, "classes": classes}


def softmax(S, tau=0.03):
    z = (S - S.max(1, keepdims=True)) / tau
    e = np.exp(np.where(np.isfinite(z), z, -np.inf))
    return e / e.sum(1, keepdims=True)


def conformal(tm, prof, alpha=0.1, tau=0.03):
    """Leave-one-compound-out split-conformal prediction sets.

    For each query, the calibration set is every treatment of *other*
    compounds; non-conformity = 1 - p(true MoA). Returns per-treatment sets
    plus empirical coverage and mean set size."""
    classes, S = class_scores(tm, prof)
    P = softmax(S, tau)
    ci = np.array([classes.index(m) for m in tm.moa.values])
    nc = 1 - P[np.arange(len(tm)), ci]
    comp = tm.compound.values
    sets, cover = [], []
    for i in range(len(tm)):
        cal = nc[comp != comp[i]]
        n = len(cal)
        q = np.quantile(cal, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher")
        s = [classes[k] for k in np.where(1 - P[i] <= q)[0]] or [classes[int(P[i].argmax())]]  # never empty
        sets.append(s)
        cover.append(tm.moa.values[i] in s)
    sizes = np.array([len(s) for s in sets])
    return {"alpha": alpha, "coverage": float(np.mean(cover)), "mean_set_size": float(sizes.mean()),
            "singleton_rate": float((sizes == 1).mean()), "sets": sets, "P": P, "classes": classes}


def novelty_auroc(tm, prof):
    """Can the system tell when a compound's MoA is NOT in its reference
    library? Compare max similarity to the library when the true MoA is
    present (in-distribution) vs. when the whole MoA is held out."""
    from sklearn.metrics import roc_auc_score
    classes, S = class_scores(tm, prof)
    ci = np.array([classes.index(m) for m in tm.moa.values])
    s_in = S.max(1)
    S_out = S.copy()
    S_out[np.arange(len(tm)), ci] = -np.inf
    s_out = S_out.max(1)
    y = np.r_[np.zeros(len(tm)), np.ones(len(tm))]
    return float(roc_auc_score(y, -np.r_[s_in, s_out])), s_in, s_out


def predict_query(q: np.ndarray, ref_tm: pd.DataFrame, ref_prof: np.ndarray, exclude_compound=None, k=5, tau=0.03):
    """MoA ranking + nearest reference treatments for an arbitrary profile."""
    p = l2(ref_prof)
    sim = p @ l2(q[None])[0]
    ok = np.ones(len(ref_tm), bool) if exclude_compound is None else (ref_tm.compound.values != exclude_compound)
    sim = np.where(ok, sim, -np.inf)
    classes = sorted(ref_tm.moa.unique())
    S = np.array([[sim[ref_tm.moa.values == c].max() for c in classes]])
    P = softmax(S, tau)[0]
    order = np.argsort(-sim)[:k]
    nn = ref_tm.iloc[order].assign(similarity=sim[order])
    return pd.Series(P, index=classes).sort_values(ascending=False), nn, float(sim.max())
