"""Dose-response: phenotypic activity, cytotoxicity and the window between them.

* viability  = cell count of a well / median DMSO cell count on the same plate
* activity   = distance of the well profile from the DMSO cloud in whitened
               space, expressed as a percentile-calibrated score against the
               distribution of DMSO wells themselves (leave-one-well-out).
Both are fitted with a 4-parameter Hill curve on log10 concentration.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def viability(wmeta, morph):
    dm = wmeta.compound == "DMSO"
    ref = morph.cell_count[dm].groupby(wmeta.plate[dm]).median()
    return (morph.cell_count / wmeta.plate.map(ref)).values


def activity(wmeta, x):
    """Phenotypic activity per well.

    Returns the raw distance from the plate's DMSO centre (in whitened units)
    and an empirical p-value against DMSO wells' own distances."""
    dm = (wmeta.compound == "DMSO").values
    d = np.linalg.norm(x, axis=1)
    null = np.sort(d[dm])
    p = 1 - np.searchsorted(null, d, side="left") / len(null)
    thr = float(np.quantile(null, 0.99))
    return d, np.clip(p, 1 / len(null), 1), thr


def hill(x, bottom, top, logec50, h):
    return bottom + (top - bottom) / (1 + 10 ** ((logec50 - x) * h))


def fit_hill(conc, y, increasing=True):
    """Fit a monotone 4-parameter Hill curve on log10(conc).

    The direction is enforced by parameterising the plateau difference as a
    non-negative amplitude: increasing fits go up with dose, decreasing fits
    (viability) go down. Returns bottom/top in the usual hill() convention and
    logEC50 (NaN if the fit fails)."""
    x = np.log10(conc)
    y = np.asarray(y, float)
    if len(np.unique(x)) < 3:
        return {"logec50": np.nan, "top": np.nan, "bottom": np.nan, "h": np.nan, "ok": False}
    sign = 1.0 if increasing else -1.0
    ys = sign * y                                   # fit an increasing curve to ys
    lo, hi = x.min() - 1, x.max() + 1

    def f(x_, base, amp, logec50, h):
        return hill(x_, base, base + amp, logec50, h)

    p0 = [np.min(ys), max(np.ptp(ys), 1e-3), np.median(x), 1.0]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # soft-L1 loss: robust to outlier wells (e.g. apoptotic nuclear fragments
            # inflating the cell count at the most toxic doses)
            scale = max(0.1 * np.ptp(ys), 1e-3)
            popt, _ = curve_fit(f, x, ys, p0=p0, bounds=([-np.inf, 0, lo, 0.3], [np.inf, np.inf, hi, 6]),
                                method="trf", loss="soft_l1", f_scale=scale, max_nfev=20000)
        base, amp, ec, h = popt
        return {"bottom": sign * base, "top": sign * (base + amp), "logec50": ec, "h": h, "ok": True}
    except Exception:
        return {"logec50": np.nan, "top": np.nan, "bottom": np.nan, "h": np.nan, "ok": False}


def compound_dose_table(wmeta, viab, act_d, act_thr):
    """Per-compound potency summary across its concentration series."""
    rows = []
    for comp, g in wmeta.groupby("compound"):
        if comp == "DMSO":
            continue
        idx = g.index.values
        conc = g.conc.values
        v, a = viab[idx], act_d[idx]
        fv = fit_hill(conc, v, increasing=False)
        fa = fit_hill(conc, a, increasing=True)
        # simple, assumption-light potency readouts
        by = pd.DataFrame({"conc": conc, "v": v, "a": a}).groupby("conc").median()
        # active = above threshold at this dose AND the next higher one (or the top
        # dose): guards against 1-in-100 vehicle false positives across 8 doses
        above = (by.a > act_thr).values
        sustained = above & np.r_[above[1:], True]
        active = by.index[sustained]
        toxic = by.index[by.v < 0.5]
        first_active = float(active.min()) if len(active) else np.nan
        first_toxic = float(toxic.min()) if len(toxic) else np.nan
        # effect must be real within tested range for the EC50 to count
        ec_a = fa["logec50"] if fa["ok"] and by.a.max() > act_thr else np.nan
        ec_v = fv["logec50"] if fv["ok"] and by.v.min() < 0.5 else np.nan
        rows.append({
            "compound": comp, "n_conc": len(by), "conc_min": by.index.min(), "conc_max": by.index.max(),
            "pheno_logEC50": ec_a, "tox_logIC50": ec_v,
            "first_active_conc": first_active, "first_toxic_conc": first_toxic,
            "min_viability": float(by.v.min()), "max_activity": float(by.a.max()),
            # window > 0: phenotype appears at doses below overt cytotoxicity
            "window_log10": (np.log10(first_toxic) - np.log10(first_active)) if np.isfinite(first_active) and np.isfinite(first_toxic) else np.nan,
            "fit_activity": fa, "fit_viability": fv,
        })
    return pd.DataFrame(rows)


TRIAGE = {
    "inactive": "inactive in tested range",
    "tox_only": "cytotoxic without a distinct phenotype",
    "nontox": "active, not cytotoxic in tested range",
    "wide": "active >=10x below cytotoxic dose",
    "narrow": "narrow window (<10x)",
    "tox_first": "phenotype only at cytotoxic doses",
}


def risk_label(row):
    """Plain-language triage label used in reports and the app."""
    act, tox = np.isfinite(row.first_active_conc), np.isfinite(row.first_toxic_conc)
    if not act:
        return TRIAGE["tox_only"] if tox else TRIAGE["inactive"]
    if not tox:
        return TRIAGE["nontox"]
    if row.window_log10 >= 1:
        return TRIAGE["wide"]
    if row.window_log10 > 0:
        return TRIAGE["narrow"]
    return TRIAGE["tox_first"]
