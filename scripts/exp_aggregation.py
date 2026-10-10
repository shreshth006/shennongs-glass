"""Post-hoc experiment: how replicate wells are aggregated into treatment profiles.

The main run (run_analysis.py) uses the median over replicate wells and raw
cosine similarity. Here we also try the mean, and centring every treatment
profile on the mean of all non-DMSO treatments (label-free). To keep the number
honest, the setting is also chosen by nested leave-one-compound-out selection:
for each held-out compound the best configuration is picked on the remaining
compounds only, then used to predict the held-out one.

Writes results/aggregation_experiment.csv and results/aggregation_nested.json.
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from shennong.moa import NSCB_EXCLUDE, _masks, annotated, evaluate  # noqa: E402
from shennong.profiles import build_profiles, l2, load_features, treatment_profiles  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--features", default=os.path.join(ROOT, "features"))
ap.add_argument("--out", default=os.path.join(ROOT, "results"))
args = ap.parse_args()

f = load_features(args.features)


def configs(n_comp):
    wmeta, _, B, _ = build_profiles(f, n_comp)
    g = wmeta.groupby(["compound", "conc"]).indices
    keys = sorted(g)
    tm_all, _ = treatment_profiles(wmeta, B["fusion"])
    nondmso = (tm_all.compound != "DMSO").values
    out = {}
    for w, agg, centre in itertools.product([0.4, 0.7, 1.0], ["median", "mean"], [False, True]):
        W = np.hstack([l2(B["dino_cell"]), l2(B["dino_field"]), w * l2(B["classical"])])
        fn = np.median if agg == "median" else np.mean
        P = np.stack([fn(W[g[k]], 0) for k in keys])
        if centre:
            P = P - P[nondmso].mean(0)
        out[(w, agg, centre)] = P
    return tm_all, out


def nested(tm_all, cfg, batch_aware):
    tm, _ = annotated(tm_all, next(iter(cfg.values())))
    keep = ~tm.moa.isin(NSCB_EXCLUDE).values if batch_aware else np.ones(len(tm), bool)
    tm = tm[keep].reset_index(drop=True)
    comp, moa, ban = tm.compound.values, tm.moa.values, _masks(tm, batch_aware)
    sims = {}
    for k, P in cfg.items():
        p = l2(annotated(tm_all, P)[1][keep])
        sims[k] = p @ p.T
    correct, chosen = [], []
    for c in np.unique(comp):
        hold = comp == c
        q_in, q_out = np.where(~hold)[0], np.where(hold)[0]
        best = max(sims, key=lambda k: (moa[np.where(ban | hold[None, :], -np.inf, sims[k])[q_in].argmax(1)] == moa[q_in]).mean())
        chosen.append(best)
        s = np.where(ban, -np.inf, sims[best])
        correct += list(moa[s[q_out].argmax(1)] == moa[q_out])
    return float(np.mean(correct)), pd.Series([str(c) for c in chosen]).value_counts().to_dict()


rows, nest = [], {}
for n_comp in [64, 128, 256]:
    tm_all, cfg = configs(n_comp)
    for (w, agg, centre), P in cfg.items():
        tm, p = annotated(tm_all, P)
        rows.append({"tvn_components": n_comp, "classical_weight": w, "aggregation": agg, "centred": centre,
                     "NSC": evaluate(tm, p)["accuracy"], "NSCB": evaluate(tm, p, batch_aware=True)["accuracy"]})
    if n_comp == 128:
        nest = {"NSC": nested(tm_all, cfg, False), "NSCB": nested(tm_all, cfg, True)}
df = pd.DataFrame(rows)
os.makedirs(args.out, exist_ok=True)
df.to_csv(os.path.join(args.out, "aggregation_experiment.csv"), index=False)
json.dump({k: {"accuracy": v[0], "chosen": v[1]} for k, v in nest.items()},
          open(os.path.join(args.out, "aggregation_nested.json"), "w"), indent=1)
print(df.pivot_table(index=["aggregation", "centred"], columns="tvn_components", values=["NSC", "NSCB"], aggfunc=["min", "max"]).round(3))
print("nested (TVN 128):", {k: round(v[0], 4) for k, v in nest.items()})

# ---- reliability with the improved setting (TVN 128, classical weight 0.7: the main-run choices)
from shennong import moa  # noqa: E402

tm_all, cfg = configs(128)
rel = {}
for label, key in [("main_run", (0.7, "median", False)), ("mean_centred", (0.7, "mean", True))]:
    tma, pa = annotated(tm_all, cfg[key])
    r = {"NSC": evaluate(tma, pa)["accuracy"], "NSCB": evaluate(tma, pa, batch_aware=True)["accuracy"]}
    for a in (0.05, 0.1, 0.2):
        c = moa.conformal(tma, pa, alpha=a)
        r[f"conformal_{a}"] = {k: c[k] for k in ("coverage", "mean_set_size", "singleton_rate")}
    r["novelty_auroc"] = moa.novelty_auroc(tma, pa)[0]
    rel[label] = r
json.dump(rel, open(os.path.join(args.out, "aggregation_reliability.json"), "w"), indent=1)
print(json.dumps(rel, indent=1))
