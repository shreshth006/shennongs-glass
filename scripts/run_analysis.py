"""End-to-end analysis: features -> profiles -> benchmark, reliability,
dose-response, interpretation. Writes results/ (metrics + figures) and
artifacts/ (everything the demo app needs).

    python scripts/run_analysis.py --features features/
"""
import argparse
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shennong import dose, interpret, moa  # noqa: E402
from shennong.profiles import build_profiles, l2, load_features, treatment_profiles  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--features", default="features")
ap.add_argument("--out", default="results")
ap.add_argument("--artifacts", default="artifacts")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
os.makedirs(args.artifacts, exist_ok=True)
FIG = lambda n: os.path.join(args.out, n)  # noqa: E731
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

print("loading features ...")
f = load_features(args.features)
wmeta, morph, blocks, norms = build_profiles(f)
print("wells", len(wmeta), {k: v.shape for k, v in blocks.items()})

# ablation: DINOv2 cell embeddings with plate centring only (no whitening)
from shennong.profiles import to_wells  # noqa: E402
_, _, ecell_raw, _ = to_wells(f)
x = ecell_raw.copy()
dm = (wmeta.compound == "DMSO").values
for p in wmeta.plate.unique():
    m = (wmeta.plate == p).values
    x[m] -= x[m & dm].mean(0)
blocks_eval = {k: v for k, v in blocks.items() if k != "activity"}
blocks_eval["dino_cell_no_tvn"] = x

# ------------------------------------------------------------------ benchmark
metrics = {"n_images": int(len(f.meta)), "n_wells": int(len(wmeta)), "n_morph": int(f.morph.shape[1]), "benchmark": {}}
rows = []
for name, xb in blocks_eval.items():
    tm, prof = treatment_profiles(wmeta, xb)
    tma, pa = moa.annotated(tm, prof)
    r1, r2 = moa.evaluate(tma, pa), moa.evaluate(tma, pa, batch_aware=True)
    metrics["benchmark"][name] = {"NSC": r1["accuracy"], "NSC_balanced": r1["balanced_accuracy"],
                                  "NSCB": r2["accuracy"], "NSCB_balanced": r2["balanced_accuracy"],
                                  "n_NSC": r1["n"], "n_NSCB": r2["n"]}
    rows += [{"features": name, "protocol": "NSC", "accuracy": r1["accuracy"]},
             {"features": name, "protocol": "NSCB", "accuracy": r2["accuracy"]}]
    print(f"{name:18s} NSC {r1['accuracy']:.3f}  NSCB {r2['accuracy']:.3f}")

# robustness: the main configuration (128 TVN components, classical weight 0.7)
# was fixed before any results; report how much the benchmark moves around it
from shennong.profiles import TVN  # noqa: E402
_, _, ecell_w, eimg_w = to_wells(f)
plates_w = wmeta.plate.values
grid = []
for k in (32, 64, 128, 256):
    c_k = TVN(k).fit_transform(ecell_w, plates_w, dm)
    f_k = TVN(k).fit_transform(eimg_w, plates_w, dm)
    variants = {"cells": c_k, "field": f_k, "cells+field": np.hstack([l2(c_k), l2(f_k)]),
                "fusion (w=0.4)": np.hstack([l2(c_k), l2(f_k), 0.4 * l2(blocks["classical"])]),
                "fusion (w=0.7)": np.hstack([l2(c_k), l2(f_k), 0.7 * l2(blocks["classical"])])}
    for vname, xv in variants.items():
        tm_v, p_v = treatment_profiles(wmeta, xv)
        a_v, pa_v = moa.annotated(tm_v, p_v)
        grid.append({"tvn_components": k, "profile": vname,
                     "NSC": moa.evaluate(a_v, pa_v)["accuracy"], "NSCB": moa.evaluate(a_v, pa_v, batch_aware=True)["accuracy"]})
grid = pd.DataFrame(grid)
grid.to_csv(os.path.join(args.out, "sensitivity_grid.csv"), index=False)
metrics["sensitivity"] = {"NSC_min": float(grid.NSC.min()), "NSC_max": float(grid.NSC.max()), "NSC_median": float(grid.NSC.median()),
                          "NSCB_min": float(grid.NSCB.min()), "NSCB_max": float(grid.NSCB.max()), "NSCB_median": float(grid.NSCB.median()),
                          "fusion_NSC_min": float(grid[grid.profile.str.startswith("fusion")].NSC.min()),
                          "fusion_NSC_max": float(grid[grid.profile.str.startswith("fusion")].NSC.max())}
print(grid.pivot(index="profile", columns="tvn_components", values="NSC").round(3))

bench = pd.DataFrame(rows)
labels = {"classical": "Classical\nmorphology", "dino_cell_no_tvn": "DINOv2 cells\n(no TVN)",
          "dino_cell": "DINOv2 cells", "dino_field": "DINOv2 field", "fusion": "Fusion\n(Shennong's Glass)"}
order = ["classical", "dino_cell_no_tvn", "dino_field", "dino_cell", "fusion"]
plt.figure(figsize=(8, 4))
ax = sns.barplot(data=bench, x="features", y="accuracy", hue="protocol", order=order, palette=["#3b6fb6", "#e08a2e"])
ax.set_xticks(range(len(order)), [labels[o] for o in order])
for c in ax.containers:
    ax.bar_label(c, fmt="%.2f", fontsize=8)
ax.set(xlabel="", ylabel="1-NN MoA accuracy", ylim=(0, 1.05), title="BBBC021 mechanism-of-action benchmark (held-out compounds)")
plt.tight_layout(); plt.savefig(FIG("fig_benchmark.png"), dpi=200); plt.close()

# ------------------------------------------------------- main model = fusion
BEST = "fusion"
tm_all, prof_all = treatment_profiles(wmeta, blocks[BEST])
tma, pa = moa.annotated(tm_all, prof_all)
_, cls_all = treatment_profiles(wmeta, blocks["classical"])
_, cls_a = moa.annotated(tm_all, cls_all)

r = moa.evaluate(tma, pa)
cm = pd.crosstab(pd.Series(r["true"], name="true MoA"), pd.Series(r["pred"], name="predicted MoA")).reindex(index=r["classes"], columns=r["classes"], fill_value=0)
plt.figure(figsize=(8, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title(f"NSC confusion matrix (fusion, accuracy {r['accuracy']:.2f})")
plt.tight_layout(); plt.savefig(FIG("fig_confusion.png"), dpi=200); plt.close()

# UMAP atlas of all 900+ treatments; annotated ones coloured by MoA
import umap  # noqa: E402
emb2 = umap.UMAP(n_neighbors=15, min_dist=0.3, metric="cosine", random_state=0).fit_transform(l2(prof_all))
tm_all["umap_x"], tm_all["umap_y"] = emb2[:, 0], emb2[:, 1]
plt.figure(figsize=(8, 6.5))
un = tm_all.moa.isna()
plt.scatter(emb2[un, 0], emb2[un, 1], s=6, c="#cccccc", label="not annotated")
pal = sns.color_palette("tab20", tm_all.moa.nunique())
for i, m in enumerate(sorted(tm_all.moa.dropna().unique())):
    s = (tm_all.moa == m).values
    plt.scatter(emb2[s, 0], emb2[s, 1], s=28, color=pal[i], label=m, edgecolor="k", linewidth=0.3)
plt.legend(fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
plt.title("Phenotypic atlas of 906 compound-dose treatments (UMAP)")
plt.xticks([]); plt.yticks([])
plt.tight_layout(); plt.savefig(FIG("fig_umap.png"), dpi=200); plt.close()

# --------------------------------------------------------------- reliability
conf = {}
for a in (0.05, 0.1, 0.2):
    c = moa.conformal(tma, pa, alpha=a)
    conf[a] = c
    metrics.setdefault("conformal", {})[str(a)] = {k: c[k] for k in ("coverage", "mean_set_size", "singleton_rate")}
auroc, s_in, s_out = moa.novelty_auroc(tma, pa)
metrics["novelty_auroc"] = auroc

fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
al = np.array([0.05, 0.1, 0.2])
axs[0].plot(1 - al, [conf[a]["coverage"] for a in al], "o-", label="empirical coverage")
axs[0].plot([0.75, 1], [0.75, 1], "k--", lw=1, label="target")
for a in al:
    axs[0].annotate(f"set size {conf[a]['mean_set_size']:.1f}", (1 - a, conf[a]["coverage"]), textcoords="offset points", xytext=(5, -12), fontsize=8)
axs[0].set(xlabel="target coverage (1 - alpha)", ylabel="coverage on held-out compounds", title="Conformal MoA prediction sets")
axs[0].legend(fontsize=8)
axs[1].hist(s_in, bins=25, alpha=0.7, label="MoA in library")
axs[1].hist(s_out, bins=25, alpha=0.7, label="MoA held out (novel)")
axs[1].set(xlabel="max similarity to reference library", ylabel="treatments", title=f"Novel-mechanism detection (AUROC {auroc:.2f})")
axs[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG("fig_reliability.png"), dpi=200); plt.close()

# replicate reproducibility: are replicate wells closer to each other than to random wells?
xw = l2(blocks[BEST])
rep, rep_keys = [], []
rng = np.random.default_rng(0)
for (c, k), g in wmeta[wmeta.compound != "DMSO"].groupby(["compound", "conc"]):
    idx = g.index.values
    if len(idx) < 2:
        continue
    s = xw[idx] @ xw[idx].T
    rep.append(s[np.triu_indices(len(idx), 1)].mean())
    rep_keys.append((c, k))
rep = np.array(rep)
null = []
for _ in range(2000):
    i, j = rng.choice(len(xw), 2, replace=False)
    null.append(xw[i] @ xw[j])
thr95 = np.quantile(null, 0.95)
metrics["percent_replicating"] = float((rep > thr95).mean())
ann_keys = set(zip(tma.compound, tma.conc))
rep_ann = np.array([r for r, key in zip(rep, rep_keys) if key in ann_keys])
metrics["percent_replicating_annotated"] = float((rep_ann > thr95).mean())

# ------------------------------------------------------------ dose-response
viab = dose.viability(wmeta, morph)
act_d, act_p, act_thr = dose.activity(wmeta, blocks["activity"])
wmeta["viability"], wmeta["activity"], wmeta["activity_p"] = viab, act_d, act_p
dt = dose.compound_dose_table(wmeta, viab, act_d, act_thr)
dt["triage"] = dt.apply(dose.risk_label, axis=1)
metrics["dose"] = {"n_compounds": int(len(dt)), "activity_threshold": act_thr,
                   "triage_counts": dt.triage.value_counts().to_dict()}

# MoA prediction for every treatment (incl. ~800 un-annotated doses)
preds = []
for i, row in tm_all.iterrows():
    if row.compound == "DMSO":
        preds.append(("DMSO", 1.0, 1.0)); continue
    P, nn, smax = moa.predict_query(prof_all[i], tma, pa, exclude_compound=row.compound)
    preds.append((P.index[0], float(P.iloc[0]), smax))
tm_all["pred_moa"], tm_all["pred_conf"], tm_all["max_sim"] = zip(*preds)
tm_all = tm_all.merge(wmeta.groupby(["compound", "conc"])[["viability", "activity"]].median().reset_index(), on=["compound", "conc"])

# one annotated example per triage category (chosen by code, not by hand)
annotated_c = set(tma.compound)
ex = []
for key in ("wide", "nontox", "narrow", "tox_first"):
    cand = dt[(dt.triage == dose.TRIAGE[key]) & dt.compound.isin(annotated_c)].sort_values("max_activity", ascending=False)
    if len(cand):
        ex.append(cand.compound.iloc[0])
fig, axs = plt.subplots(1, len(ex), figsize=(4.4 * len(ex), 3.6), squeeze=False)
for ax, comp in zip(axs[0], ex):
    g = wmeta[wmeta.compound == comp]
    row = dt[dt.compound == comp].iloc[0]
    by = g.groupby("conc")[["viability", "activity"]]
    med, q1, q3 = by.median(), by.quantile(0.25), by.quantile(0.75)
    lx = np.log10(med.index.values)
    xs = np.linspace(lx.min(), lx.max(), 100)
    ax.errorbar(lx, med.viability, yerr=[med.viability - q1.viability, q3.viability - med.viability], fmt="o", ms=4, c="#3b6fb6", capsize=2)
    if row.fit_viability["ok"]:
        fv = row.fit_viability; ax.plot(xs, dose.hill(xs, fv["bottom"], fv["top"], fv["logec50"], fv["h"]), c="#3b6fb6")
    ax.set_ylabel("viability (rel. DMSO)", color="#3b6fb6"); ax.set_ylim(0, 1.4)
    ax2 = ax.twinx()
    ax2.errorbar(lx, med.activity, yerr=[med.activity - q1.activity, q3.activity - med.activity], fmt="s", ms=4, c="#d1495b", capsize=2)
    if row.fit_activity["ok"]:
        fa = row.fit_activity; ax2.plot(xs, dose.hill(xs, fa["bottom"], fa["top"], fa["logec50"], fa["h"]), c="#d1495b")
    ax2.axhline(act_thr, ls="--", c="#d1495b", lw=0.8)
    ax2.set_ylabel("phenotypic activity", color="#d1495b"); ax2.grid(False)
    ax2.set_ylim(bottom=0)
    moa_c = tma[tma.compound == comp].moa.iloc[0]
    ax.set_title(f"{comp} ({moa_c})\n{row.triage}", fontsize=9); ax.set_xlabel("log10 concentration (uM)")
plt.tight_layout(w_pad=2.5); plt.savefig(FIG("fig_dose_examples.png"), dpi=200); plt.close()

fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
both = dt[np.isfinite(dt.first_active_conc) & np.isfinite(dt.first_toxic_conc)]
pal_t = dict(zip(dose.TRIAGE.values(), sns.color_palette("Set2", len(dose.TRIAGE))))
for lab, g in both.groupby("triage"):
    axs[0].scatter(np.log10(g.first_active_conc), np.log10(g.first_toxic_conc), s=36, color=pal_t[lab], label=lab, edgecolor="k", linewidth=0.3)
lo = np.log10(min(both.first_active_conc.min(), both.first_toxic_conc.min())) - 0.3
hi = np.log10(max(both.first_active_conc.max(), both.first_toxic_conc.max())) + 0.3
axs[0].plot([lo, hi], [lo, hi], "k--", lw=1)
axs[0].plot([lo, hi], [lo + 1, hi + 1], "k:", lw=1)
axs[0].set(xlabel="log10 lowest phenotypically active conc. (uM)", ylabel="log10 lowest cytotoxic conc. (uM)",
           title=f"Compounds with both effects (n={len(both)})")
axs[0].legend(fontsize=7, loc="upper left")
cnt = dt.triage.value_counts().reindex(list(dose.TRIAGE.values()), fill_value=0)
axs[1].barh(range(len(cnt)), cnt.values, color=[pal_t[k] for k in cnt.index], edgecolor="k", linewidth=0.3)
axs[1].set_yticks(range(len(cnt)), cnt.index, fontsize=8); axs[1].invert_yaxis()
for i, v in enumerate(cnt.values):
    axs[1].text(v + 0.5, i, str(v), va="center", fontsize=8)
axs[1].set(xlabel="compounds", title=f"Triage of all {len(dt)} compounds")
plt.tight_layout(); plt.savefig(FIG("fig_window.png"), dpi=200); plt.close()

# ------------------------------------------------------------ interpretation
sig = interpret.moa_signatures(tma, cls_a, norms["morph_cols"])
top = sig.std().sort_values(ascending=False).index[:14]   # most MoA-discriminative
sig_plot = sig[top].rename(columns=interpret.feature_text)
plt.figure(figsize=(11, 6.5))
sns.heatmap(sig_plot.clip(-6, 6), cmap="RdBu_r", center=0, vmin=-6, vmax=6, annot=True, fmt=".1f", annot_kws={"size": 7},
            cbar_kws={"label": "median z vs DMSO (SD units, clipped at 6)"})
plt.xticks(rotation=40, ha="right"); plt.ylabel("")
plt.title("Interpretable morphological signature of each MoA")
plt.tight_layout(); plt.savefig(FIG("fig_signatures.png"), dpi=200); plt.close()

# ----------------------------------------------------------------- save all
dt.drop(columns=["fit_activity", "fit_viability"]).to_csv(os.path.join(args.out, "compound_dose_table.csv"), index=False)
tm_all.drop(columns=["batches"]).to_csv(os.path.join(args.out, "treatment_predictions.csv"), index=False)

_, cls_prof_all = treatment_profiles(wmeta, blocks["classical"])
# single-field null (uploads are one field, no plate-matched DMSO): distance of
# individual DMSO fields after global-DMSO centring, the same transform the app uses
dimg = (f.meta.compound == "DMSO").values
act_null_img = np.sort(np.linalg.norm(norms["tvn_act"].transform_new(f.emb_cell[dimg]), axis=1))
metrics["single_field_activity_threshold"] = float(np.quantile(act_null_img, 0.99))
# single-field spread of the classical features on DMSO (for explaining uploads)
_mi = f.morph.values[dimg]
norms["morph_img_sd"] = np.nanstd(_mi - np.nanmedian(_mi, 0), 0)
# novelty threshold: 5th percentile of in-library max similarity
metrics["novelty_threshold"] = float(np.quantile(s_in, 0.05))
thumbs = f.meta.assign(thumb=f.meta.ImageNumber.astype(int).astype(str) + "_" + f.meta.TableNumber.astype(int).astype(str) + ".jpg")
thumbs = thumbs.groupby(["compound", "conc"]).thumb.apply(list).to_dict()
with open(os.path.join(args.artifacts, "reference.pkl"), "wb") as fh:
    pickle.dump({"tm_all": tm_all, "prof_all": prof_all, "cls_all": cls_prof_all, "tma": tma, "pa": pa,
                 "norms": norms, "dose_table": dt, "wells": wmeta.assign(**{c: morph[c].values for c in ["cell_count"]}),
                 "act_thr": act_thr, "act_null": np.sort(act_d[(wmeta.compound == "DMSO").values]),
                 "act_null_img": act_null_img, "thumbs": thumbs,
                 "config": json.load(open(os.path.join(args.features, "config.json"))) if os.path.exists(os.path.join(args.features, "config.json")) else None,
                 "conformal_q": {a: float(np.quantile(1 - conf[a]["P"][np.arange(len(tma)), [conf[a]["classes"].index(m) for m in tma.moa]], 1 - a)) for a in conf},
                 "metrics": metrics}, fh)
with open(os.path.join(args.out, "metrics.json"), "w") as fh:
    json.dump(metrics, fh, indent=2, default=float)
print(json.dumps({k: v for k, v in metrics.items() if k != "dose"}, indent=1, default=float))
print("triage", metrics["dose"]["triage_counts"])
