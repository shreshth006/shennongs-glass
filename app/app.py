"""Shennong's Glass demo: interactive phenotypic drug-screening assistant.

    python app/app.py            # then open http://127.0.0.1:7860

Tabs
  1. Analyze images   - upload DAPI / tubulin / actin fields -> segmentation,
                        MoA ranking with conformal set, novelty flag,
                        phenotypic activity, plain-language report
  2. Compound explorer - dose-response, cytotoxicity window and dose-wise MoA
                        for all 112 BBBC021 compounds
  3. Phenotypic atlas  - UMAP of 906 compound-dose treatments
  4. Benchmark         - validation figures and metrics
"""
import glob
import os
import pickle
import sys
import tarfile

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import tifffile
from PIL import Image
from skimage.segmentation import find_boundaries

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from shennong import dose, moa  # noqa: E402
from shennong.interpret import describe  # noqa: E402
from shennong.profiles import l2  # noqa: E402

ART = os.environ.get("SHENNONG_ART", os.path.join(ROOT, "artifacts"))
RES = os.environ.get("SHENNONG_RES", os.path.join(ROOT, "results"))
with open(os.path.join(ART, "reference.pkl"), "rb") as fh:
    R = pickle.load(fh)
TM, PROF, TMA, PA, NORMS, DT = R["tm_all"], R["prof_all"], R["tma"], R["pa"], R["norms"], R["dose_table"]
MET = R["metrics"]
from shennong.extract import DEFAULT_CFG  # noqa: E402
CFG = R.get("config") or DEFAULT_CFG
THUMBS = os.path.join(ART, "thumbs")
if not os.path.isdir(THUMBS) and os.path.exists(os.path.join(ART, "thumbs.tar")):
    with tarfile.open(os.path.join(ART, "thumbs.tar")) as t:
        t.extractall(ART)

_EMB = None


def embedder():
    global _EMB
    if _EMB is None:
        from shennong.extract import Embedder
        _EMB = Embedder(CFG)
    return _EMB


def thumb(compound, conc, k=0):
    lst = R["thumbs"].get((compound, conc), [])
    if not lst:
        return None
    p = os.path.join(THUMBS, lst[min(k, len(lst) - 1)])
    return p if os.path.exists(p) else None


def conformal_set(P, alpha=0.1):
    q = R["conformal_q"][alpha]
    return [c for c, p in P.items() if 1 - p <= q] or [P.index[0]]


# --------------------------------------------------------------- tab 1
def analyze(dapi_f, tub_f, act_f, holdout="(none)"):
    if not (dapi_f and tub_f and act_f):
        raise gr.Error("Please provide all three channels (DAPI, tubulin, actin).")
    from shennong.extract import process
    chans = [tifffile.imread(f if isinstance(f, str) else f.name) for f in (dapi_f, tub_f, act_f)]
    morph, crops, n, field, half, nuc = process(*chans, cfg=CFG)
    ecell, efield = embedder()(crops, n, field)

    # same normalisation as the reference, using global DMSO statistics
    zc = np.array([(morph.get(c, np.nan) - NORMS["morph_dmso_med"][i]) / (NORMS["morph_mad"][i] or 1)
                   for i, c in enumerate(NORMS["morph_cols"])])
    # explanations for a single field use single-field DMSO variability
    sd_img = NORMS.get("morph_img_sd", NORMS["morph_mad"])
    z_expl = np.array([(morph.get(c, np.nan) - NORMS["morph_dmso_med"][i]) / (sd_img[i] or 1)
                       for i, c in enumerate(NORMS["morph_cols"])])
    zc = np.clip(np.nan_to_num(zc), -20, 20)
    xc = NORMS["tvn_cell"].transform_new(ecell[None])[0]
    xa = NORMS["tvn_act"].transform_new(ecell[None])[0]
    xf = NORMS["tvn_img"].transform_new(efield[None])[0]
    q = np.hstack([l2(xc[None])[0], l2(xf[None])[0], 0.7 * l2(zc[None])[0]])

    excl = None if holdout in (None, "", "(none)") else holdout
    P, nn, smax = moa.predict_query(q, TMA, PA, exclude_compound=excl, k=6)
    cset = conformal_set(P)
    act = float(np.linalg.norm(xa))
    null = R["act_null_img"]
    pct = float(np.searchsorted(null, act) / len(null) * 100)
    novel = smax < MET["novelty_threshold"]

    rgb = np.stack([half[2], half[1], half[0]], -1).copy()
    rgb[find_boundaries(nuc)] = [1, 1, 0]
    overlay = Image.fromarray((rgb * 255).astype(np.uint8))

    zser = pd.Series(z_expl, index=NORMS["morph_cols"])
    # single fields are noisier than plate-matched wells: three honest tiers
    tier = "active" if pct >= 95 else ("weak" if pct >= 80 else "inactive")
    active = tier != "inactive"
    lines = [
        f"### Report" + (f"  \n_Blind test: **{excl}** removed from the reference library before prediction._" if excl else ""),
        f"**Cells detected:** {int(morph['cell_count'])} in this field "
        f"(DMSO fields typically {int(np.median([v for v in NORMS['dmso_count_by_plate'].values()]) / 4)})",
        f"**Phenotypic activity:** {act:.1f} - stronger than **{pct:.0f}%** of untreated (DMSO) fields"
        + {"active": " -> **active**", "weak": " -> **possibly active** (borderline for a single field)",
           "inactive": " -> within normal vehicle variation"}[("active" if pct >= 95 else ("weak" if pct >= 80 else "inactive"))],
    ]
    if tier == "weak":
        lines.append("_Low-confidence call: the phenotype is weak in this single field. Analyse more fields "
                     "or replicate wells before acting on it._")
    if active:
        lines += [
            f"**Most likely mechanism:** {P.index[0]} (p = {P.iloc[0]:.2f}); runner-up {P.index[1]} ({P.iloc[1]:.2f})",
            f"**90% conformal prediction set:** {', '.join(cset)} _(coverage is guaranteed for treatment-level "
            f"profiles - replicate wells - and is optimistic for a single field)_"
            + ("  \n_A set with several members means the image is ambiguous between these mechanisms._" if len(cset) > 1 else ""),
        ]
    else:
        lines.append("**Mechanism:** no call - the cells are within normal vehicle variation, so any "
                     "mechanism ranking would be noise (the table below is shown for transparency only).")
    if novel and active:
        lines.append(f"**Novelty warning:** max similarity to the reference library is {smax:.2f}, below the "
                     f"5th percentile of known compounds - this may be a mechanism *not* in the library.")
    lines.append("**What changed in the cells:**")
    lines += [f"- {s}" for s in describe(zser)]
    gallery = [(thumb(r.compound, r.conc), f"{r.compound} {r.conc:g} uM | {r.moa} | sim {r.similarity:.2f}")
               for r in nn.itertuples() if thumb(r.compound, r.conc)]
    probs = P.head(6).round(3).rename("probability").reset_index().rename(columns={"index": "mechanism"})
    return overlay, "\n\n".join(lines), probs, gallery


# --------------------------------------------------------------- tab 2
def explore(compound):
    g = TM[TM.compound == compound].sort_values("conc")
    row = DT[DT.compound == compound].iloc[0]
    wells = R["wells"]
    w = wells[wells.compound == compound]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    xs = np.linspace(np.log10(w.conc.min()), np.log10(w.conc.max()), 100)
    ax.scatter(np.log10(w.conc), w.viability, s=12, c="#3b6fb6", alpha=0.6)
    fv, fa = row.fit_viability, row.fit_activity
    if fv["ok"]:
        ax.plot(xs, dose.hill(xs, fv["bottom"], fv["top"], fv["logec50"], fv["h"]), c="#3b6fb6")
    ax.set(xlabel="log10 concentration (uM)", ylim=(0, 1.4))
    ax.set_ylabel("viability (cell count rel. DMSO)", color="#3b6fb6")
    ax2 = ax.twinx()
    ax2.scatter(np.log10(w.conc), w.activity, s=12, c="#d1495b", alpha=0.6)
    if fa["ok"]:
        ax2.plot(xs, dose.hill(xs, fa["bottom"], fa["top"], fa["logec50"], fa["h"]), c="#d1495b")
    ax2.axhline(R["act_thr"], ls="--", lw=0.8, c="#d1495b")
    ax2.set_ylabel("phenotypic activity", color="#d1495b")
    ax.set_title(f"{compound}: {row.triage}", fontsize=10)
    fig.tight_layout()

    known = g.moa.dropna().unique()
    tbl = g[["conc", "viability", "activity", "pred_moa", "pred_conf"]].copy()
    tbl["active"] = tbl.activity > R["act_thr"]
    tbl.loc[~tbl.active, "pred_moa"] = "(inactive - no call)"
    tbl = tbl.round(3)
    # explain at the lowest active dose (closest to a therapeutic dose)
    ia = g[g.activity > R["act_thr"]]
    expl = []
    if len(ia):
        r0 = ia.iloc[0]
        i0 = TM.index[(TM.compound == compound) & (TM.conc == r0.conc)][0]
        z = pd.Series(R["cls_all"][i0], index=NORMS["morph_cols"])
        expl = [f"At the lowest active dose ({r0.conc:g} uM):"] + [f"- {s}" for s in describe(z)]
    md = [f"**Annotated MoA (ground truth):** {', '.join(known) if len(known) else 'not annotated'}",
          f"**Predicted MoA** is made with this compound *held out* from the reference library.",
          f"**Lowest phenotypically active dose:** {row.first_active_conc:g} uM" if np.isfinite(row.first_active_conc) else "**No phenotypic activity in tested range**",
          f"**Lowest cytotoxic dose (viability < 50%):** {row.first_toxic_conc:g} uM" if np.isfinite(row.first_toxic_conc) else "**No overt cytotoxicity in tested range**",
          f"**Triage:** {row.triage}"] + expl
    gallery = [(thumb(compound, c), f"{c:g} uM") for c in g.conc if thumb(compound, c)]
    return fig, "\n\n".join(md), tbl, gallery


# --------------------------------------------------------------- tab 3
def atlas(color_by):
    d = TM[TM.compound != "DMSO"].copy()
    d["MoA (annotated)"] = d.moa.fillna("not annotated")
    d["MoA (predicted)"] = np.where(d.activity > R["act_thr"], d.pred_moa, "inactive")
    fig = px.scatter(d, x="umap_x", y="umap_y", color=color_by, hover_data=["compound", "conc", "viability", "pred_conf"],
                     height=620, color_discrete_sequence=px.colors.qualitative.Dark24)
    fig.update_traces(marker=dict(size=7, line=dict(width=0.3, color="black")))
    fig.update_layout(xaxis_visible=False, yaxis_visible=False, legend_title_text="")
    return fig


def bench_md():
    b = MET["benchmark"]
    rows = "\n".join(f"| {k} | {v['NSC']:.3f} | {v['NSCB']:.3f} |" for k, v in b.items())
    c = MET["conformal"]
    return (f"| Features | NSC | NSCB |\n|---|---|---|\n{rows}\n\n"
            f"**Conformal sets (target 90%)**: coverage {c['0.1']['coverage']:.2f}, mean size {c['0.1']['mean_set_size']:.2f}  \n"
            f"**Novel-MoA detection AUROC**: {MET['novelty_auroc']:.2f}  \n"
            f"**Percent replicating**: {MET['percent_replicating']:.2f}")


EX = sorted(glob.glob(os.path.join(ROOT, "app", "examples", "*")))
examples, example_labels = [], []
for d in EX:
    fs = sorted(glob.glob(os.path.join(d, "*.tif")))
    comp = os.path.basename(d).rsplit("_", 1)[0].replace("_", " ")
    if len(fs) == 3:
        examples.append(fs + [comp if comp in set(TMA.compound) else "(none)"])
        example_labels.append(os.path.basename(d).replace("_", " ").replace("uM", " uM"))

with gr.Blocks(title="Shennong's Glass") as demo:
    gr.Markdown("# Shennong's Glass\n*Shennong tasted a hundred herbs through a transparent body to tell medicine from poison. "
                "An organ-on-chip is that transparent body; this is the eye that watches it.*\n\n"
                "Foundation-model profiling of DAPI / beta-tubulin / F-actin images: mechanism of action, "
                "phenotypic potency, cytotoxicity window and calibrated uncertainty. Reference library: BBBC021 (MCF-7, 113 compounds).")
    with gr.Tab("1 - Analyze images"):
        with gr.Row():
            f1, f2, f3 = gr.File(label="DAPI (nuclei)"), gr.File(label="beta-tubulin"), gr.File(label="F-actin")
        hold = gr.Dropdown(["(none)"] + sorted(TMA.compound.unique()), value="(none)",
                           label="Blind test: remove this compound from the reference library")
        if examples:
            gr.Examples(examples, [f1, f2, f3, hold], example_labels=example_labels,
                        label="Example fields (the compound is held out from the library)")
        btn = gr.Button("Analyze", variant="primary")
        with gr.Row():
            ov = gr.Image(label="Segmentation (yellow = nuclei)", height=420)
            rep = gr.Markdown()
        probs = gr.Dataframe(label="Mechanism ranking")
        nn = gr.Gallery(label="Most similar reference treatments", columns=6, height=220)
        btn.click(analyze, [f1, f2, f3, hold], [ov, rep, probs, nn])
    with gr.Tab("2 - Compound explorer"):
        comps = sorted(DT.compound)
        dd = gr.Dropdown(comps, value="taxol" if "taxol" in comps else comps[0], label="Compound")
        with gr.Row():
            pl = gr.Plot()
            md = gr.Markdown()
        tb = gr.Dataframe(label="Dose-wise readout")
        gal = gr.Gallery(label="Representative field per dose", columns=8, height=200)
        dd.change(explore, dd, [pl, md, tb, gal])
        demo.load(explore, dd, [pl, md, tb, gal])
    with gr.Tab("3 - Phenotypic atlas"):
        cb = gr.Radio(["MoA (annotated)", "MoA (predicted)"], value="MoA (predicted)", label="Colour by")
        at = gr.Plot()
        cb.change(atlas, cb, at)
        demo.load(atlas, cb, at)
    with gr.Tab("4 - Benchmark & reliability"):
        gr.Markdown(bench_md())
        figs = [p for p in ["fig_benchmark.png", "fig_reliability.png", "fig_confusion.png", "fig_signatures.png", "fig_window.png", "fig_dose_examples.png"]
                if os.path.exists(os.path.join(RES, p))]
        gr.Gallery([os.path.join(RES, p) for p in figs], columns=2, height="auto")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", allowed_paths=[ART, RES, os.path.join(ROOT, "app")])
