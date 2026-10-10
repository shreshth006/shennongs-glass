"""Write report/numbers.tex (LaTeX macros) from results/metrics.json so every
number in the report comes straight from the analysis run."""
import json
import os
import sys

import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
res = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results")
m = json.load(open(os.path.join(res, "metrics.json")))
dt = pd.read_csv(os.path.join(res, "compound_dose_table.csv"))
b = m["benchmark"]


def pct(x):
    return f"{100 * x:.1f}\\%"


cmds = {
    "NImages": f"{m['n_images']:,}", "NMorph": str(m.get("n_morph", 23)), "NWells": f"{m['n_wells']:,}",
    "NSCfusion": pct(b["fusion"]["NSC"]), "NSCBfusion": pct(b["fusion"]["NSCB"]),
    "NSCcell": pct(b["dino_cell"]["NSC"]), "NSCBcell": pct(b["dino_cell"]["NSCB"]),
    "NSCfield": pct(b["dino_field"]["NSC"]), "NSCBfield": pct(b["dino_field"]["NSCB"]),
    "NSCclassical": pct(b["classical"]["NSC"]), "NSCBclassical": pct(b["classical"]["NSCB"]),
    "NSCnotvn": pct(b["dino_cell_no_tvn"]["NSC"]), "NSCBnotvn": pct(b["dino_cell_no_tvn"]["NSCB"]),
    "NSCbalfusion": pct(b["fusion"]["NSC_balanced"]),
    "CovNinety": pct(m["conformal"]["0.1"]["coverage"]), "SizeNinety": f"{m['conformal']['0.1']['mean_set_size']:.2f}",
    "SingNinety": pct(m["conformal"]["0.1"]["singleton_rate"]),
    "CovNinetyFive": pct(m["conformal"]["0.05"]["coverage"]), "SizeNinetyFive": f"{m['conformal']['0.05']['mean_set_size']:.2f}",
    "CovEighty": pct(m["conformal"]["0.2"]["coverage"]), "SizeEighty": f"{m['conformal']['0.2']['mean_set_size']:.2f}",
    "SensFusMin": pct(m["sensitivity"]["fusion_NSC_min"]), "SensFusMax": pct(m["sensitivity"]["fusion_NSC_max"]),
    "SensNSCmin": pct(m["sensitivity"]["NSC_min"]), "SensNSCmax": pct(m["sensitivity"]["NSC_max"]),
    "SensNSCmed": pct(m["sensitivity"]["NSC_median"]), "SensNSCBmin": pct(m["sensitivity"]["NSCB_min"]),
    "SensNSCBmax": pct(m["sensitivity"]["NSCB_max"]), "SensNSCBmed": pct(m["sensitivity"]["NSCB_median"]),
    "NoveltyAUROC": f"{m['novelty_auroc']:.2f}", "PctReplicating": pct(m["percent_replicating"]), "PctReplicatingAnn": pct(m["percent_replicating_annotated"]),
    "NCompounds": str(m["dose"]["n_compounds"]),
    "NInactive": str(int((dt.triage == "inactive in tested range").sum())),
    "NToxOnly": str(int((dt.triage == "cytotoxic without a distinct phenotype").sum())),
    "NNonTox": str(int((dt.triage == "active, not cytotoxic in tested range").sum())),
    "NWideWindow": str(int((dt.triage == "active >=10x below cytotoxic dose").sum())),
    "NNarrow": str(int((dt.triage == "narrow window (<10x)").sum())),
    "NToxFirst": str(int((dt.triage == "phenotype only at cytotoxic doses").sum())),
}
os.makedirs(os.path.join(ROOT, "report"), exist_ok=True)
g = pd.read_csv(os.path.join(res, "sensitivity_grid.csv"))
order = ["cells", "field", "cells+field", "fusion (w=0.4)", "fusion (w=0.7)"]
rows = []
for prof in order:
    gg = g[g.profile == prof].set_index("tvn_components")
    cells = " & ".join(f"{100*gg.NSC[k]:.1f} / {100*gg.NSCB[k]:.1f}" for k in sorted(gg.index))
    name = prof + (" (main)" if prof == "fusion (w=0.7)" else "")
    rows.append(f"{name} & {cells} \\\\")
ks = sorted(g.tvn_components.unique())
with open(os.path.join(ROOT, "report", "sensitivity_table.tex"), "w") as fh:
    fh.write("\\begin{tabular}{l" + "c" * len(ks) + "}\n\\toprule\n")
    fh.write("Profile & " + " & ".join(f"TVN {k}" for k in ks) + " \\\\\n\\midrule\n")
    fh.write("\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
with open(os.path.join(ROOT, "report", "numbers.tex"), "w") as fh:
    for k, v in cmds.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(open(os.path.join(ROOT, "report", "numbers.tex")).read())
