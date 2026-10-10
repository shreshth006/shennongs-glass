"""Fetch a few raw BBBC021 fields (one per example treatment) for the demo.

Images are copyright AstraZeneca and are not redistributed in this repo; they
are pulled from the public Kaggle mirror of BBBC021 (stanleyhua/bbbc021).
Only fields from the *last* replicate plate are used, so they can be shown
as "unseen" examples (the reference library still contains other plates).
"""
import os
import subprocess
import zipfile

import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
EX = {"DMSO": 0.0, "taxol": 3.0, "colchicine": 1.0, "cytochalasin B": 10.0, "etoposide": 10.0, "emetine": 1.0}
im = pd.read_csv(os.path.join(ROOT, "data/bbbc021/BBBC021_v1_image.csv"))
for comp, conc in EX.items():
    g = im[(im.Image_Metadata_Compound == comp) & (im.Image_Metadata_Concentration == conc)]
    if g.empty:
        continue
    r = g.sort_values("Image_Metadata_Plate_DAPI").iloc[-1]
    out = os.path.join(ROOT, "app/examples", f"{comp.replace(' ', '_')}_{conc:g}uM")
    os.makedirs(out, exist_ok=True)
    plate = r.Image_PathName_DAPI.split("/")[-1]
    for i, ch in enumerate(("DAPI", "Tubulin", "Actin")):
        dst = os.path.join(out, f"{i + 1}_{ch}.tif")
        if os.path.exists(dst):
            continue
        f = f"BBBC021_v1_images_{plate}/{plate}/{r['Image_FileName_' + ch]}"
        subprocess.run(["kaggle", "datasets", "download", "stanleyhua/bbbc021", "-f", f, "-p", out, "-q"], check=True)
        z = os.path.join(out, os.path.basename(f) + ".zip")
        src = os.path.join(out, os.path.basename(f))
        if os.path.exists(z):
            zipfile.ZipFile(z).extractall(out)
            os.remove(z)
        os.rename(src, dst)
    print("ok", out)
