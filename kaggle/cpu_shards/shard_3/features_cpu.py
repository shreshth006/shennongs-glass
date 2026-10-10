"""BBBC021 feature extraction (runs as a Kaggle GPU notebook).

For every image set (DAPI / beta-tubulin / F-actin) this computes:
  * classical, interpretable per-cell morphology (nuclei segmentation + cell
    territories), aggregated per image (median / IQR) + cell count
  * DINOv2 embeddings of single-cell crops (per channel), averaged per image
  * DINOv2 embeddings of the whole field (per channel)
  * an RGB thumbnail for the demo app

Outputs land in /kaggle/working: meta.parquet, morph.parquet, emb_cell.npy,
emb_img.npy, thumbs/<ImageNumber>.jpg
"""
import glob
import os
import time
T0 = time.time()
print("start", flush=True)

import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage as ndi
from skimage import feature, filters, measure, morphology, segmentation

LIMIT = 0           # >0: only process this many image sets (smoke test)
MODEL = "dinov2_vits14"   # dinov2_vits14 for the CPU variant
EMB_CROP = 112      # resolution single-cell crops are embedded at
FIELD = (224, 280)  # resolution the whole field is embedded at (multiples of 14)
SITES = (1, 3)        # e.g. (1, 3): only these fields of view per well
SHARD, NSHARD = 3, 8
N_CELLS = 16        # single-cell crops per field
CROP = 64           # crop size at half resolution (~128 px full res, one cell)
HALF = 2            # downsample factor for segmentation / crops
BATCH = 8
OUT = os.environ.get("SHENNONG_OUT", "/kaggle/working")
LOCAL_ROOT = os.environ.get("SHENNONG_ROOT")        # run outside Kaggle on a local copy


def find_root():
    # shallow search only: the dataset holds ~40k files, a recursive glob stalls
    for d in ["/kaggle/input", *glob.glob("/kaggle/input/*"), *glob.glob("/kaggle/input/*/*"),
              *glob.glob("/kaggle/input/*/*/*")]:
        if os.path.exists(os.path.join(d, "BBBC021_v1_image.csv")):
            return d
    raise FileNotFoundError("BBBC021 dataset not attached: " + str(glob.glob("/kaggle/input/*")))


ROOT = LOCAL_ROOT or find_root()
meta = pd.read_csv(os.environ.get("SHENNONG_META", os.path.join(ROOT, "BBBC021_v1_image.csv")))
if LOCAL_ROOT:  # keep only image sets whose 3 channels are present locally
    _ok = lambda r: all(os.path.exists(os.path.join(ROOT, f"BBBC021_v1_images_{r[f'Image_PathName_{c}'].split('/')[-1]}",
                        r[f"Image_PathName_{c}"].split("/")[-1], r[f"Image_FileName_{c}"])) for c in ("DAPI", "Tubulin", "Actin"))
    meta = meta[meta.apply(_ok, axis=1)]
if SITES:
    meta = meta[meta.Image_FileName_DAPI.str.extract(r"_s(\d)_")[0].astype(int).isin(SITES).values]
meta = meta.iloc[SHARD::NSHARD]
if LIMIT:
    meta = meta.sample(LIMIT, random_state=0)
meta = meta.reset_index(drop=True)
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "root", ROOT, "image sets", len(meta), flush=True)


def path(row, ch):
    plate = row[f"Image_PathName_{ch}"].split("/")[-1]
    p = os.path.join(ROOT, f"BBBC021_v1_images_{plate}", plate, row[f"Image_FileName_{ch}"])
    if os.path.exists(p):
        return p
    # some mirrored files carry an extra "<Week>_<date>_" prefix
    hits = glob.glob(os.path.join(ROOT, f"BBBC021_v1_images_{plate}", plate, "*" + row[f"Image_FileName_{ch}"]))
    return hits[0]


def norm(img):
    lo, hi = np.percentile(img, (0.5, 99.8))
    return np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)


def segment(dapi, cyto):
    """Nuclei by Otsu + distance-transform watershed; cells by expanding nuclei
    into the cytoplasm foreground."""
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


NUC_PROPS = ["area", "perimeter", "eccentricity", "solidity", "major_axis_length", "minor_axis_length"]


def work(i):
        row = meta.iloc[i]
        chans = [norm(tifffile.imread(path(row, c))) for c in ("DAPI", "Tubulin", "Actin")]
        full = np.stack(chans)                                   # 3,H,W
        half = full[:, ::HALF, ::HALF]
        dapi, tub, act = half
        nuc, cell = segment(dapi, np.maximum(tub, act))

        # ---- classical morphology ----
        rows = []
        nprops = measure.regionprops(nuc, intensity_image=dapi)
        cprops = {p.label: p for p in measure.regionprops(cell, intensity_image=np.stack([tub, act], -1))}
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
        if len(cells):
            for k in cells.columns:
                v = cells[k].dropna()
                morph[f"{k}_med"] = float(v.median()) if len(v) else np.nan
                morph[f"{k}_iqr"] = float(v.quantile(0.75) - v.quantile(0.25)) if len(v) else np.nan
        for j, nm in enumerate(("dapi", "tub", "act")):
            morph[f"{nm}_fg_frac"] = float((half[j] > filters.threshold_otsu(half[j])).mean())
            morph[f"{nm}_texture"] = float(np.abs(filters.sobel(half[j])).mean())

        # ---- single-cell crops ----
        h = CROP // 2
        pad = np.pad(half, ((0, 0), (h, h), (h, h)))
        cents = [tuple(map(int, p.centroid)) for p in nprops if p.area >= 30]
        rng = np.random.default_rng(i)
        if len(cents) > N_CELLS:
            cents = [cents[k] for k in rng.choice(len(cents), N_CELLS, replace=False)]
        crops = np.zeros((N_CELLS, 3, CROP, CROP), np.float32)
        for k, (y, x) in enumerate(cents):
            crops[k] = pad[:, y:y + CROP, x:x + CROP]

        # ---- whole field, resized for the ViT (multiple of 14) ----
        field = np.stack([np.asarray(Image.fromarray(c).resize((FIELD[1], FIELD[0]), Image.BOX)) for c in full])

        # ---- thumbnail (R=actin, G=tubulin, B=DAPI) ----
        rgb = (np.stack([act, tub, dapi], -1) * 255).astype(np.uint8)
        Image.fromarray(rgb).resize((320, 256), Image.BILINEAR).save(
            f"{OUT}/thumbs/{int(row.ImageNumber)}_{int(row.TableNumber)}.jpg", quality=85)

        return i, morph, crops, len(cents), field


def load_model():
    try:
        m = torch.hub.load("facebookresearch/dinov2", MODEL, trust_repo=True)
        print("DINOv2 from torch.hub", flush=True)
    except Exception as e:  # offline fallback: Kaggle Models copy
        print("hub failed:", e, flush=True)
        import timm
        w = glob.glob("/kaggle/input/dinov2*/**/*.pth", recursive=True)
        m = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=False, img_size=518)
        m.load_state_dict(torch.load(w[0], map_location="cpu"), strict=False)
    return m.eval().to(DEV).to(DTYPE)


DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEV == "cuda" else torch.float32
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)


@torch.no_grad()
def embed(model, x, size=None):
    """x: (B,H,W) single-channel in [0,1] -> (B, 2*768) [CLS | mean patch]."""
    x = x.to(DEV).float()[:, None]
    if size:
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    x = ((x.repeat(1, 3, 1, 1) - MEAN) / STD).to(DTYPE)
    out = model.forward_features(x)
    return torch.cat([out["x_norm_clstoken"], out["x_norm_patchtokens"].mean(1)], 1).float().cpu()


def main():
    os.makedirs(f"{OUT}/thumbs", exist_ok=True)
    import multiprocessing as mp
    nw = os.cpu_count() if torch.cuda.is_available() else max(1, os.cpu_count() // 2)
    pool = mp.get_context("fork").Pool(nw)   # fork BEFORE any CUDA init
    print("pool up", nw, "workers", flush=True)
    if not torch.cuda.is_available():
        torch.set_num_threads(os.cpu_count())
    model = load_model()
    n = len(meta)
    results = pool.imap(work, range(n), chunksize=2)
    dl = (list(next(results) for _ in range(min(BATCH, n - s))) for s in range(0, n, BATCH))
    D = model.embed_dim
    emb_cell = np.zeros((n, 3 * D), np.float16)         # CLS of cell crops, mean over cells
    emb_img = np.zeros((n, 3 * 2 * D), np.float16)      # CLS|meanpatch of whole field
    morphs = [None] * n
    t0 = time.time()
    for bi, batch in enumerate(dl):
        idx = [b[0] for b in batch]
        for b in batch:
            morphs[b[0]] = b[1]
        crops = torch.from_numpy(np.stack([b[2] for b in batch]))            # B,N,3,C,C
        B = crops.shape[0]
        e = embed(model, crops.permute(0, 2, 1, 3, 4).reshape(-1, CROP, CROP), size=(EMB_CROP, EMB_CROP))
        e = e[:, :D].view(B, 3, N_CELLS, D)
        for k, b in enumerate(batch):
            nc = max(b[3], 1)
            emb_cell[b[0]] = e[k, :, :nc].mean(1).reshape(-1).numpy()
        fields = torch.from_numpy(np.stack([b[4] for b in batch]))           # B,3,H,W
        ef = embed(model, fields.reshape(-1, *FIELD)).view(B, 3 * 2 * D)
        emb_img[idx] = ef.numpy()
        if bi % 50 == 0:
            done = (bi + 1) * BATCH
            print(f"{done}/{n}  {time.time() - t0:.0f}s  eta {(time.time() - t0) / done * (n - done) / 60:.1f} min", flush=True)

    meta.to_parquet(f"{OUT}/meta.parquet")
    pd.DataFrame(morphs).to_parquet(f"{OUT}/morph.parquet")
    np.save(f"{OUT}/emb_cell.npy", emb_cell)
    np.save(f"{OUT}/emb_img.npy", emb_img)
    os.system(f"cd {OUT} && tar cf thumbs.tar thumbs && rm -rf thumbs")
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
