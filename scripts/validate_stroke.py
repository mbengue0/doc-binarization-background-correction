"""Compare stroke-width estimator variants with the stroke width measured on
the ground truth (same distance/skeleton rule applied to the GT text mask).
Development images only."""
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, threshold_sauvola
from skimage.morphology import skeletonize, remove_small_objects
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from docbin import io, background  # noqa

def sw_from_mask(t, q=50, min_obj=10):
    t = remove_small_objects(t, max_size=min_obj)
    if t.sum() == 0: return np.nan
    d = ndi.distance_transform_edt(t); v = d[skeletonize(t)]
    return 2 * np.percentile(v, q) if v.size else np.nan

def variants(g):
    f = g.astype(float)
    sv = lambda w, k: f <= threshold_sauvola(f, window_size=w, k=k, r=128)
    out = {}
    out["A_sauvola25_k0.2 (current)"] = sw_from_mask(sv(25, .2))
    out["B_sauvola51_k0.2"] = sw_from_mask(sv(51, .2))
    out["C_sauvola51_k0.4"] = sw_from_mask(sv(51, .4))
    out["D_sauvola25_k0.4"] = sw_from_mask(sv(25, .4))
    out["E_otsu"] = sw_from_mask(f <= threshold_otsu(g))
    corr, _ = background.correct(g, "median", 30)  # scale-independent pre-correction
    out["F_median30_then_otsu"] = sw_from_mask(corr <= threshold_otsu(corr))
    out["G_sauvola25_k0.2_q75"] = sw_from_mask(sv(25, .2), q=75)
    return out

def one(r):
    y, i = str(r["year"]), str(r["id"])
    g = io.load_gray(f"data/dibco/{y}/img/{i}.png"); gt = io.load_binary_gt(f"data/dibco/{y}/gt/{i}.png") == 0
    return {"year": y, "image": i, "gt_sw": sw_from_mask(gt, min_obj=0), **variants(g)}

if __name__ == "__main__":
    m = pd.read_csv("data/dibco/manifest.csv"); dev = m[m.split == "dev"]
    with Pool(2) as p: rows = p.map(one, dev.to_dict("records"))
    d = pd.DataFrame(rows); d.to_csv("results/dev_tuning/stroke_validation.csv", index=False)
    cols = [c for c in d if c[1] == "_"]
    summ = pd.DataFrame({c: {"median |error| px": (d[c] - d.gt_sw).abs().median(),
                             "mean |error| px": (d[c] - d.gt_sw).abs().mean(),
                             "within 1px (%)": 100 * ((d[c] - d.gt_sw).abs() <= 1).mean(),
                             "under by >2px (n)": int(((d.gt_sw - d[c]) > 2).sum()),
                             "spearman": d[c].corr(d.gt_sw, method="spearman")} for c in cols}).T
    print(d.gt_sw.describe().round(2)); print(summ.round(2).to_string())
