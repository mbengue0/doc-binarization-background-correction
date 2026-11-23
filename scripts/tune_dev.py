"""Grid search on the development images (see TUNING_PROTOCOL.md).

For speed, the corrected image is computed once per (image, method, radius)
and every thresholding setting is applied to it; Sauvola/Niblack share one
local mean/std computation per window (identical formulas to scikit-image,
checked in tests/test_metrics.py::test_fast_local_thresholds_match_skimage).

Output: results/dev_tuning/scores.csv (one row per image x configuration)
        results/dev_tuning/stroke_widths.csv
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import doxapy
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from docbin import background, io, stroke  # noqa: E402
from docbin.threshold import local_mean_std  # noqa: E402

SAUVOLA = [(w, k) for w in (15, 31, 51, 101, 151, 201) for k in (0.1, 0.2, 0.3, 0.5)]
NIBLACK = [(w, k) for w in (15, 31, 51, 101, 151, 201) for k in (0.2, 0.4, 0.6, 1.0, 1.5, 2.0)]
FIXED_R = [3, 5, 8, 12, 16, 20, 25, 30, 40]
AUTO_F = [1, 1.5, 2, 3, 4, 5, 6, 8]
METHODS = ["closing", "median", "rolling_ball"]
DATA = Path("data/dibco")


def score(gt, mask):
    r = doxapy.calculate_performance(gt, io.text_mask_to_image(mask))
    psnr = float("inf") if r["psnr"] >= 1e300 else r["psnr"]
    return r["fm"], psnr, r["drdm"]


def all_thresholds(img):
    """Yield (threshold name, w, k, mask) for every grid setting."""
    from skimage.filters import threshold_otsu
    yield "otsu", None, None, img <= threshold_otsu(img)
    f = img.astype(np.float64)
    for w in sorted({w for w, _ in SAUVOLA} | {w for w, _ in NIBLACK}):
        m, s = local_mean_std(f, w)
        for ww, k in SAUVOLA:
            if ww == w:
                yield "sauvola", w, k, f <= m * (1 + k * (s / 128.0 - 1))
        for ww, k in NIBLACK:
            if ww == w:
                yield "niblack", w, k, f <= m - k * s


def process(row):
    year, doc = str(row["year"]), str(row["id"])
    g = io.load_gray(DATA / year / "img" / f"{doc}.png")
    gt = io.load_binary_gt(DATA / year / "gt" / f"{doc}.png")
    sw = stroke.estimate_stroke_width(g)
    out = []
    base = dict(year=year, image=doc, type=row["type"])
    for thr, w, k, mask in all_thresholds(g):
        out.append({**base, "correction": "none", "sizing": "-", "scale": np.nan, "radius": np.nan,
                    "threshold": thr, "w": w, "k": k, **dict(zip(("fm", "psnr", "drd"), score(gt, mask)))})
    # which radii are needed, and which (sizing, scale) pairs map to each radius
    uses = {}
    for r in FIXED_R:
        uses.setdefault(r, []).append(("fixed", r))
    for fct in AUTO_F:
        uses.setdefault(background.auto_radius(sw["stroke_width"], fct), []).append(("auto", fct))
    for meth in METHODS:
        for r, labels in sorted(uses.items()):
            corr, _ = background.correct(g, meth, r)
            for thr, w, k, mask in all_thresholds(corr):
                fm, psnr, drd = score(gt, mask)
                for sizing, scale in labels:
                    out.append({**base, "correction": meth, "sizing": sizing, "scale": scale, "radius": r,
                                "threshold": thr, "w": w, "k": k, "fm": fm, "psnr": psnr, "drd": drd})
    return out, {**base, **sw}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/dev_tuning")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="debug: first N images only")
    a = ap.parse_args()
    man = pd.read_csv(DATA / "manifest.csv")
    dev = man[man.split == "dev"]
    assert (dev.year != 2017).all()
    rows = dev.to_dict("records")[: a.limit]
    # largest images first so the pool finishes evenly
    rows.sort(key=lambda r: -r["height"] * r["width"])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    allrows, sws = [], []
    if (out / "scores.csv.gz").exists():  # resume: keep finished images, skip them
        prev = pd.read_csv(out / "scores.csv.gz", dtype={"image": str, "year": str})
        prev_sw = pd.read_csv(out / "stroke_widths.csv", dtype={"image": str, "year": str})
        done = set(zip(prev_sw.year, prev_sw.image))
        allrows, sws = prev.to_dict("records"), prev_sw.to_dict("records")
        rows = [r for r in rows if (str(r["year"]), str(r["id"])) not in done]
        print(f"resuming: {len(done)} images already done", flush=True)
    with Pool(a.workers) as p:
        for i, (res, sw) in enumerate(p.imap_unordered(process, rows), 1):
            allrows += res; sws.append(sw)
            print(f"{i}/{len(rows)} {sw['year']}/{sw['image']} sw={sw['stroke_width']:.2f}", flush=True)
            if i % 10 == 0 or i == len(rows):
                pd.DataFrame(allrows).to_csv(out / "scores.csv.gz", index=False)
                pd.DataFrame(sws).to_csv(out / "stroke_widths.csv", index=False)
