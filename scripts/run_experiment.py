"""Run every configuration on a folder of images with binary GT.

Usage:
  python scripts/run_experiment.py --images data/synthetic/img --gt data/synthetic/gt \
      --config configs/baseline.yaml --out results/smoke

Writes:
  per_image.csv      one row per (image, configuration): metrics, time, radius, stroke width
  summary.csv        mean/median per configuration
  config_used.yaml   exact parameters, package versions, git-free run record
  binarized/...      every output binary image (DIBCO polarity)
"""
import argparse
import importlib.metadata as md
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from docbin import io, pipeline  # noqa: E402
from docbin.metrics import dibco_metrics  # noqa: E402


def configurations(cfg):
    for thr, p in cfg["thresholds"].items():
        yield dict(correction=None, sizing=None, thr=thr, thr_params=p or {})
        for c in cfg["corrections"]:
            yield dict(correction=c, sizing="fixed", thr=thr, thr_params=p or {},
                       fixed_radius=cfg["fixed_radius"][c])
            yield dict(correction=c, sizing="auto", thr=thr, thr_params=p or {},
                       auto_factor=cfg["auto_factor"][c])


def config_name(c):
    return f"{c['correction'] or 'none'}-{c['sizing'] or 'na'}-{c['thr']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    rows = []
    for img_path in sorted(Path(a.images).iterdir()):
        gt_path = next(Path(a.gt).glob(img_path.stem + "*"), None)
        if gt_path is None:
            print("no GT for", img_path.name); continue
        gray, gt = io.load_gray(img_path), io.load_binary_gt(gt_path)
        weights = __import__("doxapy").generate_pseudo_weights(gt)  # once per GT
        for c in configurations(cfg):
            res, t = pipeline.timed(pipeline.run, gray, c["correction"], c["sizing"], c["thr"],
                                    c["thr_params"], fixed_radius=c.get("fixed_radius"),
                                    auto_factor=c.get("auto_factor"), repeats=cfg["timing_repeats"])
            binary = io.text_mask_to_image(res["mask"])
            io.save(out / "binarized" / config_name(c) / f"{img_path.stem}.png", binary)
            m = dibco_metrics(gt, binary, weights)
            rows.append({"image": img_path.stem, "config": config_name(c),
                         "correction": res["correction"], "sizing": res["sizing"],
                         "threshold": res["threshold"], "radius": res.get("radius"),
                         "stroke_width": res.get("stroke_width"), "time_s": t, **m})
        print("done", img_path.name)

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_image.csv", index=False)
    # PSNR is infinite when an output has zero wrong pixels (10*log10(N/0)).
    # Per-image CSV keeps inf; the summary averages PSNR over finite values
    # only and reports how many images were perfect.
    df["psnr_perfect"] = np.isinf(df["psnr"])
    tmp = df.assign(psnr=df["psnr"].replace(np.inf, np.nan))
    keys = ["correction", "sizing", "threshold"]
    summ = tmp.groupby(keys)[["fm", "pseudo_fm", "psnr", "drd", "time_s"]].agg(["mean", "median"]).round(3)
    summ[("psnr", "n_perfect")] = df.groupby(keys)["psnr_perfect"].sum()
    summ.to_csv(out / "summary.csv")
    record = {"config": cfg, "args": vars(a), "python": platform.python_version(),
              "platform": platform.platform(),
              "packages": {p: md.version(p) for p in
                           ["numpy", "scipy", "scikit-image", "opencv-python-headless", "doxapy"]}}
    yaml.safe_dump(record, open(out / "config_used.yaml", "w"), sort_keys=False)
    print(summ.to_string())


if __name__ == "__main__":
    main()
