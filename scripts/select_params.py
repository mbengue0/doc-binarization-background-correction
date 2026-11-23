"""Apply the selection rules in TUNING_PROTOCOL.md to the development scores.

Writes configs/frozen.yaml and results/dev_tuning/{selection.csv, fixed_vs_auto_curves.csv}.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

TIE = 0.01  # FM points


def pick(df, col, smaller_first):
    """Best row by mean FM; ties within TIE go to the smallest value(s) in smaller_first."""
    best = df[col].max()
    cand = df[df[col] >= best - TIE]
    return cand.sort_values(smaller_first).iloc[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/dev_tuning/scores.csv.gz")
    ap.add_argument("--out", default="configs/frozen.yaml")
    a = ap.parse_args()
    d = pd.read_csv(a.scores)
    d["w"] = d["w"].fillna(-1); d["k"] = d["k"].fillna(-1)  # Otsu has no params
    n_img = d.groupby(["year", "image"]).ngroups
    keys = ["correction", "sizing", "scale", "threshold", "w", "k"]
    mean = d.groupby(keys, dropna=False)["fm"].mean().reset_index()
    mean["n"] = d.groupby(keys, dropna=False)["fm"].count().values
    assert (mean["n"] == n_img).all(), "every configuration must cover every dev image"

    frozen = {"selected_on": f"{n_img} development images, mean F-measure", "baselines": {}, "corrected": {}}
    rows = []
    # Rule 1: thresholding alone
    for thr in ["otsu", "sauvola", "niblack"]:
        sub = mean[(mean.correction == "none") & (mean.threshold == thr)]
        b = pick(sub, "fm", ["w", "k"])
        frozen["baselines"][thr] = {"w": int(b.w), "k": float(b.k)} if thr != "otsu" else {}
        rows.append({"correction": "none", "threshold": thr, "w": b.w, "k": b.k, "dev_fm": b.fm})

    # Rule 2: correction + thresholding, theta shared by both arms
    for meth in ["closing", "median", "rolling_ball"]:
        for thr in ["otsu", "sauvola", "niblack"]:
            sub = mean[(mean.correction == meth) & (mean.threshold == thr)]
            per_theta = []
            for (w, k), g in sub.groupby(["w", "k"]):
                bf = g[g.sizing == "fixed"].fm.max(); ba = g[g.sizing == "auto"].fm.max()
                per_theta.append({"w": w, "k": k, "best_fixed": bf, "best_auto": ba, "avg": (bf + ba) / 2})
            pt = pd.DataFrame(per_theta)
            th = pick(pt, "avg", ["w", "k"])
            g = sub[(sub.w == th.w) & (sub.k == th.k)]
            rf = pick(g[g.sizing == "fixed"], "fm", ["scale"])
            ra = pick(g[g.sizing == "auto"], "fm", ["scale"])
            frozen["corrected"][f"{meth}+{thr}"] = {
                "threshold_params": ({"w": int(th.w), "k": float(th.k)} if thr != "otsu" else {}),
                "fixed_radius": int(rf.scale), "auto_factor": float(ra.scale)}
            rows.append({"correction": meth, "threshold": thr, "w": th.w, "k": th.k,
                         "fixed_radius": rf.scale, "dev_fm_fixed": rf.fm,
                         "auto_factor": ra.scale, "dev_fm_auto": ra.fm})
    frozen["main_comparison"] = {"pipeline": "closing+sauvola", "metric": "fm", "test_set": "DIBCO 2017",
                                 "test": "paired Wilcoxon signed-rank, two-sided, zero_method=zsplit"}
    sel = pd.DataFrame(rows)
    out_dir = Path(a.scores).parent
    sel.to_csv(out_dir / "selection.csv", index=False)
    # FM-vs-scale curves under the selected theta (for figures)
    curves = []
    for key, v in frozen["corrected"].items():
        meth, thr = key.split("+")
        tp = v["threshold_params"]
        g = mean[(mean.correction == meth) & (mean.threshold == thr) &
                 (mean.w == tp.get("w", -1)) & (mean.k == tp.get("k", -1))]
        curves.append(g.assign(pipeline=key))
    pd.concat(curves).to_csv(out_dir / "fixed_vs_auto_curves.csv", index=False)
    Path(a.out).write_text(yaml.safe_dump(frozen, sort_keys=False))
    print(sel.round(3).to_string())


if __name__ == "__main__":
    main()
