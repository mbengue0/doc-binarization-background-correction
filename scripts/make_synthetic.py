"""Make a few SYNTHETIC shadowed text pages with exact ground truth.

Purpose: a smoke test that the code runs end to end. These images are NOT
part of the experiment and no result from them goes in the paper.

Each page: printed-looking text (OpenCV Hershey font) on a light page, then
multiplied by an illumination field = smooth gradient * soft dark shadow blob,
plus mild Gaussian noise. The GT is the text mask before degradation.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

WORDS = ("document binarization background shadow stroke width kernel threshold "
         "morphology median rolling ball illumination text page paper ink").split()


def make_page(rng, h=600, w=900, thickness=2, scale=0.9):
    page = np.full((h, w), 225, np.uint8)
    mask = np.zeros((h, w), np.uint8)
    y = 50
    while y < h - 30:
        line = " ".join(rng.choice(WORDS, size=6))
        cv2.putText(mask, line, (30, y), cv2.FONT_HERSHEY_SIMPLEX, scale, 255, thickness, cv2.LINE_AA)
        y += int(40 * scale + 10)
    is_text = mask > 127
    page[is_text] = 45
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    yy, xx = yy / h, xx / w
    grad = 1.0 - 0.35 * xx  # light falls off to the right
    cy, cx, r = rng.uniform(0.3, 0.7), rng.uniform(0.2, 0.6), rng.uniform(0.15, 0.3)
    blob = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r ** 2)))
    illum = grad * (1.0 - 0.55 * blob)
    img = page.astype(np.float64) * illum + rng.normal(0, 4, (h, w))
    gt = np.where(is_text, 0, 255).astype(np.uint8)
    return np.clip(img, 0, 255).astype(np.uint8), gt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out); (out / "img").mkdir(parents=True, exist_ok=True); (out / "gt").mkdir(exist_ok=True)
    for i in range(a.n):
        th = [1, 2, 3, 4][i % 4]  # vary stroke width across pages
        img, gt = make_page(rng, thickness=th, scale=0.6 + 0.25 * th)
        cv2.imwrite(str(out / "img" / f"syn{i:02d}.png"), img)
        cv2.imwrite(str(out / "gt" / f"syn{i:02d}.png"), gt)
    print(f"wrote {a.n} synthetic pages to {out}")
