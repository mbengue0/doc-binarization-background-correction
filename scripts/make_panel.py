"""Side-by-side panel for visual inspection: input | background | corrected | binary | GT."""
import argparse, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from docbin import io, background, stroke, threshold  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True); ap.add_argument("--gt")
ap.add_argument("--method", default="closing"); ap.add_argument("--factor", type=float, default=3.5)
ap.add_argument("--thr", default="otsu"); ap.add_argument("--out", required=True)
a = ap.parse_args()
g = io.load_gray(a.image)
sw = stroke.estimate_stroke_width(g)["stroke_width"]
r = background.auto_radius(sw, a.factor)
corr, bg = background.correct(g, a.method, r)
panels = [("input", g), (f"{a.thr} on input", io.text_mask_to_image(threshold.binarize(g, a.thr))),
          (f"background ({a.method}, r={r})", bg), ("corrected", corr),
          (f"{a.thr} on corrected", io.text_mask_to_image(threshold.binarize(corr, a.thr)))]
if a.gt: panels.append(("ground truth", io.load_binary_gt(a.gt)))
fig, ax = plt.subplots(2, 3, figsize=(15, 7))
for axi, (t, im) in zip(ax.flat, panels):
    axi.imshow(im, cmap="gray", vmin=0, vmax=255); axi.set_title(t); axi.axis("off")
for axi in ax.flat[len(panels):]: axi.axis("off")
fig.suptitle(f"{Path(a.image).name}: estimated stroke width {sw:.1f}px")
fig.tight_layout(); fig.savefig(a.out, dpi=90)
