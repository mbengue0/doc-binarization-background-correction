"""Estimate text stroke width (line thickness) from an input image.

Simplified rule (deliberately simpler than the SWT-based estimator in
Xiong et al., EURASIP JIVP 2021). Validated against GT-measured stroke width
on development data; see TUNING_PROTOCOL.md, Amendment 1.

  1. Flatten slow illumination changes: divide the image by a median-filter
     background (fixed radius 30 px), then make a provisional text mask with
     Otsu.
  2. Remove tiny specks (<= 10 px) from that mask.
  3. Distance transform: for every text pixel, the distance to the nearest
     background pixel.
  4. Skeletonize the mask. On the skeleton (the centre line of each stroke)
     the distance value is about half the stroke thickness.
  5. Stroke width = 2 * median(distance on skeleton pixels).

Small example: a horizontal bar 6 pixels thick. Its centre line lies 3 pixels
from either edge, so the distance there is ~3 and the estimate is 2*3 = 6.

Note: this is STROKE WIDTH, not letter size. An "o" 30 px tall drawn with a
4 px pen has letter size ~30 and stroke width ~4.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, skeletonize

from .background import correct


def estimate_stroke_width(gray: np.ndarray, pre_radius: int = 30, min_object_px: int = 10,
                          fallback: float = 3.0) -> dict:
    """Return dict with the estimate and diagnostics (for logging)."""
    flat, _ = correct(gray, "median", pre_radius)
    text = flat <= threshold_otsu(flat)
    text = remove_small_objects(text, max_size=min_object_px)
    if text.sum() == 0:
        return {"stroke_width": fallback, "n_skeleton_px": 0, "used_fallback": True}
    dist = ndi.distance_transform_edt(text)
    skel = skeletonize(text)
    vals = dist[skel]
    if vals.size == 0:
        return {"stroke_width": fallback, "n_skeleton_px": 0, "used_fallback": True}
    sw = float(2.0 * np.median(vals))
    return {"stroke_width": sw, "n_skeleton_px": int(vals.size), "used_fallback": False}
