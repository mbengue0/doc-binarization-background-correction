"""One pipeline run = (optional background correction) -> thresholding.

Timing protocol: wall-clock time.perf_counter() around the processing only
(image already in memory, no disk I/O, no metric computation); the run is
repeated `repeats` times and the MEDIAN is reported, after one untimed
warm-up run.
"""
import time

import numpy as np

from . import background, stroke, threshold


def run(gray: np.ndarray, correction: str | None, sizing: str | None,
        thr: str, thr_params: dict, fixed_radius: int | None = None,
        auto_factor: float | None = None) -> dict:
    info = {"correction": correction or "none", "sizing": sizing or "-", "threshold": thr}
    img = gray
    if correction:
        if sizing == "fixed":
            radius = int(fixed_radius)
        elif sizing == "auto":
            sw = stroke.estimate_stroke_width(gray)
            info.update(stroke_width=sw["stroke_width"], sw_fallback=sw["used_fallback"])
            radius = background.auto_radius(sw["stroke_width"], auto_factor)
        else:
            raise ValueError(sizing)
        info["radius"] = radius
        img, _ = background.correct(gray, correction, radius)
    mask = threshold.binarize(img, thr, **thr_params)
    return {"mask": mask, "corrected": img, **info}


def timed(fn, *args, repeats: int = 3, **kwargs):
    fn(*args, **kwargs)  # warm-up
    ts, out = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        ts.append(time.perf_counter() - t0)
    return out, float(np.median(ts))
