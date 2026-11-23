"""Background estimation and correction.

Each method estimates a "text-free" background image B and the corrected
image is the flat-field ratio  C = clip(255 * I / B).  Where the page is in
shadow, both I and B are dark, so the ratio comes back close to white; text
stays darker than its local background, so it survives.

Each method has its OWN scale parameter. They are related to stroke width,
but they do not mean the same thing:

  closing     : `radius` = radius (px) of a disk structuring element.
                Grayscale closing removes dark structures that the disk cannot
                fit into, so the disk DIAMETER (2*radius+1) must exceed the
                stroke width for text to be erased from B.
  median      : `radius` -> square window of side 2*radius+1. A dark pixel is
                removed from B only if text occupies < half of the window, so
                the window must be well over twice the stroke width in dense
                text.
  rolling_ball: `radius` = radius (px) of the ball rolled UNDER the inverted
                intensity surface (skimage.restoration.rolling_ball). Text
                strokes thinner than the ball are excluded from B.

For the automatic rule all three use radius = ceil(factor * stroke_width),
with `factor` chosen per method on development data only.
"""
import math

import cv2
import numpy as np
from skimage.restoration import rolling_ball


def bg_closing(gray: np.ndarray, radius: int) -> np.ndarray:
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.morphologyEx(gray, cv2.MORPH_CLOSE, se, borderType=cv2.BORDER_REPLICATE)


def bg_median(gray: np.ndarray, radius: int) -> np.ndarray:
    k = 2 * radius + 1
    # cv2.medianBlur only supports k <= 5 for non-uint8 / needs odd k; uint8 is fine for any odd k.
    return cv2.medianBlur(gray, k)


def bg_rolling_ball(gray: np.ndarray, radius: int) -> np.ndarray:
    # skimage's rolling ball estimates a DARK background under BRIGHT objects.
    # Our text is dark on bright paper, so invert, estimate, invert back.
    inv = 255 - gray
    bg_inv = rolling_ball(inv, radius=radius)
    return (255 - bg_inv).astype(np.uint8)


METHODS = {"closing": bg_closing, "median": bg_median, "rolling_ball": bg_rolling_ball}


def correct(gray: np.ndarray, method: str, radius: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (corrected image, background estimate)."""
    bg = METHODS[method](gray, int(radius))
    c = 255.0 * gray.astype(np.float64) / np.maximum(bg.astype(np.float64), 1.0)
    return np.clip(c, 0, 255).astype(np.uint8), bg


def auto_radius(stroke_width: float, factor: float, min_radius: int = 1) -> int:
    return max(min_radius, int(math.ceil(factor * stroke_width)))
