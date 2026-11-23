"""Otsu, Sauvola and Niblack thresholding (scikit-image implementations).

All return a boolean TEXT mask (True = text): text = gray <= T, i.e. the
complement of scikit-image's convention 'foreground = image > T'.

  Otsu    : one global threshold that best separates two intensity groups.
  Sauvola : local T = m * (1 + k * (s / R - 1)), m/s = local mean/std in a
            window, R = 128 (set explicitly; see sauvola()).
  Niblack : local T = m - k * s  (skimage convention; positive k lowers T).
"""
import numpy as np
from skimage.filters import threshold_niblack, threshold_otsu, threshold_sauvola
from skimage.filters.thresholding import _mean_std


def local_mean_std(gray_float, window: int):
    """Local mean and std exactly as scikit-image's Sauvola/Niblack compute them
    (private helper; equality with the public functions is unit-tested)."""
    return _mean_std(gray_float, window)


def otsu(gray, **_):
    return gray <= threshold_otsu(gray)


def sauvola(gray, window: int = 25, k: float = 0.2, r: float = 128.0, **_):
    # r passed explicitly: skimage otherwise derives R from the dtype, and for
    # float input that is 1.0, not 128 -> wrong thresholds on 0..255 data.
    return gray <= threshold_sauvola(gray.astype(np.float64), window_size=window, k=k, r=r)


def niblack(gray, window: int = 25, k: float = 0.2, **_):
    return gray <= threshold_niblack(gray.astype(np.float64), window_size=window, k=k)


METHODS = {"otsu": otsu, "sauvola": sauvola, "niblack": niblack}


def binarize(gray, method: str, **params):
    return METHODS[method](gray, **params)
