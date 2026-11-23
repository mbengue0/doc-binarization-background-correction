"""Evaluation metrics.

DIBCO metrics come from doxapy, whose output was checked against the official
DIBCO evaluation tool's published example (tests/test_metrics.py):
FM 66.214, pseudo-FM 71.4372, PSNR 14.5629, DRD 13.8647.

Both inputs must be uint8 with text = 0 and background = 255.
"""
import doxapy
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def dibco_metrics(gt: np.ndarray, binary: np.ndarray, weights=None) -> dict:
    if gt.shape != binary.shape:
        raise ValueError(f"shape mismatch {gt.shape} vs {binary.shape}")
    for name, a in (("gt", gt), ("binary", binary)):
        if a.dtype != np.uint8 or not set(np.unique(a)) <= {0, 255}:
            raise ValueError(f"{name} must be uint8 with values in {{0,255}}")
    if weights is None:
        weights = doxapy.generate_pseudo_weights(gt)  # (precision, recall)
    pw, rw = weights
    r = doxapy.calculate_performance(gt, binary, pw, rw)
    # doxapy reports a perfect result (0 wrong pixels) as DBL_MAX, not inf.
    psnr = float("inf") if r["psnr"] >= 1e300 else r["psnr"]
    return {"fm": r["fm"], "pseudo_fm": r["pseudo_fm"], "psnr": psnr, "drd": r["drdm"],
            "recall": r["recall"], "precision": r["precision"]}


def restoration_metrics(pred: np.ndarray, ref: np.ndarray) -> dict:
    """Grayscale PSNR/SSIM between a corrected image and a shadow-free reference.

    For SD7K. Whether to apply a photometric alignment first (the corrected
    image is normalised to a white page; the shadow-free photo keeps the paper
    tone) is an OPEN DECISION, to be fixed after inspecting the data.
    """
    return {"psnr": float(peak_signal_noise_ratio(ref, pred, data_range=255)),
            "ssim": float(structural_similarity(ref, pred, data_range=255))}
