"""Check our metric wrapper against the official DIBCO tool's published example.

Files 5.bmp / 5_gt.bmp and the expected numbers come from
https://github.com/kzagoris/DibcoEvaluation (readme, TestResources/).
"""
from pathlib import Path

import numpy as np
import pytest

from docbin.io import load_binary_gt
from docbin.metrics import dibco_metrics
from docbin.stats import paired_wilcoxon
from docbin.stroke import estimate_stroke_width

HERE = Path(__file__).parent


def test_matches_official_dibco_example():
    gt = load_binary_gt(HERE / "5_gt.bmp")
    b = load_binary_gt(HERE / "5.bmp")
    m = dibco_metrics(gt, b)
    assert m["fm"] == pytest.approx(66.214, abs=1e-3)
    assert m["pseudo_fm"] == pytest.approx(71.4372, abs=1e-3)
    assert m["psnr"] == pytest.approx(14.5629, abs=1e-3)
    assert m["drd"] == pytest.approx(13.8647, abs=1e-3)


def test_perfect_prediction():
    gt = load_binary_gt(HERE / "5_gt.bmp")
    m = dibco_metrics(gt, gt.copy())
    assert m["fm"] == pytest.approx(100) and m["drd"] == pytest.approx(0)


@pytest.mark.parametrize("w", [3, 6, 10])
def test_stroke_width_on_bars(w):
    img = np.full((200, 400), 230, np.uint8)
    for y in range(20, 180, 40):
        img[y:y + w, 30:370] = 40  # horizontal bars of known thickness
    est = estimate_stroke_width(img)["stroke_width"]
    assert abs(est - w) <= 1.5


def test_wilcoxon_counts_zeros():
    r = paired_wilcoxon([1, 2, 3, 4, 5, 6], [0, 2, 1, 3, 2, 1])
    assert r["n_zero_diff"] == 1 and r["n_docs"] == 6


def test_fast_local_thresholds_match_skimage():
    from skimage.filters import threshold_niblack, threshold_sauvola
    from docbin.io import load_gray
    from docbin.threshold import local_mean_std
    g = load_gray(HERE / "5_gt.bmp").astype(float)
    g = g * 0.5 + np.random.default_rng(0).normal(0, 10, g.shape)
    for w in (15, 51):
        m, s = local_mean_std(g, w)
        for k in (0.1, 0.5):
            assert np.array_equal(g <= m * (1 + k * (s / 128 - 1)), g <= threshold_sauvola(g, window_size=w, k=k, r=128))
            assert np.array_equal(g <= m - k * s, g <= threshold_niblack(g, window_size=w, k=k))


def test_otsu_keeps_text_on_two_level_image():
    from docbin.threshold import otsu
    img = np.full((50, 50), 255, np.uint8); img[20:25, 5:45] = 44
    assert otsu(img).sum() == 5 * 40
