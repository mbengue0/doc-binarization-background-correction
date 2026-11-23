# Simple background correction for document binarization under uneven lighting

Research question: does choosing the background-correction kernel size
automatically from the estimated text stroke width improve binarization compared
with a fixed kernel size?

Status: **development tuning done (DIBCO 2009-2016); final test on DIBCO 2017 not yet run;
SD7K not yet started.** Every methodological decision and change is logged in
[`TUNING_PROTOCOL.md`](TUNING_PROTOCOL.md). DIBCO 2017 has not been used for any choice.

Datasets are not stored in this repository; `scripts/prepare_dibco.py` downloads
them from the official DIBCO sites and rebuilds `data/dibco/` (checked against
`data/dibco/manifest.csv`).

## Layout

```
configs/baseline.yaml      provisional parameters (to be tuned on dev data, then frozen)
src/docbin/
  io.py                    loading, polarity (text = 0, background = 255)
  stroke.py                stroke-width estimate (Sauvola mask -> distance transform -> skeleton)
  background.py            closing / median / rolling-ball background + flat-field division
  threshold.py             Otsu / Sauvola / Niblack (scikit-image)
  metrics.py               DIBCO metrics (doxapy) + PSNR/SSIM for SD7K
  stats.py                 paired Wilcoxon signed-rank test (document-level)
  pipeline.py              one run + timing protocol
scripts/
  make_synthetic.py        synthetic shadowed pages, used ONLY for the smoke test
  run_experiment.py        all configurations on a folder -> per_image.csv, summary.csv, config_used.yaml
  make_panel.py            visual inspection figure
tests/test_metrics.py      metric check against the official DIBCO example, plus stroke-width and Wilcoxon tests
third_party/DIBCO_metrics.m  official MATLAB metric code (reference only)
data/                      put datasets here (see below)
results/                   outputs
```

## Setup

```
pip install -r requirements.txt
PYTHONPATH=src python -m pytest tests        # must pass before any experiment
python scripts/make_synthetic.py
python scripts/run_experiment.py --images data/synthetic/img --gt data/synthetic/gt \
    --config configs/baseline.yaml --out results/smoke_synthetic
```

## Data

DIBCO (done): `python scripts/prepare_dibco.py --download raw_downloads --extracted dibco_extracted --out data/dibco`
downloads the official archives, organises them and writes `data/dibco/manifest.csv`
(86 development images from 2009-2016, 20 test images from 2017).

Sources:

* DIBCO 2017 (final test set): https://vc.ee.duth.gr/dibco2017/ -> `data/dibco/2017/{img,gt}/`
* Earlier DIBCO / H-DIBCO sets (proposed development data; links are listed on the
  DIBCO 2017 page) -> `data/dibco/<year>/{img,gt}/`
* SD7K: https://github.com/CXH-Research/DocShadow-SD7K (OneDrive / OpenXLab links in the
  README). Only the predefined subset goes in `data/sd7k/`; the subset manifest will be saved
  as `data/sd7k/manifest.csv`.

GT files must have the same file name stem as their input image (for example
`3.bmp` and `3_gt.bmp`).

## Conventions verified so far

* Metric code (doxapy 0.9.9) reproduces the official DIBCO tool's example exactly:
  FM 66.214, pseudo-FM 71.4372, PSNR 14.5629, DRD 13.8647.
* Polarity: text = 0 (black), background = 255 (white), as in the official code.
* scikit-image Sauvola takes R from the dtype (R = 1 for floats), so we pass R = 128.
* DIBCO 2017: pseudo-F-measure computed with doxapy's generated weights is identical
  (max difference 0.0000 over all 20 images) to pseudo-F-measure with the official
  EvalWeights .dat files.
* doxapy reports a perfect PSNR as DBL_MAX; we convert it to inf. Summaries average
  PSNR over finite values and count perfect images separately.
* Timing: perf_counter around processing only, 1 warm-up run, median of 3 runs.

## Development tuning (reproduce)

```
python scripts/validate_stroke.py          # stroke-width estimator vs GT (dev only)
python scripts/tune_dev.py                 # ~1.5 h on 2 cores -> results/dev_tuning/scores.csv.gz
python scripts/select_params.py            # applies TUNING_PROTOCOL.md rules -> configs/frozen.yaml
```

## Third-party material

`third_party/DIBCO_metrics.m` and `tests/5.bmp`, `tests/5_gt.bmp` come from
https://github.com/kzagoris/DibcoEvaluation (MIT licence, see `third_party/LICENSE_DibcoEvaluation_MIT`).
