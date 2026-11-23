# Development tuning protocol (written before running the tuning)

Data: the 86 development images (DIBCO 2009, H-DIBCO 2010, DIBCO 2011,
H-DIBCO 2012, DIBCO 2013, H-DIBCO 2014, H-DIBCO 2016). DIBCO 2017 is not
read during tuning.

Selection score: mean F-measure over the 86 images (each image counts once).
Ties (difference < 0.01 FM) go to the smaller window / radius / factor (cheaper).

## Grids

| Item | Values |
|---|---|
| Sauvola window w | 15, 31, 51, 101 |
| Sauvola k | 0.1, 0.2, 0.3, 0.5 (R = 128) |
| Niblack window w | 15, 31, 51, 101 |
| Niblack k | 0.2, 0.4, 0.6, 1.0 (T = mean - k * std) |
| Otsu | no parameters |
| Fixed radius r (all 3 corrections) | 3, 5, 8, 12, 16, 20, 25, 30, 40 |
| Automatic factor f (radius = ceil(f * stroke width)) | 1, 1.5, 2, 3, 4, 5 |

Radius meaning per method is in `src/docbin/background.py`.

## Rules

1. **Thresholding alone.** For Sauvola and Niblack choose (w, k) with the best
   mean FM on the uncorrected images.
2. **Correction + thresholding** (9 combinations). The threshold settings θ must
   be the same in the fixed and automatic arms (other settings held constant).
   For each θ compute best_fixed(θ) = max over r and best_auto(θ) = max over f.
   Choose θ* maximising (best_fixed(θ) + best_auto(θ)) / 2, i.e. settings that
   suit both arms equally. Then r* = best r under θ*, f* = best f under θ*.
3. **Main comparison** (fixed in advance): closing + Sauvola, fixed r* vs
   automatic f*, F-measure on DIBCO 2017, paired Wilcoxon signed-rank test
   (two-sided, zero differences split with 'zsplit').
4. After selection the values are written to `configs/frozen.yaml` and not
   changed before the DIBCO 2017 run.

Only FM, PSNR and DRD are computed during tuning (pseudo-FM needs per-image
weights and is ~25x slower); all four DIBCO metrics are computed on the test set.

---

## Amendment 1 (2026-11-22, after development round 1, before any test-set use)

Round 1 (scores kept in `results/dev_tuning_round1/`) revealed two problems on
development data. DIBCO 2017 had not been read.

1. **Stroke-width estimator underestimated on noisy pages** (e.g. 2012/H11:
   estimate 2 px, strokes ~6 px). Estimator variants were compared against the
   stroke width measured with the same distance/skeleton rule on the GT text
   masks of the 86 development images (`scripts/validate_stroke.py`,
   `results/dev_tuning/stroke_validation.csv`):

   | variant | mean abs. error | within 1 px | Spearman |
   |---|---|---|---|
   | round-1 rule: Sauvola w25 k0.2 mask | 1.19 px | 48% | 0.59 |
   | Otsu mask | 0.85 px | 66% | 0.81 |
   | **median-filter correction (r = 30) then Otsu mask** | **0.74 px** | 59% | **0.83** |

   New rule: the last row. Chosen over plain Otsu because a global threshold
   on an unevenly lit page turns shadows into large false "strokes"; the
   pre-correction removes slow illumination changes first. Note: the
   automatic rule therefore contains one fixed scale of its own (r = 30).

2. **Selected values at the edge of the grid** (Niblack w = 101, k = 1.0;
   median auto factor = 5; Sauvola w = 101). Grids extended:
   Sauvola/Niblack w add 151, 201; Niblack k add 1.5, 2.0; auto factor add 6, 8.

3. **Threshold comparison convention.** Text is now `gray <= T` (complement of
   scikit-image's `foreground = image > T`) for all three methods. With `<`,
   Otsu removed all text on a two-level image (threshold equals the text value).
   Effect on real scans is negligible but conventions are now consistent.

Example after the fix: 2012/H11 estimate 4.5 px (GT-measured 6.3 px; was 2.0 px).

Selection rules (1-4 above) are unchanged.

## Round 2 outcome (development data only)

Selection written to `configs/frozen.yaml` (candidate, not yet used on test data).
Main comparison (closing + Sauvola, w = 101, k = 0.2): fixed r = 8 -> dev FM 84.08;
automatic f = 1.5 -> dev FM 84.16. All three values are inside the grid.

Grid edges still reached:
* median correction: auto factor 8 and fixed r = 40 (median+Niblack). Curves are flat
  there (factor 6 -> 8: +0.04 FM), so the edge does not matter.
* Niblack window w = 201 (all pipelines). FM still rises with w (151 -> 201: +2.0 FM
  without correction). Niblack is therefore likely under-tuned within this grid.

Stroke-width estimator after Amendment 1: mean absolute error 0.74 px against the
GT-measured width; it still tends to underestimate thick strokes (6-8 px), see
`results/dev_tuning/stroke_width_vs_gt.png`.
