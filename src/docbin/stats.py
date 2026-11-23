"""Paired Wilcoxon signed-rank test for the main comparison.

Input: one score per DOCUMENT for method A and method B (same documents, same
order). For SD7K, scores of several shadow versions of one document must be
averaged into one document-level score BEFORE calling this (repeated versions
are not independent).

Zero differences (A == B on a document) use the 'zsplit' convention by
default: each zero contributes half its rank to each side, so they are not
silently dropped. Ties among non-zero |differences| get average ranks
(SciPy default). With ties/zeros SciPy uses the normal approximation.
"""
import numpy as np
from scipy.stats import wilcoxon


def paired_wilcoxon(a, b, zero_method: str = "zsplit", alternative: str = "two-sided") -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    res = wilcoxon(a, b, zero_method=zero_method, alternative=alternative)
    return {"n_docs": int(d.size), "n_zero_diff": int(np.sum(d == 0)),
            "median_diff": float(np.median(d)), "mean_diff": float(np.mean(d)),
            "n_a_better": int(np.sum(d > 0)), "n_b_better": int(np.sum(d < 0)),
            "statistic": float(res.statistic), "p_value": float(res.pvalue),
            "zero_method": zero_method, "alternative": alternative}
