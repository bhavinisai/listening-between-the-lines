"""Benjamini-Hochberg FDR correction, shared by the v2 stat_analysis scripts
(statsmodels is not installed in the convokit environment)."""

import numpy as np


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values, in the input order."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = p[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out
