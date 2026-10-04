"""Small shared statistics helpers (no scipy dependency)."""
from __future__ import annotations

from statistics import NormalDist

import numpy as np


def mean_t(x: np.ndarray) -> float:
    """t-statistic for H0: mean = 0 (independent observations)."""
    x = np.asarray(x, dtype=float)
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def corr_t(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """t-statistic of the Pearson correlation of each column of X (T x k) with y (T,)."""
    X = np.asarray(X, dtype=float).reshape(len(y), -1)
    Xc, yc = X - X.mean(0), y - y.mean()
    r = (Xc * yc[:, None]).sum(0) / np.sqrt((Xc ** 2).sum(0) * (yc ** 2).sum())
    return r * np.sqrt((len(y) - 2) / (1 - r ** 2))


def bonferroni_crit(m: int, alpha: float = 0.05) -> float:
    """Two-sided critical |t| (normal approx.) after Bonferroni correction for m tests."""
    return NormalDist().inv_cdf(1 - alpha / (2 * max(1, int(m))))
