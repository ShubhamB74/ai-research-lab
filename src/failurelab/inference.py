"""Cluster bootstrap inference.

Unit of resampling = a clean/trapped PAIR (the two twins share one random world), with every rep and every
condition of that pair kept together. Treating runs as independent would give intervals that are far too
narrow. Metrics are ratios num/den summed over rows, so each cluster contributes two numbers and a bootstrap
draw is a matrix product (fast, vectorised).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

METRICS = ("recall", "false_flag_rate", "clean_accuracy", "trapped_accuracy", "fooled_rate",
           "rfwr_share", "brier_trapped", "brier_clean")


@dataclass(frozen=True)
class Est:
    point: float
    lo: float
    hi: float
    level: float

    def as_dict(self) -> dict:
        return {"point": self.point, "lo": self.lo, "hi": self.hi, "level": self.level}

    @property
    def defined(self) -> bool:
        return bool(np.isfinite(self.point) and np.isfinite(self.lo) and np.isfinite(self.hi))


def row_contributions(df: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-row (numerator, denominator) for every metric. Definitions match the README."""
    tr = (df["variant"] == "trapped").to_numpy()
    cl = ~tr
    correct = df["conclusion_correct"].to_numpy(bool)
    tp, fn, fp = (df[c].to_numpy(float) for c in ("tp", "fn", "fp"))
    f = lambda x: np.asarray(x, dtype=float)
    where = lambda x, m: np.where(m, f(x), 0.0)
    return {
        "recall": (where(tp, tr), where(tp + fn, tr)),
        "false_flag_rate": (where(fp > 0, cl), f(cl)),
        "clean_accuracy": (where(correct, cl), f(cl)),
        "trapped_accuracy": (where(correct, tr), f(tr)),
        "fooled_rate": (where(df["fooled"].to_numpy(bool), tr), f(tr)),
        "rfwr_share": (where(df["right_for_wrong_reason"].to_numpy(bool), tr), f(tr & correct)),
        "brier_trapped": (where(df["brier"].to_numpy(float), tr), f(tr)),
        "brier_clean": (where(df["brier"].to_numpy(float), cl), f(cl)),
    }


class Study:
    """Holds the cluster index and ONE fixed set of bootstrap draws, so that every estimate and every
    paired contrast in an analysis uses common random numbers (reproducible, and contrasts are coherent)."""

    def __init__(self, pair_ids, B: int = 10000, seed: int = 12345):
        ids = sorted(set(pair_ids))
        self.index = {p: i for i, p in enumerate(ids)}
        self.P, self.B = len(ids), B
        rng = np.random.default_rng(seed)
        self.C = rng.multinomial(self.P, np.full(self.P, 1.0 / self.P), size=B).astype(float)  # B x P counts

    def vectors(self, df: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray]:
        num, den = row_contributions(df)[metric]
        idx = df["pair_id"].map(self.index).to_numpy()
        return (np.bincount(idx, weights=num, minlength=self.P),
                np.bincount(idx, weights=den, minlength=self.P))

    def _draws(self, num: np.ndarray, den: np.ndarray) -> np.ndarray:
        d = self.C @ den
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(d > 0, (self.C @ num) / np.where(d > 0, d, 1), np.nan)

    @staticmethod
    def _interval(draws: np.ndarray, level: float) -> tuple[float, float]:
        ok = draws[np.isfinite(draws)]
        if len(ok) < 50:
            return (float("nan"), float("nan"))
        a = 100 * (1 - level) / 2
        lo, hi = np.percentile(ok, [a, 100 - a])
        return float(lo), float(hi)

    def estimate(self, vec: tuple[np.ndarray, np.ndarray], level: float) -> Est:
        num, den = vec
        point = float(num.sum() / den.sum()) if den.sum() > 0 else float("nan")
        lo, hi = self._interval(self._draws(num, den), level)
        return Est(point, lo, hi, level)

    def contrast(self, a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray], level: float) -> Est:
        """Estimate of metric_a - metric_b with a paired (same clusters, same draws) interval."""
        pa = a[0].sum() / a[1].sum() if a[1].sum() > 0 else float("nan")
        pb = b[0].sum() / b[1].sum() if b[1].sum() > 0 else float("nan")
        lo, hi = self._interval(self._draws(*a) - self._draws(*b), level)
        return Est(float(pa - pb), lo, hi, level)
