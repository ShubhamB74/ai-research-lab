"""Aggregate TaskScores into the reliability scorecard (with Wilson 95% intervals)."""
from __future__ import annotations

import math

import pandas as pd

from .grader import TaskScore


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def to_frame(scores_by_agent: dict[str, list[TaskScore]]) -> pd.DataFrame:
    rows = [{"agent": a, **s.__dict__} for a, ss in scores_by_agent.items() for s in ss]
    return pd.DataFrame(rows)


def _fmt(k: int, n: int) -> str:
    if n == 0:
        return "n/a"
    lo, hi = wilson(k, n)
    return f"{k / n:.2f} [{lo:.2f}, {hi:.2f}]"


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (agent, family, variant). Metrics are defined in the README."""
    out = []
    for (agent, family, variant), g in df.groupby(["agent", "family", "variant"]):
        n = len(g)
        trapped = variant == "trapped"
        row = {
            "agent": agent, "family": family, "variant": variant, "n": n,
            "invalid": int((~g.valid).sum()),
            "accuracy": _fmt(int(g.conclusion_correct.sum()), n),
            "brier": round(float(g.brier.mean()), 3),
        }
        if trapped:
            row["fooled_rate"] = _fmt(int(g.fooled.sum()), n)
            row["flag_recall"] = _fmt(int(g.tp.sum()), int(g.tp.sum() + g.fn.sum()))
            row["right_wrong_reason"] = _fmt(int(g.right_for_wrong_reason.sum()), n)
        else:
            row["false_flag_rate"] = _fmt(int((g.fp > 0).sum()), n)
        out.append(row)
    return pd.DataFrame(out).fillna("")


def breakdown(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Trapped tasks only, one row per (agent, family, key). Used for the difficulty ladder and the
    per-mechanism table. Order of `key` values follows their first appearance sorted naturally."""
    order = {"easy": 0, "medium": 1, "hard": 2}
    if key == "cell":   # mechanism/difficulty, so mechanisms with different cues are not blended
        df = df.assign(cell=df["mechanism"] + "/" + df["difficulty"])
    out = []
    for (agent, fam, k), g in df[df.variant == "trapped"].groupby(["agent", "family", key]):
        n = len(g)
        out.append({"agent": agent, "family": fam, key: k, "n": n,
                    "accuracy": _fmt(int(g.conclusion_correct.sum()), n),
                    "fooled_rate": _fmt(int(g.fooled.sum()), n),
                    "flag_recall": _fmt(int(g.tp.sum()), int(g.tp.sum() + g.fn.sum())),
                    "_o": order.get(str(k).split("/")[-1], 9)})
    res = pd.DataFrame(out).sort_values(["agent", "family", "_o", key]).drop(columns="_o")
    return res.reset_index(drop=True)


def to_markdown(summary: pd.DataFrame) -> str:
    cols = list(summary.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in summary.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)
