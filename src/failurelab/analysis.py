"""Reference analyses for the Phase 2 families (survivorship, multiple testing).

Used ONLY by the scripted baseline agents. Three levels of diligence per task kind:
  naive   : the analysis a hurried analyst would run; never flags.
  surface : cheap checks that need no deeper reasoning (explicit cues, column counts).
  careful : full diligence; proves each cell is solvable from the public files alone.

Agents identify the task kind from the schema (columns) only; trap family and variant are never
exposed in public task files.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .stats import bonferroni_crit, corr_t, mean_t
from .taxonomy import Trap

T_CUT = 3.0        # decision threshold for 'effect exists' (calibrated worlds: null<2, effect>4.5)
NAIVE_T = 1.96     # a hurried analyst's p < 0.05


@dataclass
class Result:
    exists: bool
    direction: str
    flags: list[str] = field(default_factory=list)
    notes: list[dict] = field(default_factory=list)


def kind(df: pd.DataFrame) -> str:
    cols = set(df.columns)
    if {"signal", "fwd_ret_1d", "ret"} <= cols:
        return "leakage"
    if "index_add_date" in cols:
        return "backfill"
    if "library_size" in cols:
        return "mt_hidden"
    if "fwd_ret" in cols and _mt_cols(df):
        return "mt_visible"
    if {"date", "asset", "ret"} <= cols:
        return "survivor_panel"
    return "unknown"


def _mt_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if re.fullmatch(r"s\d+", c)]


def _sign(x: float) -> str:
    return "positive" if x > 0 else "negative"


def _decide(t: float, cut: float, notes: list, flags: list | None = None) -> Result:
    return Result(abs(t) > cut, _sign(t), flags or [], notes)


# ------------------------------------------------------------------ survivorship
def _survivor_cues(df: pd.DataFrame) -> dict[str, bool]:
    cues: dict[str, bool] = {}
    if "status" in df.columns:
        cues["status_has_no_delistings"] = not (df["status"] == "delisted").any()
    ids = df["asset"].unique()
    if all(re.fullmatch(r"S\d+", str(a)) for a in ids):
        nums = sorted(int(str(a)[1:]) for a in ids)
        cues["id_gaps"] = nums[-1] + 1 != len(nums)
    return cues


def _balanced(df: pd.DataFrame) -> bool:
    return df.groupby("asset").size().nunique() == 1


# ------------------------------------------------------------------ naive
def naive(df: pd.DataFrame, k: str) -> Result:
    if k in ("survivor_panel", "backfill"):
        t = mean_t(df["ret"].to_numpy())
        return _decide(t, NAIVE_T, [{"desc": "t-test of pooled mean return", "t": t}])
    if k == "mt_visible":
        cols = _mt_cols(df)
        ts = corr_t(df[cols].to_numpy(), df["fwd_ret"].to_numpy())
        i = int(np.argmax(np.abs(ts)))
        return _decide(float(ts[i]), NAIVE_T, [{"desc": f"best signal {cols[i]}", "t": float(ts[i])}])
    if k == "mt_hidden":
        t = float(corr_t(df["signal"].to_numpy(), df["fwd_ret"].to_numpy())[0])
        return _decide(t, NAIVE_T, [{"desc": "t of delivered signal", "t": t}])
    return Result(False, "none")


# ------------------------------------------------------------------ careful / surface
def _careful_survivor(df: pd.DataFrame, use_balance: bool) -> Result:
    cues = _survivor_cues(df)
    if use_balance:
        cues["balanced_panel_no_attrition"] = _balanced(df)
    notes = [{"desc": "survivorship cues", "cues": cues}]
    if any(cues.values()):
        # The bias is upward, so survivor-only data can never support a positive effect.
        notes.append({"desc": "survivor-only data cannot identify an effect; no supported effect"})
        return Result(False, "none", [Trap.SURVIVORSHIP.value], notes)
    t = mean_t(df["ret"].to_numpy())
    notes.append({"desc": "pooled mean return", "t": t})
    return _decide(t, T_CUT, notes)


def _careful_backfill(df: pd.DataFrame) -> Result:
    early = pd.to_datetime(df["date"]) < pd.to_datetime(df["index_add_date"])
    notes = [{"desc": "rows before index_add_date", "n": int(early.sum())}]
    flags = [Trap.SURVIVORSHIP.value] if early.any() else []
    t = mean_t(df.loc[~early, "ret"].to_numpy())
    notes.append({"desc": "pooled mean on point-in-time member rows", "t": t})
    return _decide(t, T_CUT, notes, flags)


def _careful_mt(df: pd.DataFrame, k: str, read_library: bool) -> Result:
    if k == "mt_visible":
        cols = _mt_cols(df)
        ts = corr_t(df[cols].to_numpy(), df["fwd_ret"].to_numpy())
        m = len(cols)
    else:
        ts = corr_t(df["signal"].to_numpy(), df["fwd_ret"].to_numpy())
        m = int(df["library_size"].iloc[0]) if read_library else 1
    i = int(np.argmax(np.abs(ts)))
    crit = bonferroni_crit(m)
    notes = [{"desc": "tests searched", "m": m}, {"desc": "best |t| vs Bonferroni crit",
                                                  "t": float(ts[i]), "crit": crit}]
    return _decide(float(ts[i]), crit, notes, [Trap.MULTIPLE_TESTING.value] if m > 1 else [])


def careful(df: pd.DataFrame, k: str) -> Result:
    if k == "survivor_panel":
        return _careful_survivor(df, use_balance=True)
    if k == "backfill":
        return _careful_backfill(df)
    if k in ("mt_visible", "mt_hidden"):
        return _careful_mt(df, k, read_library=True)
    return Result(False, "none")


def surface(df: pd.DataFrame, k: str) -> Result:
    """Cheap checks only: explicit status/id cues, the index-add column, visible column counts.
    Does NOT notice a balanced panel and does NOT read the library_size metadata."""
    if k == "survivor_panel":
        return _careful_survivor(df, use_balance=False)
    if k == "backfill":
        return _careful_backfill(df)
    if k == "mt_visible":
        return _careful_mt(df, k, read_library=True)
    if k == "mt_hidden":
        return naive(df, k)
    return Result(False, "none")
