"""Reference agents. They exist to VALIDATE THE BENCHMARK ITSELF, before any LLM is evaluated.

  Oracle         -> must score perfectly (proves the grader and answer key agree). It CHEATS.
  Naive          -> correlates signal with the target, never flags. Must be fooled by leakage.
  AlwaysFlag     -> Naive + flags everything. High recall but 100% false alarms on clean twins:
                    the reason clean twins exist.
  HeuristicCareful -> proves the task is SOLVABLE from public files alone: checks as-of metadata and
                    reverse-engineers how `signal` loads on future returns, then rebuilds an honest
                    signal. If this fails on a cell, the task is unfair, not the agent.
  SanityCheck    -> a plausible 'good practice' agent running CHEAP checks only. Leakage: flags only
                    if the effect size looks too good to be true (|corr| > 0.15). Survivorship: only
                    explicit cues (status column, id gaps). Multiple testing: only visible column
                    counts. Catches the easy cells, misses subtle ones; this is what makes the
                    difficulty ladders and mechanism contrasts meaningful.
All agents identify the task kind from the data schema only (see analysis.kind).
  Random         -> chance-level floor.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import analysis
from ..schema import GroundTruth, Task
from ..taxonomy import Trap
from .base import Agent


def _tstat(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    r = float(np.corrcoef(x, y)[0, 1])
    t = r * np.sqrt((len(x) - 2) / max(1e-12, 1 - r * r))
    return r, float(t)


def _answer(task: Task, exists: bool, direction: str, conf: float, flags: list[str], **extra) -> dict:
    return {"task_id": task.task_id, "effect_exists": exists,
            "direction": direction if exists else "none", "confidence": conf,
            "flags": flags, **extra}


def _from_result(task: Task, res: analysis.Result, conf: float, hypothesis: str) -> dict:
    return _answer(task, res.exists, res.direction, conf, list(res.flags),
                   hypothesis=hypothesis, experiments=res.notes)


class NaiveAgent(Agent):
    name = "naive"
    T_CUT = 3.0

    def run(self, task: Task) -> dict[str, Any]:
        df = pd.read_csv(task.data_path)
        k = analysis.kind(df)
        if k != "leakage":
            return _from_result(task, analysis.naive(df, k), 0.6, "headline statistic is significant")
        r, t = _tstat(df["signal"].to_numpy(), df["fwd_ret_1d"].to_numpy())
        exists = abs(t) > self.T_CUT
        return _answer(task, exists, "positive" if r > 0 else "negative", 0.6, [],
                       hypothesis="signal predicts next-day return", predicted_direction="positive",
                       experiments=[{"desc": "pooled corr(signal, fwd_ret_1d)", "r": r, "t": t}])


class AlwaysFlagAgent(NaiveAgent):
    name = "always_flag"

    def run(self, task: Task) -> dict[str, Any]:
        out = super().run(task)
        out["flags"] = [Trap.LEAKAGE.value, Trap.SURVIVORSHIP.value, Trap.MULTIPLE_TESTING.value]
        return out


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def run(self, task: Task) -> dict[str, Any]:
        exists = self.rng.random() < 0.5
        return _answer(task, exists, self.rng.choice(["positive", "negative"]), 0.5,
                       [f for f in task.allowed_flags if self.rng.random() < 0.2])


def _prepare(task: Task) -> pd.DataFrame:
    return pd.read_csv(task.data_path).sort_values(["asset", "date"]).reset_index(drop=True)


def _honest_signal(df: pd.DataFrame) -> pd.Series:
    """The vendor says `signal` is a 5-day smoothed return: rebuild it from past returns only."""
    return df.groupby("asset")["ret"].transform(lambda s: s.rolling(5).mean())


class SanityCheckAgent(Agent):
    """Flags leakage only when the effect size is implausible for daily returns."""
    name = "sanity_check"
    T_CUT = 3.0
    TOO_GOOD = 0.15

    def run(self, task: Task) -> dict[str, Any]:
        raw = pd.read_csv(task.data_path)
        k = analysis.kind(raw)
        if k != "leakage":
            return _from_result(task, analysis.surface(raw, k), 0.75, "cheap checks, then headline statistic")
        df = _prepare(task)
        signal, flags = df["signal"], []
        r, t = _tstat(signal.to_numpy(), df["fwd_ret_1d"].to_numpy())
        if abs(r) > self.TOO_GOOD:
            flags.append(Trap.LEAKAGE.value)
            signal = _honest_signal(df)
            r, t = _tstat(signal.to_numpy(), df["fwd_ret_1d"].to_numpy())
        exists = abs(t) > self.T_CUT
        return _answer(task, exists, "positive" if r > 0 else "negative", 0.75, flags,
                       hypothesis="signal predicts next-day return unless effect size is implausible")


class HeuristicCarefulAgent(Agent):
    """Two independent leak detectors, then an honest re-test.

    1. Metadata: any `signal_asof` later than `date` means the value was recorded after the fact.
    2. Reverse-engineering: regress `signal` on returns at t-6..t and t+1, t+2. An honest trailing
       signal has exactly zero loading on future returns; any loading is a future peek.
    """
    name = "heuristic_careful"
    T_CUT = 3.0
    LEAD_TOL = 1e-6

    def _future_loading(self, df: pd.DataFrame) -> float:
        g = df.groupby("asset")["ret"]
        cols = {f"lag{k}": g.shift(k) for k in range(0, 7)}
        cols["lead1"], cols["lead2"] = g.shift(-1), g.shift(-2)
        X = pd.DataFrame(cols)
        ok = X.notna().all(axis=1) & df["signal"].notna()
        coef = np.linalg.lstsq(X[ok].to_numpy(), df.loc[ok, "signal"].to_numpy(), rcond=None)[0]
        return float(np.max(np.abs(coef[-2:])))

    def run(self, task: Task) -> dict[str, Any]:
        raw = pd.read_csv(task.data_path)
        k = analysis.kind(raw)
        if k != "leakage":
            return _from_result(task, analysis.careful(raw, k), 0.8, "correct for the flaw, then test")
        df = _prepare(task)
        notes: list[dict] = []
        leak = False
        if "signal_asof" in df.columns:
            late = bool((pd.to_datetime(df["signal_asof"]) > pd.to_datetime(df["date"])).any())
            notes.append({"desc": "any signal_asof > date", "result": late})
            leak = leak or late
        loading = self._future_loading(df)
        notes.append({"desc": "max |loading| of signal on returns at t+1,t+2", "result": loading})
        leak = leak or loading > self.LEAD_TOL
        signal, flags = df["signal"], []
        if leak:
            flags.append(Trap.LEAKAGE.value)
            signal = _honest_signal(df)
        r, t = _tstat(signal.to_numpy(), df["fwd_ret_1d"].to_numpy())
        notes.append({"desc": "corr(honest signal, fwd_ret_1d)", "r": r, "t": t})
        exists = abs(t) > self.T_CUT
        return _answer(task, exists, "positive" if r > 0 else "negative", 0.8, flags,
                       hypothesis="past returns weakly predict next-day return",
                       experiments=notes)


class OracleAgent(Agent):
    """CHEATS by reading the private answer key. Only for validating the grader."""
    name = "oracle"

    def __init__(self, private_root: Path):
        self.private_root = Path(private_root)

    def run(self, task: Task) -> dict[str, Any]:
        truth = GroundTruth.load(self.private_root / f"{task.task_id}.json")
        return _answer(task, truth.effect_exists, truth.true_direction, 1.0, list(truth.traps_present))
