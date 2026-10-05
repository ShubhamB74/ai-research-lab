"""Simulation-based precision planning: choose the number of pairs and reps BEFORE spending API money.

Simulates experiments with assumed true rates (pair-level heterogeneity included) and applies the SAME decision
rules as the real analysis (study.evaluate_hypotheses), so 'power' here means: how often would the real
pipeline return 'supported' / 'falsified' / 'inconclusive' if these rates were true.
Covers H1 and H2 (they need only flag and accuracy rates). Assumed rates are guesses, so look at a range.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .inference import Study
from .prereg import DEFAULT_RULES
from .study import evaluate_hypotheses


def _expit(x):
    return 1 / (1 + np.exp(-x))


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def simulate_df(rng, pairs, reps, recall, ffr, clean_acc, trapped_acc, pair_sd, conditions):
    """recall/ffr: dict condition -> assumed rate."""
    u, v = rng.normal(size=pairs), rng.normal(size=pairs)          # pair-level difficulty (shared across conditions)
    rows = []
    for cond in conditions:
        p_rec = _expit(_logit(recall[cond]) + pair_sd * u)[:, None] * np.ones((1, reps))
        p_ff = _expit(_logit(ffr[cond]) + pair_sd * v)[:, None] * np.ones((1, reps))
        tp = (rng.random((pairs, reps)) < p_rec).astype(int)
        ff = (rng.random((pairs, reps)) < p_ff).astype(int)
        c_t = rng.random((pairs, reps)) < trapped_acc
        c_c = rng.random((pairs, reps)) < clean_acc
        for p in range(pairs):
            for r in range(reps):
                rows.append((f"p{p}", cond, "trapped", tp[p, r], 1 - tp[p, r], 0, bool(c_t[p, r]), False,
                             bool(c_t[p, r] and not tp[p, r]), 0.25))
                rows.append((f"p{p}", cond, "clean", 0, 0, ff[p, r], bool(c_c[p, r]), False, False, 0.05))
    return pd.DataFrame(rows, columns=["pair_id", "condition", "variant", "tp", "fn", "fp", "conclusion_correct",
                                       "fooled", "right_for_wrong_reason", "brier"])


def simulate_power(pairs: int, reps: int, recall_none: float, recall_checklist: float, ffr_none: float = 0.05,
                   ffr_checklist: float = 0.05, clean_acc: float = 0.95, trapped_acc: float = 0.5,
                   pair_sd: float = 1.0, n_models: int = 1, sims: int = 200, B: int = 400, seed: int = 0,
                   rules: dict | None = None) -> dict:
    rules = rules or DEFAULT_RULES
    rng = np.random.default_rng(seed)
    level = 1 - 0.05 / (4 * n_models)
    recall = {"none": recall_none, "checklist": recall_checklist}
    ffr = {"none": ffr_none, "checklist": ffr_checklist}
    tally = {h: {"supported": 0, "falsified": 0, "inconclusive": 0, "not_evaluable": 0} for h in ("H1", "H2")}
    for _ in range(sims):
        df = simulate_df(rng, pairs, reps, recall, ffr, clean_acc, trapped_acc, pair_sd, ["none", "checklist"])
        study = Study(df.pair_id.unique(), B=B, seed=int(rng.integers(1 << 30)))
        res = evaluate_hypotheses(df, rules, study, level, which=("H1", "H2"))
        for h in tally:
            tally[h][res[h]["verdict"]] += 1
    return {h: {k: v / sims for k, v in t.items()} for h, t in tally.items()}
