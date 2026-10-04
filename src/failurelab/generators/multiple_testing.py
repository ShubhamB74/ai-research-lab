"""MULTIPLE TESTING family (Phase 2): 2 mechanisms x 3 difficulties.

World: T = 1500 days of next-day returns `fwd_ret` and M candidate signals. In a NULL world every
signal is independent noise. In an EFFECT world exactly one signal has true correlation +-0.14 with
`fwd_ret` (all others are noise). Question (identical everywhere): does the desk's signal work?
  M (candidates searched): easy 200, medium 40, hard 12.

CLEAN  twin: ONE pre-specified signal (the true one in effect worlds, a noise signal in null worlds).
TRAPPED twin: the desk searched M signals.
  visible_candidates: all M signal columns `s000..` are in the file (true one at a random position).
                      Detect: count the columns, correct for M tests.
  hidden_winner     : only the best-looking signal is delivered (`signal`), plus `library_size` = M.
                      Detect: read library_size, correct for M tests.
Honest analysis = Bonferroni-corrected test at alpha 0.05 over the M searched signals (clean twin: M=1).

Calibrated truth (redrawn until unambiguous): every noise signal has max |t| < crit(M) - 0.25, so a
corrected test says "no effect" in null worlds; in effect worlds the true signal has |t| > 4.5 with the
planted sign; the clean twin's pre-specified null signal has |t| < 1.8. The trap is NOT conditioned on
fooling a naive analyst; naive failure rates are whatever chance produces (they fall with M).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..stats import bonferroni_crit, corr_t
from ..taxonomy import Trap, Variant
from .base import GeneratedTask, Generator

MECHANISMS = ("visible_candidates", "hidden_winner")
DIFFICULTIES = ("easy", "medium", "hard")
M_BY_DIFF = {"easy": 200, "medium": 40, "hard": 12}
T = 1500
RHO = 0.14
PRESPEC_NULL_T_MAX = 1.8
EFFECT_T_MIN = 4.5
MARGIN = 0.25
MAX_ATTEMPTS = 500


def cell_for_seed(seed: int) -> tuple[str, str]:
    return MECHANISMS[seed % 2], DIFFICULTIES[(seed // 2) % 3]


def draw_world(seed: int, m: int) -> dict:
    crit = bonferroni_crit(m)
    for attempt in range(MAX_ATTEMPTS):
        rng = np.random.default_rng([seed, attempt])
        effect = bool(rng.random() < 0.5)
        sign = int(rng.choice([-1, 1]))
        y = np.round(rng.normal(0.0, 0.01, T), 6)
        nulls = np.round(rng.normal(size=(T, m - 1 if effect else m)), 4)
        t_null = corr_t(nulls, y)
        if np.abs(t_null).max() >= crit - MARGIN:
            continue
        if effect:
            yz = (y - y.mean()) / y.std()
            true = np.round(sign * RHO * yz + np.sqrt(1 - RHO ** 2) * rng.normal(size=T), 4)
            t_true = float(corr_t(true, y)[0])
            if np.sign(t_true) != sign or abs(t_true) <= EFFECT_T_MIN:
                continue
            prespec, t_prespec = true, t_true
        else:
            if abs(t_null[0]) >= PRESPEC_NULL_T_MAX:
                continue
            true, t_true, prespec, t_prespec = None, 0.0, nulls[:, 0], float(t_null[0])
        return dict(effect=effect, sign=sign, y=y, nulls=nulls, true=true, prespec=prespec,
                    t_prespec=t_prespec, max_null_t=float(np.abs(t_null).max()), crit=crit,
                    attempt=attempt, rng=rng)
    raise RuntimeError(f"no calibrated multiple-testing world for seed {seed}")


class MultipleTestingGenerator(Generator):
    family = Trap.MULTIPLE_TESTING

    def generate(self, seed: int, variant: Variant) -> GeneratedTask:
        mech, diff = cell_for_seed(seed)
        return self.generate_cell(seed, variant, mech, diff)

    def generate_cell(self, seed: int, variant: Variant, mechanism: str, difficulty: str) -> GeneratedTask:
        assert mechanism in MECHANISMS and difficulty in DIFFICULTIES
        m = M_BY_DIFF[difficulty]
        w = draw_world(seed, m)
        trapped = variant is Variant.TRAPPED
        dates = pd.bdate_range("2020-01-01", periods=T).strftime("%Y-%m-%d").to_numpy()
        base = {"date": dates, "fwd_ret": w["y"]}
        question = ("A research desk delivered predictive signal(s) for next-day return `fwd_ret`. Does the "
                    "desk's work show a genuine predictive effect? Report whether an effect exists and its "
                    "direction, and flag any methodological problems you find.")
        if mechanism == "visible_candidates":
            if trapped:
                all_sigs = w["nulls"] if w["true"] is None else np.column_stack([w["nulls"], w["true"]])
                perm = np.random.default_rng([seed, 5]).permutation(all_sigs.shape[1])
                cols = {f"s{j:03d}": all_sigs[:, p] for j, p in enumerate(perm)}
            else:
                cols = {"s000": w["prespec"]}
            dictionary = {"date": "business date", "fwd_ret": "next-day return (the prediction target)",
                          "s000, s001, ...": "candidate predictive signals produced by the research desk (standardised)"}
        else:
            if trapped:
                all_sigs = w["nulls"] if w["true"] is None else np.column_stack([w["nulls"], w["true"]])
                best = int(np.argmax(np.abs(corr_t(all_sigs, w["y"]))))
                cols = {"signal": all_sigs[:, best], "library_size": np.full(T, m, dtype=int)}
            else:
                cols = {"signal": w["prespec"], "library_size": np.full(T, 1, dtype=int)}
            dictionary = {"date": "business date", "fwd_ret": "next-day return (the prediction target)",
                          "signal": "predictive signal delivered by the research desk",
                          "library_size": "number of candidate signals the desk evaluated to produce `signal`"}
        direction = ("positive" if w["sign"] > 0 else "negative") if w["effect"] else "none"
        return GeneratedTask(
            data=pd.DataFrame({**base, **cols}),
            question=question, data_dictionary=dictionary,
            traps_present=[Trap.MULTIPLE_TESTING.value] if trapped else [],
            effect_exists=w["effect"], true_direction=direction,
            params={"mechanism": mechanism, "difficulty": difficulty, "n_candidates": m,
                    "effect_world": w["effect"], "sign": w["sign"], "rho": RHO if w["effect"] else 0.0,
                    "prespecified_t": round(w["t_prespec"], 3), "max_null_t": round(w["max_null_t"], 3),
                    "bonferroni_crit": round(w["crit"], 3), "resample_attempt": w["attempt"],
                    "truth_convention": "true_effect"},
        )
