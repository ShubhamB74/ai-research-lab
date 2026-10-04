"""SURVIVORSHIP family (Phase 2): 2 mechanisms x 3 difficulties.

Both ask: is the average daily return of the universe different from zero? Every bias here is UPWARD.

survivor_only_panel
  World: 100 small-cap names x 400 days, iid returns N(mu, 2%). A name is delisted on the first day
  its cumulative log return reaches -0.40 (its last row is that day). mu in {0, +0.0008}.
  CLEAN  : every name that traded, each until its last day (point-in-time universe). The pooled mean
           is unbiased (Wald's identity), so an honest analyst recovers mu.
  TRAPPED: only names alive at the end. Their pooled mean is strongly biased upward (t ~ +4 at mu=0).
  TRUTH CONVENTION (disclosed): the survivor-only file cannot identify a positive effect, because the
  bias is upward and cannot be separated from mu. The answer the data can SUPPORT is "no effect", so
  effect_exists=False for every trapped task, even when mu > 0. Clean twins keep the true answer.
  Difficulty = how visible the missing names are (all three show a balanced panel with zero attrition):
    easy  : `status` column (clean has 'delisted' rows, trapped has none) + sequential ids with gaps
    medium: sequential vendor ids with gaps
    hard  : opaque ids; the only cue is zero attrition in a universe claimed to include every name

backfilled_membership
  World: 100 candidate stocks x 400 days. A stock joins the index on the first day after its trailing
  60-day log return exceeds +0.20. mu in {0, +0.0010, -0.0010}.
  CLEAN  : rows only from the join date onward (point-in-time membership).
  TRAPPED: the vendor backfilled history before the join date. Pre-join returns were the reason the
           stock joined, so they are biased upward. The `index_add_date` column makes it detectable,
           and honest filtering recovers the clean data, so the TRUE mu (any sign) is identifiable.
  Difficulty = backfill length: easy 60 days, medium 30, hard 10 (weaker bias, same cue).

Calibrated truth: worlds are redrawn until the clean (honest) estimator is unambiguous
(null |t| < 2, effect |t| > 4.5 with the right sign) and, for survivor_only_panel, attrition is in
[12%, 60%].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..stats import mean_t
from ..taxonomy import Trap, Variant
from .base import GeneratedTask, Generator

MECHANISMS = ("survivor_only_panel", "backfilled_membership")
DIFFICULTIES = ("easy", "medium", "hard")
SIGMA = 0.02
NULL_T_MAX = 2.0
EFFECT_T_MIN = 4.5
MAX_ATTEMPTS = 300

# survivor_only_panel
N_A, T_A, BARRIER, MU_A = 100, 400, 0.40, 0.0008
ATTRITION_RANGE = (0.12, 0.60)
# backfilled_membership
N_B, T_B, LOOKBACK, JOIN_THR, MU_B, MIN_AFTER = 100, 400, 60, 0.20, 0.0010, 100
BACKFILL_DAYS = {"easy": 60, "medium": 30, "hard": 10}


def cell_for_seed(seed: int) -> tuple[str, str]:
    """6 cells cycle with the seed (use n divisible by 6)."""
    return MECHANISMS[seed % 2], DIFFICULTIES[(seed // 2) % 3]


def _calibrated(mu: float, t: float) -> bool:
    return abs(t) < NULL_T_MAX if mu == 0.0 else (np.sign(t) == np.sign(mu) and abs(t) > EFFECT_T_MIN)


def draw_world_a(seed: int) -> dict:
    for attempt in range(MAX_ATTEMPTS):
        rng = np.random.default_rng([seed, attempt])
        mu = float(rng.choice([0.0, MU_A]))
        r = np.round(rng.normal(mu, SIGMA, (T_A, N_A)), 6)
        hit = np.cumsum(r, axis=0) <= -BARRIER
        first = np.where(hit.any(0), hit.argmax(0), T_A)      # T_A = never delisted
        last = np.minimum(first, T_A - 1)                     # last observed day (breach day included)
        alive = np.arange(T_A)[:, None] <= last[None, :]
        survived = first == T_A
        attrition = 1.0 - float(survived.mean())
        t = mean_t(r[alive])
        if ATTRITION_RANGE[0] <= attrition <= ATTRITION_RANGE[1] and _calibrated(mu, t):
            return dict(mu=mu, r=r, last=last, alive=alive, survived=survived, attrition=attrition,
                        honest_t=t, attempt=attempt, survivor_t=mean_t(r[:, survived].ravel()))
    raise RuntimeError(f"no calibrated survivorship world for seed {seed}")


def draw_world_b(seed: int) -> dict:
    for attempt in range(MAX_ATTEMPTS):
        rng = np.random.default_rng([seed, attempt])
        mu = float(rng.choice([0.0, MU_B, -MU_B]))
        r = np.round(rng.normal(mu, SIGMA, (T_B, N_B)), 6)
        c = np.vstack([np.zeros((1, N_B)), np.cumsum(r, axis=0)])
        trail = c[LOOKBACK:] - c[:-LOOKBACK]                  # trail[i] = sum r[i .. i+L-1]
        cand = trail[: T_B - MIN_AFTER - LOOKBACK + 1] > JOIN_THR
        add = np.where(cand.any(0), cand.argmax(0) + LOOKBACK, -1)  # member from the day after the window
        members = np.where(add >= 0)[0]
        if len(members) < 30:
            continue
        member_rows = np.arange(T_B)[:, None] >= add[None, members]
        t = mean_t(r[:, members][member_rows])
        if _calibrated(mu, t):
            return dict(mu=mu, r=r, add=add, members=members, honest_t=t, attempt=attempt)
    raise RuntimeError(f"no calibrated backfill world for seed {seed}")


class SurvivorshipGenerator(Generator):
    family = Trap.SURVIVORSHIP

    def generate(self, seed: int, variant: Variant) -> GeneratedTask:
        mech, diff = cell_for_seed(seed)
        return self.generate_cell(seed, variant, mech, diff)

    def generate_cell(self, seed: int, variant: Variant, mechanism: str, difficulty: str) -> GeneratedTask:
        assert mechanism in MECHANISMS and difficulty in DIFFICULTIES
        if mechanism == "survivor_only_panel":
            return self._survivor_only(seed, variant, difficulty)
        return self._backfill(seed, variant, difficulty)

    # ---------------------------------------------------------------- survivor_only_panel
    def _survivor_only(self, seed: int, variant: Variant, difficulty: str) -> GeneratedTask:
        w = draw_world_a(seed)
        trapped = variant is Variant.TRAPPED
        dates = pd.bdate_range("2019-01-01", periods=T_A).strftime("%Y-%m-%d").to_numpy()
        if difficulty == "hard":
            ids = np.array([f"{x:06x}" for x in np.random.default_rng([seed, 99]).choice(16 ** 6, N_A, replace=False)])
        else:
            ids = np.array([f"S{j:03d}" for j in range(N_A)])
        day, asset = np.nonzero(w["alive"])
        if trapped:
            keep = w["survived"][asset]
            day, asset = day[keep], asset[keep]
        order = np.lexsort((day, asset))
        day, asset = day[order], asset[order]
        cols = {"date": dates[day], "asset": ids[asset], "ret": w["r"][day, asset]}
        dictionary = {"date": "business date", "ret": "close-to-close return on `date`"}
        if difficulty == "easy":
            delisted_row = (~w["survived"][asset]) & (day == w["last"][asset])
            cols["status"] = np.where(delisted_row & (not trapped), "delisted", "active")
            dictionary["status"] = "'active', or 'delisted' on a name's final row"
            dictionary["asset"] = "vendor master-list identifier (assigned sequentially across the full universe)"
        elif difficulty == "medium":
            dictionary["asset"] = "vendor master-list identifier (assigned sequentially across the full universe)"
        else:
            dictionary["asset"] = "opaque vendor identifier"
        mu = w["mu"]
        exists = (mu > 0.0) and not trapped      # truth convention: survivor-only data supports no effect
        return GeneratedTask(
            data=pd.DataFrame(cols),
            question=("Daily returns for a universe of small-cap stocks; the vendor states that the file "
                      "covers every name that traded during the window. Is the average daily return "
                      "across the universe different from zero? Report whether an effect exists and its "
                      "direction, and flag any methodological problems you find."),
            data_dictionary=dictionary,
            traps_present=[Trap.SURVIVORSHIP.value] if trapped else [],
            effect_exists=exists,
            true_direction="positive" if exists else "none",
            params={"mu": mu, "mechanism": "survivor_only_panel", "difficulty": difficulty,
                    "attrition": round(w["attrition"], 3), "honest_t": round(w["honest_t"], 3),
                    "survivor_t": round(w["survivor_t"], 3), "resample_attempt": w["attempt"],
                    "truth_convention": "identifiability"},
        )

    # ---------------------------------------------------------------- backfilled_membership
    def _backfill(self, seed: int, variant: Variant, difficulty: str) -> GeneratedTask:
        w = draw_world_b(seed)
        trapped = variant is Variant.TRAPPED
        dates = pd.bdate_range("2019-01-01", periods=T_B).strftime("%Y-%m-%d").to_numpy()
        members, add = w["members"], w["add"][w["members"]]
        start = np.maximum(add - BACKFILL_DAYS[difficulty], 0) if trapped else add
        rows = np.arange(T_B)[:, None] >= start[None, :]
        day, k = np.nonzero(rows)
        order = np.lexsort((day, k))
        day, k = day[order], k[order]
        mu = w["mu"]
        direction = "positive" if mu > 0 else "negative" if mu < 0 else "none"
        return GeneratedTask(
            data=pd.DataFrame({"date": dates[day], "asset": np.array([f"M{i:03d}" for i in range(len(members))])[k],
                               "ret": w["r"][:, members][day, k], "index_add_date": dates[add][k]}),
            question=("Daily returns for stocks in a market index; `index_add_date` is the date each stock "
                      "joined the index. Is the average daily return of index members different from zero? "
                      "Report whether an effect exists and its direction, and flag any methodological "
                      "problems you find."),
            data_dictionary={"date": "business date", "asset": "stock identifier",
                             "ret": "close-to-close return on `date`",
                             "index_add_date": "date the stock was added to the index"},
            traps_present=[Trap.SURVIVORSHIP.value] if trapped else [],
            effect_exists=mu != 0.0,
            true_direction=direction,
            params={"mu": mu, "mechanism": "backfilled_membership", "difficulty": difficulty,
                    "backfill_days": BACKFILL_DAYS[difficulty] if trapped else 0,
                    "honest_t": round(w["honest_t"], 3), "resample_attempt": w["attempt"],
                    "truth_convention": "true_mu"},
        )
