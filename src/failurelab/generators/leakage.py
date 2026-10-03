"""LEAKAGE family (Phase 1): 3 mechanisms x 3 difficulty levels.

World: 20 assets x 750 days, r_t = phi * r_{t-1} + eps_t, phi in {0, +0.10, -0.10}.
  phi is the HONEST predictive effect of past returns on the next-day return (ground truth).
Question (identical for every task): does `signal` genuinely predict `fwd_ret_1d`?
  H_t = trailing 5-day mean of returns (r_{t-4..t})   <- the honest signal; used by every CLEAN twin.

Mechanisms (what the TRAPPED twin does wrong). Each has a different detection path:
  centered_window : signal blends H with a CENTERED window (r_{t-2..t+2}); peeks 1-2 days ahead.
                    Detect: signal loads on returns at t+1 / t+2 when reverse-engineered.
  missing_shift   : signal blends H with a window ending at t+1 (a forgotten one-day shift when the
                    vendor aligned the signal). Detect: loads on r_{t+1}; effect size too good.
  asof_timestamp  : a share of rows were restated after the fact. A `signal_asof` column is in the
                    data (always present, equal to `date` in clean twins); restated rows carry the
                    t+1 window and an as-of date AFTER `date`. Detect: signal_asof > date.

Difficulty = how much of the signal is contaminated:
  centered_window / missing_shift: blend weight b in {easy 1.0, medium 0.25, hard 0.10}
  asof_timestamp: share of restated rows p in {easy 1.0, medium 0.30, hard 0.08}
  Easy leaks produce an absurd effect size (corr ~0.45) that a plausibility check catches.
  Hard leaks produce corr ~0.05-0.10, which looks like a believable weak signal.

Calibrated ground truth (validity rule): a null world can by luck show |t| > 3 for an honest
analyst, which would punish a CORRECT answer. So each world is redrawn (deterministically, from
[seed, attempt]) until the honest test unambiguously agrees with the planted truth:
  null (phi = 0):   |t_honest| < 2        effect (phi != 0): |t_honest| > 4.5 with the right sign.
The attempt index is stored in params. This selection is disclosed in the README.

Rules every cell must satisfy (enforced in tests):
  * CLEAN twin's signal is EXACTLY the honest trailing mean (a true negative control).
  * Same seed => same returns in both twins; only the signal construction differs.
  * Public text is identical across twins; NaN edges trimmed identically (no shape tell).
  * Solvable from public files alone (HeuristicCarefulAgent must solve every cell).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..taxonomy import Trap, Variant
from .base import GeneratedTask, Generator

N_ASSETS = 20
N_DAYS = 750
WINDOW = 5
PHI_CHOICES = (0.0, 0.10, -0.10)

MECHANISMS = ("centered_window", "missing_shift", "asof_timestamp")
DIFFICULTIES = ("easy", "medium", "hard")
BLEND = {"easy": 1.0, "medium": 0.25, "hard": 0.10}
RESTATED_SHARE = {"easy": 1.0, "medium": 0.30, "hard": 0.08}


MAX_ATTEMPTS = 200
NULL_T_MAX = 2.0
EFFECT_T_MIN = 4.5


def _simulate(rng: np.random.Generator) -> tuple[float, np.ndarray]:
    phi = float(rng.choice(PHI_CHOICES))
    eps = rng.normal(0.0, 0.01, size=(N_DAYS, N_ASSETS))
    ret = np.zeros_like(eps)
    ret[0] = eps[0]
    for t in range(1, N_DAYS):
        ret[t] = phi * ret[t - 1] + eps[t]
    return phi, ret


def honest_t(ret: np.ndarray) -> float:
    """t-stat of the pooled corr between the honest trailing signal and next-day return."""
    wide = pd.DataFrame(ret)
    x = wide.rolling(WINDOW).mean().iloc[WINDOW:N_DAYS - WINDOW].to_numpy().ravel()
    y = wide.shift(-1).iloc[WINDOW:N_DAYS - WINDOW].to_numpy().ravel()
    r = float(np.corrcoef(x, y)[0, 1])
    return r * float(np.sqrt((len(x) - 2) / (1 - r * r)))


def draw_world(seed: int) -> tuple[float, np.ndarray, int, float]:
    """Deterministic rejection sampling so the planted truth is unambiguous (see module doc)."""
    for attempt in range(MAX_ATTEMPTS):
        phi, ret = _simulate(np.random.default_rng([seed, attempt]))
        t = honest_t(ret)
        if phi == 0.0 and abs(t) < NULL_T_MAX:
            return phi, ret, attempt, t
        if phi != 0.0 and np.sign(t) == np.sign(phi) and abs(t) > EFFECT_T_MIN:
            return phi, ret, attempt, t
    raise RuntimeError(f"no calibrated world found for seed {seed}")


def cell_for_seed(seed: int) -> tuple[str, str]:
    """Deterministic, balanced assignment: 9 cells cycle with the seed (use n divisible by 9)."""
    return MECHANISMS[seed % 3], DIFFICULTIES[(seed // 3) % 3]


class LeakageGenerator(Generator):
    family = Trap.LEAKAGE

    def generate(self, seed: int, variant: Variant) -> GeneratedTask:
        mechanism, difficulty = cell_for_seed(seed)
        return self.generate_cell(seed, variant, mechanism, difficulty)

    def generate_cell(self, seed: int, variant: Variant, mechanism: str, difficulty: str) -> GeneratedTask:
        assert mechanism in MECHANISMS and difficulty in DIFFICULTIES
        phi, ret, attempt, t_honest = draw_world(seed)

        dates = pd.bdate_range("2018-01-01", periods=N_DAYS)
        assets = [f"A{i:02d}" for i in range(N_ASSETS)]
        wide = pd.DataFrame(ret, index=dates, columns=assets)

        honest = wide.rolling(WINDOW).mean()              # r_{t-4..t}
        shifted = honest.shift(-1)                        # r_{t-3..t+1}  (window ends tomorrow)
        centered = wide.rolling(WINDOW, center=True).mean()  # r_{t-2..t+2}
        fwd = wide.shift(-1)

        trapped = variant is Variant.TRAPPED
        asof = np.repeat(dates.values[:, None], N_ASSETS, axis=1)   # as-of == date by default
        if not trapped:
            signal = honest
            strength = 0.0
        elif mechanism == "centered_window":
            strength = BLEND[difficulty]
            signal = (1 - strength) * honest + strength * centered
        elif mechanism == "missing_shift":
            strength = BLEND[difficulty]
            signal = (1 - strength) * honest + strength * shifted
        else:  # asof_timestamp
            strength = RESTATED_SHARE[difficulty]
            mask = np.random.default_rng([seed, 7]).random((N_DAYS, N_ASSETS)) < strength
            signal = honest.where(~mask, shifted)
            next_day = (dates + pd.offsets.BDay(1)).values[:, None]
            asof = np.where(mask, next_day, asof)

        keep = slice(WINDOW, N_DAYS - WINDOW)  # identical trim in both variants
        n_keep = len(dates[keep])
        date_str = dates[keep].strftime("%Y-%m-%d").to_numpy()
        cols = {
            "date": np.repeat(date_str, N_ASSETS),
            "asset": np.tile(assets, n_keep),
            "ret": wide.iloc[keep].to_numpy().ravel(),
            "signal": signal.iloc[keep].to_numpy().ravel(),
            "fwd_ret_1d": fwd.iloc[keep].to_numpy().ravel(),
        }
        dictionary = {
            "date": "business date",
            "asset": "asset identifier",
            "ret": "close-to-close return on `date`",
            "signal": "5-day smoothed return signal supplied by upstream vendor",
            "fwd_ret_1d": "return on the next business day (the prediction target)",
        }
        if mechanism == "asof_timestamp":
            cols["signal_asof"] = (pd.Series(asof[keep].ravel()).dt.strftime("%Y-%m-%d").to_numpy())
            dictionary["signal_asof"] = "date on which the `signal` value was computed/recorded by the vendor"

        direction = "positive" if phi > 0 else "negative" if phi < 0 else "none"
        return GeneratedTask(
            data=pd.DataFrame(cols),
            question=("Daily returns for 20 assets with a vendor-supplied `signal`. Does `signal` "
                      "genuinely predict the next-day return (`fwd_ret_1d`)? Report whether a "
                      "predictive effect exists and its direction, and flag any methodological "
                      "problems you find."),
            data_dictionary=dictionary,
            traps_present=[Trap.LEAKAGE.value] if trapped else [],
            effect_exists=phi != 0.0,
            true_direction=direction,
            params={"phi": phi, "window": WINDOW, "mechanism": mechanism,
                    "difficulty": difficulty, "strength": strength,
                    "resample_attempt": attempt, "honest_t": round(t_honest, 3),
                    "n_assets": N_ASSETS, "n_days": N_DAYS},
        )
