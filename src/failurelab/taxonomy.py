"""The failure taxonomy. These codes are the ONLY vocabulary agents may use to flag problems,
which is what makes flag-grading deterministic (set comparison, no text judging)."""
from enum import Enum


class Trap(str, Enum):
    LEAKAGE = "leakage"                    # look-ahead / future information in features
    SURVIVORSHIP = "survivorship"          # sample conditioned on surviving to the end
    MULTIPLE_TESTING = "multiple_testing"  # many comparisons, no correction
    # Not implemented yet (stretch goals, see README roadmap):
    OVERFITTING = "overfitting"
    CONFOUNDING = "confounding"
    REGIME_SHIFT = "regime_shift"


class Variant(str, Enum):
    CLEAN = "clean"      # same question, no planted flaw (measures false alarms)
    TRAPPED = "trapped"  # same question, one planted flaw


# Agents always see ALL codes as allowed flags, even ones not yet implemented,
# so the flag space never reveals which traps exist.
ALL_FLAG_CODES = tuple(t.value for t in Trap)
