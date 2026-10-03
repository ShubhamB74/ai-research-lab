"""PHASE 2 (stub). Multiple testing.

Design sketch:
  World: M candidate signals (M in {1, 50, 200}); K of them have a true small effect (K can be 0).
  CLEAN:   M = 1 pre-specified signal, honest p-value.
  TRAPPED: M = 200 signals provided, a few show p < 0.05 by chance; the question asks about 'the
           best-looking signal'. Ground truth after FDR/Bonferroni correction is what counts.
  Detectability: the full list of tried signals is in the data, so the denominator is visible.
"""
from __future__ import annotations

from ..taxonomy import Trap, Variant
from .base import GeneratedTask, Generator


class MultipleTestingGenerator(Generator):
    family = Trap.MULTIPLE_TESTING

    def generate(self, seed: int, variant: Variant) -> GeneratedTask:
        raise NotImplementedError("Phase 2")
