"""PHASE 2 (stub). Survivorship bias.

Design sketch:
  World: assets with a true mean return mu (can be 0). Each asset has a random 'death' time that is
         more likely after bad returns (delisting).
  CLEAN:   panel includes delisted assets through their last observation (point-in-time universe).
  TRAPPED: panel contains only assets alive at the end -> naive mean return is biased upward.
  Question: "Is the average asset return positive?" Ground truth = mu.
  Detectability: asset count is constant over the sample / no asset has a missing tail, and a
  `listing_status` or first/last-date column lets an analyst notice conditioning on survival.
"""
from __future__ import annotations

from ..taxonomy import Trap, Variant
from .base import GeneratedTask, Generator


class SurvivorshipGenerator(Generator):
    family = Trap.SURVIVORSHIP

    def generate(self, seed: int, variant: Variant) -> GeneratedTask:
        raise NotImplementedError("Phase 2")
