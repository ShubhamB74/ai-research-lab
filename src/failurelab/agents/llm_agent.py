"""PHASE 3 (stub). LLM research agent adapter.

Planned contract:
  - Input: public Task only (question, data path, data dictionary, allowed flags).
  - Tools: a sandboxed Python executor (no network, only the public task dir mounted read-only).
  - Prompt conditions (ablation): "none" | "generic_hint" | "checklist".
  - Output: ONE json object matching schema.Submission; the full transcript, code and results are
    saved as `experiments` so failure taxonomy work can be done afterwards.
  - Repeat each (task, condition) k times with different sampling seeds; report CIs.
"""
from __future__ import annotations

from ..schema import Task
from .base import Agent


class LLMAgent(Agent):
    def __init__(self, model: str, condition: str = "none", seed: int = 0):
        self.name = f"{model}|{condition}"
        self.model, self.condition, self.seed = model, condition, seed

    def run(self, task: Task):
        raise NotImplementedError("Phase 3")
