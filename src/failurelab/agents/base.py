from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..schema import Task


class Agent(ABC):
    """An agent receives ONLY the public Task and returns a raw dict (untrusted, validated later).

    Phase 3 adds LLM agents here: they get a sandboxed Python tool, the task, and the data path,
    and must end by emitting a JSON object matching the Submission schema."""
    name: str

    @abstractmethod
    def run(self, task: Task) -> dict[str, Any]: ...
