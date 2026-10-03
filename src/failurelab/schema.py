"""Data contracts. Everything an agent returns is untrusted input and is validated here."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .taxonomy import ALL_FLAG_CODES

DIRECTIONS = ("positive", "negative", "none")
SCHEMA_VERSION = 1


class SubmissionError(ValueError):
    """Raised when an agent's output does not satisfy the submission schema."""


@dataclass(frozen=True)
class Task:
    """What the agent is allowed to see. Contains NO hint about trap family or variant."""
    task_id: str
    question: str
    data_path: str
    data_dictionary: dict[str, str]
    allowed_flags: list[str]

    @classmethod
    def load(cls, task_dir: Path) -> "Task":
        raw = json.loads((Path(task_dir) / "task.json").read_text())
        raw["data_path"] = str((Path(task_dir) / raw["data_path"]).resolve())
        return cls(**raw)


@dataclass(frozen=True)
class GroundTruth:
    """Hidden answer key. Lives in the PRIVATE directory, never mounted into the agent sandbox."""
    task_id: str
    family: str
    variant: str
    traps_present: list[str]
    effect_exists: bool
    true_direction: str
    twin_id: str
    params: dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "GroundTruth":
        return cls(**json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class Submission:
    """The structured research output.

    GRADED fields: effect_exists, direction, confidence, flags.
    RECORDED-ONLY fields (stored for qualitative taxonomy work, not scored): hypothesis,
    mechanism, predicted_direction, experiments, revisions.
    """
    task_id: str
    effect_exists: bool
    direction: str
    confidence: float          # probability that your own conclusion is correct
    flags: list[str]
    hypothesis: str = ""
    mechanism: str = ""
    predicted_direction: str = "none"
    experiments: list[dict] = field(default_factory=list)
    revisions: list[str] = field(default_factory=list)


def parse_submission(raw: dict[str, Any], expected_task_id: str) -> Submission:
    """Validate untrusted agent output. Raises SubmissionError on any violation."""
    if not isinstance(raw, dict):
        raise SubmissionError("submission must be a JSON object")
    if raw.get("task_id") != expected_task_id:
        raise SubmissionError("task_id mismatch")
    if not isinstance(raw.get("effect_exists"), bool):
        raise SubmissionError("effect_exists must be a boolean")
    direction = raw.get("direction")
    if direction not in DIRECTIONS:
        raise SubmissionError(f"direction must be one of {DIRECTIONS}")
    if raw["effect_exists"] is False and direction != "none":
        raise SubmissionError("direction must be 'none' when effect_exists is false")
    if raw["effect_exists"] is True and direction == "none":
        raise SubmissionError("direction must be positive/negative when effect_exists is true")
    conf = raw.get("confidence")
    if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not 0.0 <= conf <= 1.0:
        raise SubmissionError("confidence must be a number in [0, 1]")
    flags = raw.get("flags", [])
    if not isinstance(flags, list) or any(f not in ALL_FLAG_CODES for f in flags):
        raise SubmissionError(f"flags must be a list drawn from {ALL_FLAG_CODES}")
    pred = raw.get("predicted_direction", "none")
    if pred not in DIRECTIONS:
        raise SubmissionError(f"predicted_direction must be one of {DIRECTIONS}")
    return Submission(
        task_id=expected_task_id,
        effect_exists=raw["effect_exists"],
        direction=direction,
        confidence=float(conf),
        flags=sorted(set(flags)),
        hypothesis=str(raw.get("hypothesis", "")),
        mechanism=str(raw.get("mechanism", "")),
        predicted_direction=pred,
        experiments=list(raw.get("experiments", [])),
        revisions=[str(r) for r in raw.get("revisions", [])],
    )
