"""Generator interface. A generator must be a PURE FUNCTION of (seed, variant)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..schema import ALL_FLAG_CODES, GroundTruth
from ..taxonomy import Trap, Variant
import json


@dataclass
class GeneratedTask:
    data: pd.DataFrame
    question: str
    data_dictionary: dict[str, str]
    traps_present: list[str]
    effect_exists: bool
    true_direction: str
    params: dict


class Generator(ABC):
    family: Trap

    @abstractmethod
    def generate(self, seed: int, variant: Variant) -> GeneratedTask:
        """Same seed + different variant => same underlying world, with/without the planted flaw.
        Rule: the trap must be DETECTABLE from the public files by a competent analyst."""


def write_task(task_id: str, twin_id: str, family: Trap, variant: Variant,
               gen: GeneratedTask, public_root: Path, private_root: Path) -> None:
    pub = Path(public_root) / task_id
    pub.mkdir(parents=True, exist_ok=True)
    Path(private_root).mkdir(parents=True, exist_ok=True)
    gen.data.to_csv(pub / "data.csv", index=False)
    (pub / "task.json").write_text(json.dumps({
        "task_id": task_id,
        "question": gen.question,
        "data_path": "data.csv",
        "data_dictionary": gen.data_dictionary,
        "allowed_flags": list(ALL_FLAG_CODES),
    }, indent=2))
    GroundTruth(
        task_id=task_id, family=family.value, variant=variant.value,
        traps_present=gen.traps_present, effect_exists=gen.effect_exists,
        true_direction=gen.true_direction, twin_id=twin_id, params=gen.params,
    ).save(Path(private_root) / f"{task_id}.json")
