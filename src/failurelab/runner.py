"""Run an agent over every public task, validate, grade. Invalid output is a scored failure."""
from __future__ import annotations

import json
from pathlib import Path

from .agents.base import Agent
from .grader import TaskScore, grade, grade_invalid
from .schema import GroundTruth, Task, parse_submission


def run_agent(agent: Agent, public_root: Path, private_root: Path, runs_root: Path) -> list[TaskScore]:
    public_root, private_root, runs_root = map(Path, (public_root, private_root, runs_root))
    out_dir = runs_root / agent.name.replace("|", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    scores: list[TaskScore] = []
    for tdir in sorted(p for p in public_root.iterdir() if (p / "task.json").exists()):
        task = Task.load(tdir)
        truth = GroundTruth.load(private_root / f"{task.task_id}.json")
        try:
            raw = agent.run(task)
            (out_dir / f"{task.task_id}.json").write_text(json.dumps(raw, indent=2, default=str))
            sub = parse_submission(raw, task.task_id)
            scores.append(grade(sub, truth))
        except Exception as exc:  # noqa: BLE001  (agent code is untrusted; any failure is a score)
            (out_dir / f"{task.task_id}.error.txt").write_text(repr(exc))
            scores.append(grade_invalid(truth))
    return scores
