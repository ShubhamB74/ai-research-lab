"""Agent code is untrusted: crashes, junk and mismatched IDs must become scored failures."""
import argparse

from failurelab.agents.base import Agent
from failurelab.cli import cmd_generate
from failurelab.runner import run_agent


class Crash(Agent):
    name = "crash"

    def run(self, task):
        raise RuntimeError("boom")


class Junk(Agent):
    name = "junk"

    def run(self, task):
        return "not even json"


class WrongId(Agent):
    name = "wrongid"

    def run(self, task):
        return {"task_id": "someone-else", "effect_exists": False, "direction": "none",
                "confidence": 0.5, "flags": []}


class Inconsistent(Agent):
    name = "inconsistent"

    def run(self, task):
        return {"task_id": task.task_id, "effect_exists": False, "direction": "positive",
                "confidence": 0.5, "flags": []}


def test_bad_agents_are_scored_as_invalid_not_crashed(tmp_path):
    cmd_generate(argparse.Namespace(n=2, seed=0, salt="t", out=str(tmp_path / "t")))
    pub, priv = tmp_path / "t" / "public", tmp_path / "t" / "private"
    for agent in (Crash(), Junk(), WrongId(), Inconsistent()):
        scores = run_agent(agent, pub, priv, tmp_path / "runs")
        assert len(scores) == 4
        assert all((not s.valid) and (not s.conclusion_correct) and s.brier == 1.0 for s in scores)
        assert sum(s.fn for s in scores) == 2  # missed flags still counted on the 2 trapped tasks
