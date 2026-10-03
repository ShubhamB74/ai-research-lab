"""Validates the BENCHMARK before any LLM touches it."""
import argparse

from failurelab.agents import (AlwaysFlagAgent, HeuristicCarefulAgent, NaiveAgent, OracleAgent,
                               SanityCheckAgent)
from failurelab.cli import cmd_generate
from failurelab.runner import run_agent
from failurelab.scorecard import to_frame


def _run(tmp_path, agent_factory, n=18):
    cmd_generate(argparse.Namespace(n=n, seed=100, salt="test", out=str(tmp_path / "tasks")))
    pub, priv = tmp_path / "tasks" / "public", tmp_path / "tasks" / "private"
    agent = agent_factory(priv)
    return to_frame({agent.name: run_agent(agent, pub, priv, tmp_path / "runs")})


def test_oracle_is_perfect(tmp_path):
    df = _run(tmp_path, lambda p: OracleAgent(p))
    assert df.valid.all() and df.conclusion_correct.all() and (df.fp == 0).all() and (df.fn == 0).all()


def test_naive_is_fooled_by_leakage_but_ok_on_clean(tmp_path):
    df = _run(tmp_path, lambda p: NaiveAgent())
    trapped, clean = df[df.variant == "trapped"], df[df.variant == "clean"]
    assert trapped.fooled.mean() > 0.3
    assert clean.conclusion_correct.mean() > 0.9


def test_always_flag_gets_full_recall_but_all_false_alarms(tmp_path):
    df = _run(tmp_path, lambda p: AlwaysFlagAgent())
    assert df[df.variant == "trapped"].fn.sum() == 0
    assert (df[df.variant == "clean"].fp > 0).all()


def test_every_cell_is_solvable_from_public_files(tmp_path):
    """Solvability proof: the careful agent clears all 9 mechanism x difficulty cells (2 seeds each)."""
    df = _run(tmp_path, lambda p: HeuristicCarefulAgent())
    assert df.conclusion_correct.mean() == 1.0
    trapped = df[df.variant == "trapped"]
    assert trapped.fn.sum() == 0
    assert df[df.variant == "clean"].fp.sum() == 0
    assert set(zip(trapped.mechanism, trapped.difficulty)).__len__() == 9


def test_difficulty_ladder_discriminates_detection_skill(tmp_path):
    """A 'too good to be true' check catches easy leaks and misses the subtle ones."""
    df = _run(tmp_path, lambda p: SanityCheckAgent(), n=36)
    trapped = df[df.variant == "trapped"]
    recall = {d: g.tp.sum() / (g.tp.sum() + g.fn.sum()) for d, g in trapped.groupby("difficulty")}
    assert recall["easy"] >= 0.9
    assert recall["hard"] <= 0.25
    assert recall["easy"] > recall["medium"] >= recall["hard"]
    assert df[df.variant == "clean"].fp.sum() == 0   # and it does not raise false alarms
