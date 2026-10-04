"""Validates the BENCHMARK (all three trap families) before any LLM touches it."""
import argparse
import tempfile
from pathlib import Path

from failurelab.agents import (AlwaysFlagAgent, HeuristicCarefulAgent, NaiveAgent, OracleAgent,
                               SanityCheckAgent)
from failurelab.cli import cmd_generate
from failurelab.runner import run_agent
from failurelab.scorecard import to_frame

_TASKS: dict[int, Path] = {}


def _tasks(n: int) -> Path:
    """Generate once per n (18 covers every cell in every family at least twice; 36 four+ times)."""
    if n not in _TASKS:
        d = Path(tempfile.mkdtemp())
        cmd_generate(argparse.Namespace(n=n, seed=100, salt="test", out=str(d / "tasks")))
        _TASKS[n] = d
    return _TASKS[n]


def _run(agent_factory, n=18):
    d = _tasks(n)
    pub, priv = d / "tasks" / "public", d / "tasks" / "private"
    agent = agent_factory(priv)
    return to_frame({agent.name: run_agent(agent, pub, priv, d / "runs")})


def _recall(g):
    return g.tp.sum() / (g.tp.sum() + g.fn.sum())


def test_oracle_is_perfect_in_every_family():
    df = _run(lambda p: OracleAgent(p))
    assert set(df.family) == {"leakage", "survivorship", "multiple_testing"}
    assert df.valid.all() and df.conclusion_correct.all() and (df.fp == 0).all() and (df.fn == 0).all()


def test_naive_is_fooled_in_every_family_but_fine_on_clean_twins():
    df = _run(lambda p: NaiveAgent())
    for fam, g in df[df.variant == "trapped"].groupby("family"):
        assert g.fooled.mean() > 0.2, fam
    assert df[df.variant == "clean"].conclusion_correct.mean() > 0.95


def test_always_flag_gets_full_recall_but_all_false_alarms():
    df = _run(lambda p: AlwaysFlagAgent())
    assert df[df.variant == "trapped"].fn.sum() == 0
    assert (df[df.variant == "clean"].fp > 0).all()


def test_every_cell_in_every_family_is_solvable_from_public_files():
    """Solvability proof: the careful agent clears all 21 cells (9 leakage + 6 + 6) with no false alarms."""
    df = _run(lambda p: HeuristicCarefulAgent())
    assert df.valid.all() and df.conclusion_correct.all()
    trapped = df[df.variant == "trapped"]
    assert trapped.fn.sum() == 0 and df[df.variant == "clean"].fp.sum() == 0
    assert len(set(zip(trapped.family, trapped.mechanism, trapped.difficulty))) == 21


def test_leakage_difficulty_ladder_discriminates_detection_skill():
    df = _run(lambda p: SanityCheckAgent(), n=36)
    trapped = df[(df.variant == "trapped") & (df.family == "leakage")]
    recall = {d: _recall(g) for d, g in trapped.groupby("difficulty")}
    assert recall["easy"] >= 0.9 and recall["hard"] <= 0.25
    assert recall["easy"] > recall["medium"] >= recall["hard"]
    assert df[df.variant == "clean"].fp.sum() == 0


def test_cheap_checks_find_visible_cues_but_miss_subtle_ones():
    df = _run(lambda p: SanityCheckAgent(), n=36)
    t = df[df.variant == "trapped"]
    cell = {k: _recall(g) for k, g in t.assign(cell=t.mechanism + "/" + t.difficulty).groupby("cell")}
    # survivorship: explicit cues (status column, id gaps) are caught; a balanced panel alone is not
    assert cell["survivor_only_panel/easy"] == 1.0 and cell["survivor_only_panel/medium"] == 1.0
    assert cell["survivor_only_panel/hard"] == 0.0
    assert all(cell[f"backfilled_membership/{d}"] == 1.0 for d in ("easy", "medium", "hard"))
    # multiple testing: visible column counts are caught; a library_size metadata field is not read
    assert all(cell[f"visible_candidates/{d}"] == 1.0 for d in ("easy", "medium", "hard"))
    assert all(cell[f"hidden_winner/{d}"] == 0.0 for d in ("easy", "medium", "hard"))
    assert df[df.variant == "clean"].fp.sum() == 0


def test_always_say_none_is_not_a_winning_strategy():
    """Clean twins carry real effects, so an agent that never claims an effect cannot ace the benchmark."""
    import json
    truths = [json.loads(f.read_text()) for f in (_tasks(18) / "tasks" / "private").glob("*.json")]
    clean = [t for t in truths if t["variant"] == "clean"]
    share_effect = sum(t["effect_exists"] for t in clean) / len(clean)
    assert 0.3 < share_effect < 0.8, share_effect        # a constant 'no effect' answer is wrong on this share
    for fam in ("leakage", "survivorship", "multiple_testing"):
        fam_clean = [t for t in clean if t["family"] == fam]
        assert any(t["effect_exists"] for t in fam_clean) and any(not t["effect_exists"] for t in fam_clean), fam
