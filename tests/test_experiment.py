import argparse
import tempfile
from pathlib import Path

from failurelab.cli import cmd_generate
from failurelab.experiment import plan, run_llm_experiment, select_tasks
from failurelab.llm import ScriptedClient, tool_turn
from failurelab.schema import GroundTruth
from failurelab.scripted_policies import reference_policy

_D: list[Path] = []


def _tasks() -> Path:
    if not _D:
        d = Path(tempfile.mkdtemp())
        cmd_generate(argparse.Namespace(n=18, seed=0, salt="e", out=str(d / "tasks")))
        _D.append(d / "tasks")
    return _D[0]


def test_select_tasks_returns_whole_pairs_covering_every_family():
    priv = _tasks() / "private"
    ids = select_tasks(priv, 12, seed=0)
    assert len(ids) == 12
    truths = {i: GroundTruth.load(priv / f"{i}.json") for i in ids}
    assert all(t.twin_id in ids for t in truths.values())                       # twins travel together
    assert {t.family for t in truths.values()} == {"leakage", "survivorship", "multiple_testing"}
    assert select_tasks(priv, 12, seed=0) == ids and select_tasks(priv, 12, seed=1) != ids
    assert select_tasks(priv, None) is None


def test_plan_counts_runs_and_the_call_ceiling():
    p = plan(_tasks(), ["none", "checklist"], reps=3, max_tasks=12, max_turns=10)
    assert p["tasks"] == 12 and p["task_runs"] == 12 * 3 * 2 and p["max_model_calls"] == 12 * 3 * 2 * 10


def test_experiment_writes_scores_transcripts_and_resumes(tmp_path):
    client = ScriptedClient("scripted-careful", reference_policy("HeuristicCarefulAgent"))
    kw = dict(conditions=["none", "checklist"], reps=2, max_tasks=6, sandbox_kwargs={"allow_failurelab": True})
    df = run_llm_experiment(client, _tasks(), tmp_path / "out", **kw)
    assert len(df) == 6 * 2 * 2 and set(df.rep) == {0, 1} and set(df.condition) == {"none", "checklist"}
    assert (df.status == "submitted").all() and df.valid.all() and df.conclusion_correct.all()
    assert (tmp_path / "out" / "scores.jsonl").exists()
    assert len(list((tmp_path / "out").glob("rep*/*/transcripts/*.json"))) == len(df)
    calls = client.calls
    df2 = run_llm_experiment(client, _tasks(), tmp_path / "out", **kw)       # resume: no new model calls
    assert client.calls == calls and len(df2) == len(df)
