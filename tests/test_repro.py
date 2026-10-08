import argparse
import json
import shutil
import tempfile
from pathlib import Path

from failurelab import prereg
from failurelab.cli import cmd_generate
from failurelab.experiment import run_llm_experiment
from failurelab.llm import ScriptedClient
from failurelab.repro import (analysis_reproduces, regenerate_and_compare, regrade, verify_manifest, write_manifest)
from failurelab.scripted_policies import reference_policy
from failurelab.study import run_study

_CACHE: dict = {}


def _world():
    """One scripted experiment on a small task set, shared by the tests (they copy what they tamper with)."""
    if not _CACHE:
        d = Path(tempfile.mkdtemp())
        cmd_generate(argparse.Namespace(n=2, seed=0, salt="r", out=str(d / "tasks")))
        out = d / "runs" / "scripted-careful"
        run_llm_experiment(ScriptedClient("scripted-careful", reference_policy("HeuristicCarefulAgent")), d / "tasks", out,
                           ["none"], reps=1, max_tasks=None, sandbox_kwargs={"allow_failurelab": True})
        _CACHE.update(root=d, tasks=d / "tasks", run=out)
    return _CACHE


def _copy(tmp_path):
    w = _world()
    t, r = tmp_path / "tasks", tmp_path / "run"
    shutil.copytree(w["tasks"], t)
    shutil.copytree(w["run"], r)
    return t, r


def test_manifest_detects_any_change_to_the_task_set(tmp_path):
    tasks, _ = _copy(tmp_path)
    m = tmp_path / "manifest.json"
    write_manifest(tasks, m)
    assert verify_manifest(tasks, m) == []
    victim = next((tasks / "private").glob("*.json"))
    victim.write_text(victim.read_text().replace("true", "false", 1) if "true" in victim.read_text() else victim.read_text() + " ")
    assert any("changed" in p for p in verify_manifest(tasks, m))
    next((tasks / "public").iterdir()).joinpath("data.csv").unlink()
    assert any("missing" in p for p in verify_manifest(tasks, m))


def test_task_regeneration_matches_on_this_machine_and_flags_a_different_plan(tmp_path):
    m = tmp_path / "manifest.json"
    write_manifest(_world()["tasks"], m)
    plan = {"design": {"task_set": {"n_per_family": 2, "seed": 0, "salt": "r"}}}
    assert regenerate_and_compare(plan, m) == []
    plan["design"]["task_set"]["salt"] = "other"
    assert regenerate_and_compare(plan, m)


def test_regrade_matches_stored_scores_and_catches_tampering(tmp_path):
    tasks, run = _copy(tmp_path)
    assert regrade([run], tasks) == []
    rows = [json.loads(x) for x in (run / "scores.jsonl").read_text().splitlines()]
    rows[0]["conclusion_correct"] = not rows[0]["conclusion_correct"]
    (run / "scores.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert any("conclusion_correct" in p for p in regrade([run], tasks))
    rows[0]["conclusion_correct"] = not rows[0]["conclusion_correct"]
    (run / "scores.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert regrade([run], tasks) == []
    raw = next((run / "rep0").glob("*/t*.json"))                       # tamper with a stored raw submission instead
    data = json.loads(raw.read_text())
    data["flags"] = [] if data["flags"] else ["leakage"]
    raw.write_text(json.dumps(data))
    assert regrade([run], tasks)


def test_analysis_reproduces_byte_for_byte_and_detects_changed_inputs(tmp_path):
    tasks, run = _copy(tmp_path)
    plan = json.loads((Path(__file__).resolve().parents[1] / "docs" / "prereg.json").read_text())
    plan["analysis"]["bootstrap_B"] = 1000
    pf, doc, lock = tmp_path / "p.json", tmp_path / "P.md", tmp_path / "P.lock"
    pf.write_text(json.dumps(plan)); doc.write_text("x")
    kw = dict(tasks_dir=tasks, run_dirs=[run], prereg_path=pf, doc_path=doc, lock_path=lock, allow_unreviewed=True)
    run_study(out_dir=tmp_path / "report", **kw)
    assert analysis_reproduces(tmp_path / "report", **kw) == []
    rows = [json.loads(x) for x in (run / "scores.jsonl").read_text().splitlines()]
    rows[0]["brier"] = 0.5
    (run / "scores.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert analysis_reproduces(tmp_path / "report", **kw)
