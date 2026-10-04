"""Harness behaviour with scripted 'models' (no API). Tests plumbing, not LLM behaviour."""
import argparse
import json
import tempfile
from pathlib import Path

import pytest

from failurelab.agents.llm_agent import LLMAgent
from failurelab.cli import cmd_generate
from failurelab.llm import (BudgetedClient, BudgetExceeded, ModelClient, RunAborted, ScriptedClient, text_turn,
                            tool_turn)
from failurelab.runner import run_agent
from failurelab.schema import Task

PREREG = ("preregister", {"hypothesis": "h", "mechanism": "m", "predicted_direction": "none", "falsifier": "f"})
GOOD = ("submit_answer", {"effect_exists": False, "direction": "none", "confidence": 0.7, "flags": ["leakage"]})
_DIR: list[Path] = []


def _tasks() -> Path:
    if not _DIR:
        d = Path(tempfile.mkdtemp())
        cmd_generate(argparse.Namespace(n=2, seed=0, salt="a", out=str(d)))
        _DIR.append(d)
    return _DIR[0]


def _task() -> Task:
    d = _tasks() / "public"
    return Task.load(sorted(d.iterdir())[0])


def _steps(*turns):
    """policy that plays the given turns in order, then repeats the last one"""
    def policy(messages, tools):
        i = sum(1 for m in messages if m["role"] == "assistant")
        return turns[min(i, len(turns) - 1)]
    return policy


def _agent(tmp_path, policy, **kw):
    return LLMAgent(ScriptedClient("scripted", policy), "none", transcript_dir=tmp_path / "tr", **kw)


def _transcript(tmp_path, task):
    return json.loads((tmp_path / "tr" / f"{task.task_id}.json").read_text())


def test_happy_path_records_a_complete_transcript(tmp_path):
    task = _task()
    out = _agent(tmp_path, _steps(tool_turn(PREREG), tool_turn(("run_python", {"code": "print(1+1)"})),
                                  tool_turn(GOOD))).run(task)
    assert out["task_id"] == task.task_id and out["flags"] == ["leakage"] and out["hypothesis"] == "h"
    tr = _transcript(tmp_path, task)
    assert tr["status"] == "submitted" and tr["preregistered_first"] and tr["n_python_calls"] == 1
    assert tr["cells"][0]["stdout"].strip() == "2" and tr["model_calls"] == 3
    assert tr["messages"][0]["role"] == "user" and tr["system_prompt"] and tr["condition"] == "none"


def test_running_code_before_preregistering_is_recorded(tmp_path):
    task = _task()
    _agent(tmp_path, _steps(tool_turn(("run_python", {"code": "print(1)"})), tool_turn(PREREG), tool_turn(GOOD))).run(task)
    assert _transcript(tmp_path, task)["preregistered_first"] is False


def test_bad_format_is_fed_back_and_can_be_fixed(tmp_path):
    task = _task()
    bad = ("submit_answer", {"effect_exists": False, "direction": "up", "confidence": 0.5, "flags": []})
    out = _agent(tmp_path, _steps(tool_turn(PREREG), tool_turn(bad), tool_turn(GOOD))).run(task)
    tr = _transcript(tmp_path, task)
    assert tr["status"] == "submitted" and tr["format_retries"] == 1 and out["flags"] == ["leakage"]


def test_persistently_invalid_format_is_scored_invalid(tmp_path):
    task = _task()
    bad = ("submit_answer", {"effect_exists": True, "direction": "none", "confidence": 0.5, "flags": []})
    out = _agent(tmp_path, _steps(tool_turn(PREREG), tool_turn(bad)), max_format_retries=2).run(task)
    assert _transcript(tmp_path, task)["status"] == "invalid_format"
    from failurelab.schema import SubmissionError, parse_submission
    with pytest.raises(SubmissionError):
        parse_submission(out, task.task_id)


def test_never_submitting_is_a_scored_failure_not_a_crash(tmp_path):
    d = _tasks()
    agent = _agent(tmp_path, _steps(text_turn("I think the answer is no effect.")))
    scores = run_agent(agent, d / "public", d / "private", tmp_path / "runs")
    assert scores and all((not s.valid) and not s.conclusion_correct for s in scores)
    assert all(json.loads(f.read_text())["status"] == "no_submission" for f in (tmp_path / "tr").glob("*.json"))


def test_turn_budget_is_enforced(tmp_path):
    task = _task()
    _agent(tmp_path, _steps(tool_turn(("run_python", {"code": "pass"}))), max_turns=3).run(task)
    tr = _transcript(tmp_path, task)
    assert tr["status"] == "max_turns" and tr["model_calls"] == 3


def test_tripwire_flags_code_that_pokes_at_benchmark_files(tmp_path):
    task = _task()
    code = "import os; print(os.listdir('../private'))"
    _agent(tmp_path, _steps(tool_turn(PREREG), tool_turn(("run_python", {"code": code})), tool_turn(GOOD))).run(task)
    tr = _transcript(tmp_path, task)
    assert tr["tripwire_hits"] and "private" in tr["tripwire_hits"]
    assert "private" not in tr["cells"][0]["stdout"]      # and the sandbox shows nothing there


def test_completed_tasks_are_resumed_without_calling_the_model(tmp_path):
    task = _task()
    client = ScriptedClient("scripted", _steps(tool_turn(PREREG), tool_turn(GOOD)))
    a = LLMAgent(client, "none", transcript_dir=tmp_path / "tr")
    first = a.run(task)
    calls = client.calls
    second = LLMAgent(client, "none", transcript_dir=tmp_path / "tr").run(task)
    assert client.calls == calls and second == first


class _Boom(ModelClient):
    name = "boom"

    def complete(self, system, messages, tools):
        raise RunAborted("API outage")


def test_infrastructure_failures_abort_the_run_instead_of_being_scored(tmp_path):
    d = _tasks()
    agent = LLMAgent(_Boom(), "none", transcript_dir=tmp_path / "tr")
    with pytest.raises(RunAborted):
        run_agent(agent, d / "public", d / "private", tmp_path / "runs")
    saved = [json.loads(f.read_text()) for f in (tmp_path / "tr").glob("*.json")]
    assert saved and all(s["status"] == "aborted" for s in saved)
    # an aborted transcript is NOT final: a later run redoes the task
    good = LLMAgent(ScriptedClient("scripted", _steps(tool_turn(PREREG), tool_turn(GOOD))), "none",
                    transcript_dir=tmp_path / "tr")
    assert good.run(_task())["flags"] == ["leakage"]


def test_budget_cap_stops_the_run(tmp_path):
    client = BudgetedClient(ScriptedClient("s", _steps(tool_turn(("run_python", {"code": "pass"})))), max_calls=2)
    with pytest.raises(BudgetExceeded):
        _agent_with(client, tmp_path).run(_task())


def _agent_with(client, tmp_path):
    return LLMAgent(client, "none", transcript_dir=tmp_path / "tr", max_turns=10)
