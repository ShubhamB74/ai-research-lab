"""PHASE 3 EXIT CRITERION: a scripted model run through the full harness (prompt -> tool loop ->
sandboxed Python -> submit_answer -> validation -> grading) reproduces the scores of the same analysis
run directly. This validates the plumbing; it says nothing about any LLM."""
import argparse
import json
import tempfile
from pathlib import Path

from failurelab.agents import HeuristicCarefulAgent, NaiveAgent
from failurelab.agents.llm_agent import LLMAgent
from failurelab.cli import cmd_generate
from failurelab.llm import ScriptedClient
from failurelab.runner import run_agent
from failurelab.scripted_policies import reference_policy

FIELDS = ("valid", "conclusion_correct", "fooled", "right_for_wrong_reason", "tp", "fp", "fn", "brier",
          "mechanism", "difficulty")


def test_scripted_harness_matches_direct_baselines_on_every_family(tmp_path):
    d = Path(tempfile.mkdtemp())
    cmd_generate(argparse.Namespace(n=6, seed=300, salt="h", out=str(d)))   # 36 tasks, all 3 families
    pub, priv = d / "public", d / "private"
    for cls in (NaiveAgent, HeuristicCarefulAgent):
        direct = {s.task_id: s for s in run_agent(cls(), pub, priv, tmp_path / "direct")}
        agent = LLMAgent(ScriptedClient(f"scripted-{cls.__name__}", reference_policy(cls.__name__)), "none",
                         sandbox_kwargs={"allow_failurelab": True},
                         transcript_dir=tmp_path / f"tr_{cls.__name__}")
        harness = {s.task_id: s for s in run_agent(agent, pub, priv, tmp_path / "harness")}
        assert direct.keys() == harness.keys() and len(direct) == 36
        for tid in direct:
            for f in FIELDS:
                assert getattr(direct[tid], f) == getattr(harness[tid], f), (cls.__name__, tid, f)
        transcripts = list((tmp_path / f"tr_{cls.__name__}").glob("*.json"))
        assert len(transcripts) == 36
        for t in transcripts:
            tr = json.loads(t.read_text())
            assert tr["status"] == "submitted" and tr["preregistered_first"] and tr["n_python_calls"] == 1
