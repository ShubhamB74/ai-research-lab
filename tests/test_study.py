"""The Phase 4 analysis: pre-registered verdicts, gates, determinism and number provenance."""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from failurelab import prereg
from failurelab.agents.llm_agent import TRIPWIRE
from failurelab.cli import cmd_generate
from failurelab.experiment import run_llm_experiment
from failurelab.llm import ScriptedClient
from failurelab.schema import GroundTruth
from failurelab.scripted_policies import reference_policy
from failurelab.study import Report, StudyError, run_study

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("leakage", "survivorship", "multiple_testing")


def _expit(x):
    return 1 / (1 + np.exp(-x))


def build(tmp, pairs=60, reps=3, models=("m1",), rates=None, seed=0, freeze=True, fill=True, mutate_plan=None):
    """Synthetic experiment in the real on-disk formats: private answer keys + scores.jsonl per model."""
    rng = np.random.default_rng(seed)
    rates = rates or {"none": dict(recall=0.15, ffr=0.05, clean=0.98, trapped=0.5),
                      "checklist": dict(recall=0.70, ffr=0.06, clean=0.98, trapped=0.7)}
    priv = tmp / "tasks" / "private"
    priv.mkdir(parents=True)
    for p in range(pairs):
        for suffix, other, variant in (("a", "b", "trapped"), ("b", "a", "clean")):
            GroundTruth(task_id=f"t{p:03d}{suffix}", family=FAMILIES[p % 3], variant=variant,
                        traps_present=["leakage"] if variant == "trapped" else [], effect_exists=False,
                        true_direction="none", twin_id=f"t{p:03d}{other}",
                        params={"mechanism": "m", "difficulty": "easy"}).save(priv / f"t{p:03d}{suffix}.json")
    u, v = rng.normal(0, 0.5, pairs), rng.normal(0, 0.5, pairs)
    run_dirs = []
    for model in models:
        rows = []
        for cond, r in rates.items():
            for rep in range(reps):
                for p in range(pairs):
                    tp = int(rng.random() < _expit(np.log(r["recall"] / (1 - r["recall"])) + u[p]))
                    ff = int(rng.random() < _expit(np.log(r["ffr"] / (1 - r["ffr"])) + v[p]))
                    ct, cc = rng.random() < r["trapped"], rng.random() < r["clean"]
                    base = dict(agent=f"{model}|{cond}", condition=cond, rep=rep, valid=True, mechanism="m", difficulty="easy",
                                status="submitted", format_retries=0, n_python_calls=3, preregistered_first=True,
                                tripwire_hits=[], input_tokens=1000, output_tokens=200)
                    rows.append({**base, "task_id": f"t{p:03d}a", "family": FAMILIES[p % 3], "variant": "trapped",
                                 "conclusion_correct": bool(ct), "fooled": bool(not ct), "right_for_wrong_reason": bool(ct and not tp),
                                 "tp": tp, "fp": 0, "fn": 1 - tp, "brier": 0.04 if ct else 0.64})
                    rows.append({**base, "task_id": f"t{p:03d}b", "family": FAMILIES[p % 3], "variant": "clean",
                                 "conclusion_correct": bool(cc), "fooled": False, "right_for_wrong_reason": False,
                                 "tp": 0, "fp": ff, "fn": 0, "brier": 0.04 if cc else 0.64})
        d = tmp / "runs" / model.replace(":", "_")
        d.mkdir(parents=True)
        (d / "scores.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\n")
        run_dirs.append(d)
    plan = json.loads((ROOT / "docs" / "prereg.json").read_text())
    if fill:
        plan["design"].update(models=list(models), reps=reps, conditions=list(rates), max_tasks=None)
        plan["analysis"]["bootstrap_B"] = 1000
    if mutate_plan:
        mutate_plan(plan)
    pf, doc, lock = tmp / "prereg.json", tmp / "PREREG.md", tmp / "prereg.lock"
    pf.write_text(json.dumps(plan)); doc.write_text("frozen text")
    if freeze and fill:
        prereg.freeze(pf, doc, lock)
    return dict(tasks_dir=tmp / "tasks", run_dirs=run_dirs, out_dir=tmp / "report", prereg_path=pf, doc_path=doc, lock_path=lock)


def _go(kw, **extra):
    return run_study(**{**kw, **extra})


def test_planted_effect_is_detected_across_seeds_and_never_falsified(tmp_path):
    """True effect: recall 0.15 -> 0.70, false-flag rate 0.05 -> 0.06. At this size the pre-registered rule
    supports H2 most of the time and returns 'inconclusive' when the interval cannot rule out a >10 point rise
    in false flags. It must never call a real effect falsified, and H1 (low baseline recall) holds every time."""
    verdicts = []
    for seed in range(6):
        res = _go(build(tmp_path / f"s{seed}", seed=seed))
        assert res["status"] == "CONFIRMATORY" and not res["reasons"]
        h = res["models"]["m1"]["hypotheses"]
        assert h["H1"]["verdict"] == "supported"
        assert h["H2"]["estimates"]["delta_recall"]["point"] > 0.35
        verdicts.append(h["H2"]["verdict"])
    assert "falsified" not in verdicts and verdicts.count("supported") >= 4, verdicts
    assert (tmp_path / "s0" / "report" / "report.md").exists() and (tmp_path / "s0" / "report" / "provenance.json").exists()


def test_a_null_checklist_effect_is_not_declared_supported(tmp_path):
    same = dict(recall=0.15, ffr=0.05, clean=0.98, trapped=0.5)
    res = _go(build(tmp_path, rates={"none": same, "checklist": dict(same)}))
    assert res["models"]["m1"]["hypotheses"]["H2"]["verdict"] in ("falsified", "inconclusive")


def test_checklist_that_buys_recall_with_false_alarms_is_not_supported(tmp_path):
    rates = {"none": dict(recall=0.15, ffr=0.05, clean=0.98, trapped=0.5),
             "checklist": dict(recall=0.85, ffr=0.50, clean=0.60, trapped=0.7)}
    res = _go(build(tmp_path, rates=rates))
    assert res["models"]["m1"]["hypotheses"]["H2"]["verdict"] == "falsified"


def test_confirmatory_level_is_bonferroni_over_hypotheses_and_models(tmp_path):
    res = _go(build(tmp_path, pairs=20, models=("m1", "m2")))
    assert abs(res["design"]["confirmatory_level"] - (1 - 0.05 / (4 * 2))) < 1e-12


def test_tripwire_flagged_runs_must_be_reviewed(tmp_path):
    kw = build(tmp_path)
    f = kw["run_dirs"][0] / "scores.jsonl"
    rows = [json.loads(x) for x in f.read_text().splitlines()]
    rows[0]["tripwire_hits"] = ["private"]
    f.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    with pytest.raises(StudyError):
        _go(kw)
    res = _go(kw, allow_unreviewed=True)
    assert res["status"] == "NON-CONFIRMATORY" and any("tripwire" in r for r in res["reasons"])
    key = {"model": "m1", "condition": rows[0]["condition"], "rep": rows[0]["rep"], "task_id": rows[0]["task_id"], "reason": "benign"}
    ex = tmp_path / "ex.json"
    ex.write_text(json.dumps([{**key, "action": "keep"}]))
    assert _go(kw, exclusions_path=ex)["status"] == "CONFIRMATORY"
    ex.write_text(json.dumps([{**key, "action": "exclude"}]))
    res = _go(kw, exclusions_path=ex)
    assert res["status"] == "CONFIRMATORY" and res["design"]["excluded_runs"] == 1   # exclusion is not 'incomplete'


def test_incomplete_design_stops_unless_explicitly_allowed(tmp_path):
    kw = build(tmp_path)
    f = kw["run_dirs"][0] / "scores.jsonl"
    f.write_text("\n".join(f.read_text().splitlines()[:-7]) + "\n")
    with pytest.raises(StudyError):
        _go(kw)
    res = _go(kw, allow_incomplete=True)
    assert res["status"] == "NON-CONFIRMATORY" and any("incomplete" in r for r in res["reasons"])


def test_unfrozen_modified_or_incomplete_plans_are_non_confirmatory(tmp_path):
    a = _go(build(tmp_path / "a", pairs=20, freeze=False))
    assert a["status"] == "NON-CONFIRMATORY" and any("never frozen" in r for r in a["reasons"])
    kw = build(tmp_path / "b", pairs=20)
    Path(kw["doc_path"]).write_text("edited after the freeze")
    b = _go(kw)
    assert b["status"] == "NON-CONFIRMATORY" and any("changed after it was frozen" in r for r in b["reasons"])
    c = _go(build(tmp_path / "c", pairs=20, fill=False))
    assert c["status"] == "NON-CONFIRMATORY" and any("incomplete" in r for r in c["reasons"])


def test_too_many_non_submitted_runs_blocks_confirmation(tmp_path):
    kw = build(tmp_path, pairs=30)
    f = kw["run_dirs"][0] / "scores.jsonl"
    rows = [json.loads(x) for x in f.read_text().splitlines()]
    for r in rows[: len(rows) // 4]:
        r["status"] = "no_submission"
    f.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    res = _go(kw)
    assert res["status"] == "NON-CONFIRMATORY" and any("non-submitted" in r for r in res["reasons"])


def test_analysis_is_deterministic_and_provenance_tracks_inputs(tmp_path):
    kw = build(tmp_path, pairs=30)
    _go(kw)
    first = {n: (tmp_path / "report" / n).read_bytes() for n in ("results.json", "report.md", "provenance.json")}
    _go(kw)
    assert first == {n: (tmp_path / "report" / n).read_bytes() for n in first}
    f = kw["run_dirs"][0] / "scores.jsonl"
    rows = [json.loads(x) for x in f.read_text().splitlines()]
    rows[1]["brier"] = 0.5
    f.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    _go(kw)
    assert (tmp_path / "report" / "provenance.json").read_bytes() != first["provenance.json"]


def test_every_number_in_the_report_comes_from_the_computed_results(tmp_path):
    _go(build(tmp_path, pairs=30))
    text = (tmp_path / "report" / "report.md").read_text()
    assert "Status: CONFIRMATORY" in text and "H2" in text
    r = Report()
    r.add("The recall was 0.77 in this run.")
    with pytest.raises(AssertionError):
        r.verify()                                   # a hand-typed number is rejected
    r2 = Report()
    r2.add(f"The recall was {r2.n(0.77)} in this run, over {r2.i(12)} pairs. Model `x-5` is ignored.")
    r2.verify()


def test_pipeline_on_real_harness_output_with_scripted_models(tmp_path):
    """Real scores.jsonl from run_llm_experiment (sandbox, tool loop, grading), two scripted 'models'."""
    d = Path(tempfile.mkdtemp())
    cmd_generate(argparse.Namespace(n=6, seed=0, salt="s", out=str(d / "tasks")))
    runs = []
    for model, cls in (("scripted-careful", "HeuristicCarefulAgent"), ("scripted-naive", "NaiveAgent")):
        client = ScriptedClient(model, reference_policy(cls))
        out = tmp_path / "runs" / model
        run_llm_experiment(client, d / "tasks", out, ["none", "checklist"], reps=2, max_tasks=6, seed=0,
                           sandbox_kwargs={"allow_failurelab": True})
        runs.append(out)
    plan = json.loads((ROOT / "docs" / "prereg.json").read_text())
    plan["design"].update(models=["scripted-careful", "scripted-naive"], reps=2, conditions=["none", "checklist"], max_tasks=6)
    plan["analysis"]["bootstrap_B"] = 1000
    pf, doc, lock = tmp_path / "prereg.json", tmp_path / "P.md", tmp_path / "P.lock"
    pf.write_text(json.dumps(plan)); doc.write_text("x")
    prereg.freeze(pf, doc, lock)
    kw = dict(tasks_dir=d / "tasks", run_dirs=runs, out_dir=tmp_path / "report", prereg_path=pf, doc_path=doc, lock_path=lock)
    with pytest.raises(StudyError):                  # the careful policy's code imports failurelab: tripwire fires
        run_study(**kw)
    res = run_study(**kw, allow_unreviewed=True)
    c = res["models"]["scripted-careful"]["conditions"]["none"]["metrics"]
    n = res["models"]["scripted-naive"]["conditions"]["none"]["metrics"]
    assert c["recall"]["point"] == 1.0 and c["false_flag_rate"]["point"] == 0.0 and c["trapped_accuracy"]["point"] == 1.0
    assert n["recall"]["point"] == 0.0 and n["fooled_rate"]["point"] > 0
    assert res["design"]["pairs"] == 3 and res["design"]["rows"] == 2 * 2 * 2 * 6
    assert res["models"]["scripted-careful"]["hypotheses"]["H1"]["verdict"] == "falsified"   # recall 1.0 >> 0.5
