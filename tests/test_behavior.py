import argparse
import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from failurelab.behavior import (CUE_PATTERNS, behavior_table, cue_inspected, diagnose, outcome_class, summarize)
from failurelab.cli import cmd_generate
from failurelab.experiment import run_llm_experiment
from failurelab.inference import Study
from failurelab.llm import ScriptedClient
from failurelab.runner import safe_name
from failurelab.schema import GroundTruth
from failurelab.scripted_policies import reference_policy
from failurelab.study import attach_pairs, load_runs
from synth import behavior_frame

POSITIVE = {
    "leakage": ["m = df['signal_asof'] > df['date']", "df.groupby('asset').ret.shift(-1)", "x.rolling(5, center=True).mean()",
                "# does the signal look ahead?  check for look-ahead", "rebuild the signal from raw returns", "future = df.ret.shift(-2)"],
    "survivorship": ["df.groupby('asset').size().nunique()", "df['status'].value_counts()", "attrition = 1 - last / first",
                     "(df.date < df.index_add_date).sum()", "print('is the panel balanced?')", "df.groupby('asset').date.count()"],
    "multiple_testing": ["lib = df['library_size'].iloc[0]", "from statsmodels.stats.multitest import multipletests", "thr = 0.05 / 200  # Bonferroni",
                         "cols = [c for c in df.columns if c.startswith('s')]", "n_signals = len(cols)", "apply benjamini hochberg FDR"],
}
NEGATIVE = {  # ordinary analysis code that does not examine the cue for that family
    "leakage": ["df['signal'].corr(df['fwd_ret_1d'])", "scipy.stats.pearsonr(df.signal, df.fwd_ret_1d)", "df.describe()", "print(df.head())"],
    "survivorship": ["df['ret'].mean()", "stats.ttest_1samp(df.ret, 0)", "print(df.head())", "df.ret.std()"],
    "multiple_testing": ["np.corrcoef(df['s000'], df['fwd_ret'])", "df['signal'].corr(df['fwd_ret'])", "print(df.head())", "df.fwd_ret.std()"],
}


def test_cue_detector_on_labelled_code_snippets():
    assert set(POSITIVE) == set(NEGATIVE) == set(CUE_PATTERNS)
    for fam, snippets in POSITIVE.items():
        for s in snippets:
            assert cue_inspected(fam, s), (fam, s)
    for fam, snippets in NEGATIVE.items():
        for s in snippets:
            assert not cue_inspected(fam, s), (fam, s)


def test_diagnosis_truth_table():
    assert diagnose("trapped", True, True, 1, 0) == "detected"
    assert diagnose("trapped", True, False, 1, 0) == "flagged_without_looking"
    assert diagnose("trapped", True, True, 0, 0) == "saw_not_flagged"
    assert diagnose("trapped", True, False, 0, 0) == "never_looked"
    assert diagnose("clean", True, False, 0, 1) == "false_alarm" and diagnose("clean", True, True, 0, 0) == "clean_ok"
    assert diagnose("trapped", False, True, 1, 0) == "invalid"


def test_outcome_classes():
    base = dict(valid=True, variant="trapped", fp=0, fn=0, fooled=False, conclusion_correct=True)
    oc = lambda **kw: outcome_class(pd.Series({**base, **kw}))
    assert oc() == "caught" and oc(fn=1, fooled=True, conclusion_correct=False) == "fooled_missed"
    assert oc(fn=1) == "right_wrong_reason" and oc(fn=1, conclusion_correct=False) == "missed_wrong"
    assert oc(variant="clean", fp=1) == "false_alarm" and oc(variant="clean") == "clean_ok" and oc(valid=False) == "invalid"


def test_behavior_table_reads_transcripts_and_diagnoses_each_case(tmp_path):
    priv, run = tmp_path / "private", tmp_path / "run"
    priv.mkdir()
    cases = [  # task, variant, code, tp, fp, valid, expected
        ("t1", "trapped", "df['signal_asof']", 1, 0, True, "detected"),
        ("t2", "trapped", "df.corr()", 0, 0, True, "never_looked"),
        ("t3", "trapped", "x.shift(-1)", 0, 0, True, "saw_not_flagged"),
        ("t4", "trapped", "print(1)", 1, 0, True, "flagged_without_looking"),
        ("t5", "clean", "print(1)", 0, 1, True, "false_alarm"),
        ("t6", "clean", "print(1)", 0, 0, True, "clean_ok"),
        ("t7", "trapped", "print(1)", 0, 0, False, "invalid")]
    rows = []
    for tid, variant, code, tp, fp, valid, _ in cases:
        GroundTruth(task_id=tid, family="leakage", variant=variant, traps_present=[], effect_exists=False,
                    true_direction="none", twin_id=tid, params={}).save(priv / f"{tid}.json")
        tdir = run / "rep0" / safe_name("m|none") / "transcripts"
        tdir.mkdir(parents=True, exist_ok=True)
        (tdir / f"{tid}.json").write_text(json.dumps({"cells": [{"code": code, "error": None}, {"code": "pass", "error": "boom"}]}))
        rows.append(dict(agent="m|none", model="m", condition="none", rep=0, task_id=tid, pair_id=tid, variant=variant, valid=valid,
                         tp=tp, fp=fp, fn=1 - tp if variant == "trapped" else 0, fooled=False, conclusion_correct=True,
                         _source=str(run / "scores.jsonl")))
    beh = behavior_table(pd.DataFrame(rows), priv)
    assert list(beh.diagnosis) == [c[-1] for c in cases]
    assert (beh.n_cells == 2).all() and (beh.n_error_cells == 1).all()


def test_decomposition_identity_holds_for_the_summary():
    """recall = P(looked) * P(flag|looked) + (1 - P(looked)) * P(flag|not looked)"""
    beh = behavior_frame(pairs=60, seed=3)
    s = summarize(beh, Study(beh.pair_id, B=1000, seed=1))["m1"]["none"]["overall"]
    lk, fl, fn = (s[k]["point"] for k in ("looked_rate", "flag_given_looked", "flag_given_not_looked"))
    recall = beh[beh.condition == "none"].flagged_correct.mean()
    assert abs(lk * fl + (1 - lk) * fn - recall) < 1e-9
    assert 0.25 < lk < 0.4 and fl > 0.7 and fn < 0.25


def test_pipeline_on_real_harness_transcripts(tmp_path):
    d = Path(tempfile.mkdtemp())
    cmd_generate(argparse.Namespace(n=6, seed=0, salt="b", out=str(d / "tasks")))
    out = tmp_path / "runs" / "scripted-careful"
    run_llm_experiment(ScriptedClient("scripted-careful", reference_policy("HeuristicCarefulAgent")), d / "tasks", out,
                       ["none"], reps=1, max_tasks=6, seed=0, sandbox_kwargs={"allow_failurelab": True})
    df = attach_pairs(load_runs([out]), d / "tasks" / "private")
    beh = behavior_table(df, d / "tasks" / "private")
    assert len(beh) == 6 and (beh.n_cells == 1).all()
    # the scripted agent flags every trap but its code never examines a cue: a 'flagged without looking' pattern
    assert set(beh[beh.variant == "trapped"].diagnosis) == {"flagged_without_looking"}
    assert set(beh[beh.variant == "clean"].diagnosis) == {"clean_ok"}
    assert summarize(beh, Study(beh.pair_id, B=500, seed=1))["scripted-careful"]["none"]["diagnosis_counts"] == {"flagged_without_looking": 3}
