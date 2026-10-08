import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from failurelab.cli import cmd_generate
from failurelab.coding import (SEED_CODES, SHEET_COLUMNS, cohens_kappa, detector_validity, kappa_table, parse_codes,
                               prevalence_table, sample_for_coding, validate_sheet, write_coding_sheet)
from failurelab.experiment import run_llm_experiment
from failurelab.llm import ScriptedClient
from failurelab.scripted_policies import reference_policy
from failurelab.study import attach_pairs, load_runs


def _scores(n_pairs=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_pairs):
        tp = int(rng.random() < 0.3)
        for variant, suffix in (("trapped", "a"), ("clean", "b")):
            fp = int(variant == "clean" and rng.random() < 0.1)
            rows.append(dict(model="m", condition="none", rep=0, task_id=f"t{p}{suffix}", pair_id=f"t{p}", family="leakage" if p % 2 else "survivorship",
                             variant=variant, valid=True, tp=tp if variant == "trapped" else 0, fp=fp,
                             fn=(1 - tp) if variant == "trapped" else 0, fooled=variant == "trapped" and not tp,
                             conclusion_correct=bool(tp)))
    return pd.DataFrame(rows)


def test_sampling_is_stratified_weighted_deterministic():
    df = _scores()
    a = sample_for_coding(df, per_stratum=4, seed=1)
    assert (a.groupby("stratum").size() <= 4).all()
    w = a.groupby("stratum").apply(lambda g: g.weight.sum())            # weights sum to the stratum population
    pop = df.assign(outcome_class=df.apply(__import__("failurelab.behavior", fromlist=["x"]).outcome_class, axis=1))
    pop["stratum"] = pop[["model", "condition", "family", "outcome_class"]].astype(str).agg("/".join, axis=1)
    assert all(abs(w[s] - n) < 1e-9 for s, n in pop.groupby("stratum").size().items())
    assert a.equals(sample_for_coding(df, per_stratum=4, seed=1)) and not a.equals(sample_for_coding(df, per_stratum=4, seed=2))
    big = a[a.sample_size == 4]
    for _, g in big.groupby("stratum"):
        assert g.pair_id.nunique() == len(g)                             # spread over distinct pairs when possible


def test_codes_parsing_and_validation():
    assert parse_codes("B; A;A") == ["A", "B"] and parse_codes(float("nan")) == []
    sheet = pd.DataFrame({"run_id": ["r1", "r2", "r3"], "codes": ["NEVER_LOOKED", "NEW:rationalised_zero_attrition", "MADE_UP;OTHER"]})
    problems = validate_sheet(sheet)
    assert len(problems) == 1 and "MADE_UP" in problems[0] and "NEVER_LOOKED" in SEED_CODES


def test_kappa_values():
    assert cohens_kappa([1, 1, 0, 0], [1, 0, 0, 0]) == 0.5
    assert cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert cohens_kappa([1, 0, 1, 0], [0, 1, 0, 1]) == -1.0
    a = pd.DataFrame({"run_id": ["r1", "r2", "r3", "r4"], "codes": ["X", "X;Y", "", "Y"]})
    b = pd.DataFrame({"run_id": ["r1", "r2", "r3", "r4"], "codes": ["X", "X", "", "Y"]})
    t = kappa_table(a, b).set_index("code")
    assert t.loc["X", "kappa"] == 1.0 and t.loc["Y", "percent_agreement"] == 0.75 and t.loc["Y", "coder_a"] == 2


def test_detector_validity_against_human_labels():
    sheet = pd.DataFrame({"model": ["m"] * 5, "condition": ["none"] * 5, "rep": [0] * 5, "task_id": list("abcde"),
                          "cue_inspected_human": ["y", "y", "n", "n", ""]})
    beh = pd.DataFrame({"model": ["m"] * 5, "condition": ["none"] * 5, "rep": [0] * 5, "task_id": list("abcde"),
                        "cue_inspected": [True, False, True, False, True]})
    v = detector_validity(sheet, beh)
    assert v["n"] == 4 and v["precision"] == 0.5 and v["recall"] == 0.5 and v["kappa"] == 0.0


def test_weighted_prevalence_differs_from_the_raw_share_and_matches_hand_computation():
    # stratum A: 2 coded runs standing for 20 (weight 10), stratum B: 2 coded standing for 2 (weight 1)
    rows = []
    for i, (codes, w, oc) in enumerate([("NEVER_LOOKED", 10, "fooled_missed"), ("", 10, "fooled_missed"),
                                        ("SAW_DISMISSED", 1, "missed_wrong"), ("SAW_DISMISSED", 1, "missed_wrong")]):
        rows.append(dict(model="m", condition="none", pair_id=f"p{i}", weight=w, outcome_class=oc, codes=codes))
    t = prevalence_table(pd.DataFrame(rows), B=500).set_index("code")
    assert abs(t.loc["NEVER_LOOKED", "weighted_prevalence"] - 10 / 22) < 1e-9       # raw share would be 1/4
    assert abs(t.loc["SAW_DISMISSED", "weighted_prevalence"] - 2 / 22) < 1e-9       # raw share would be 2/4
    assert t.lo.le(t.weighted_prevalence).all() and t.hi.ge(t.weighted_prevalence).all()


def test_coding_sheet_and_readable_transcripts_from_real_harness_output(tmp_path):
    d = Path(tempfile.mkdtemp())
    cmd_generate(argparse.Namespace(n=6, seed=0, salt="c", out=str(d / "tasks")))
    out = tmp_path / "runs" / "scripted-careful"
    run_llm_experiment(ScriptedClient("scripted-careful", reference_policy("HeuristicCarefulAgent")), d / "tasks", out,
                       ["none"], reps=1, max_tasks=6, seed=0, sandbox_kwargs={"allow_failurelab": True})
    scores = attach_pairs(load_runs([out]), d / "tasks" / "private")
    path = write_coding_sheet(sample_for_coding(scores, per_stratum=2, seed=0), d / "tasks" / "private", tmp_path / "coding")
    sheet = pd.read_csv(path)
    assert list(sheet.columns) == SHEET_COLUMNS and len(sheet) == 6
    assert sheet.codes.isna().all() and sheet.cue_inspected_human.isna().all()          # blank for humans to fill
    assert "cue_inspected" not in sheet.columns                                          # detector output is hidden
    assert not any("trapped" in t or "clean" in t for t in sheet.task_id)               # opaque ids
    md = Path(sheet.transcript_md[0]).read_text()
    assert "### Cell 1" in md and "```python" in md and "trap family" in md and "Output (run_python)" in md
