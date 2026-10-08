"""Deterministic behaviour diagnosis from transcripts (Phase 5, layer A).

For every run we ask two mechanical questions about the agent's own CODE: did it LOOK at the property that
exposes the planted flaw (the 'cue'), and did it FLAG the flaw? On trapped tasks that gives four diagnoses:
  detected                 looked and flagged
  saw_not_flagged          looked but did not flag      (interpretation failure)
  never_looked             did not look, did not flag   (exploration failure)
  flagged_without_looking  flagged without examining the cue (generic caution, or luck)
On clean twins: clean_ok / false_alarm.

CAVEAT (important): `cue_inspected` is a keyword heuristic over the agent's code. It is a measurement with
error. Measure that error with a human audit (coding.detector_validity) and report it next to any result.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .inference import Study
from .runner import safe_name
from .schema import GroundTruth

CUE_PATTERNS = {
    # leakage: examining the temporal alignment/construction of `signal` relative to returns
    "leakage": re.compile(
        r"signal_asof|shift\(\s*-\s*\d+|center\s*=\s*True|look.?ahead|future|reconstruct|rebuild|reproduc|"
        r"re-?comput|how .{0,20}signal .{0,20}(?:built|constructed|computed)", re.I),
    # survivorship: examining who/what is in the sample and whether anything dropped out or was added late
    "survivorship": re.compile(
        r"\.size\(\)|nunique\(\)|value_counts|\bstatus\b|delist|index_add_date|attrition|surviv|balanced|"
        r"\bgaps?\b|last_date|first_date|\.count\(\)|missing (?:ids?|names|assets)", re.I),
    # multiple testing: examining how many things were tried, or correcting for it
    "multiple_testing": re.compile(
        r"library_size|bonferroni|multipletests|\bfdr\b|benjamini|\bholm\b|p.?hack|data.?snoop|winner'?s curse|"
        r"n_?(?:signals|tests|candidates)|number of (?:signals|tests|candidates)|startswith\(\s*['\"]s['\"]|"
        r"filter\(\s*regex", re.I),
}
CORRECTION = re.compile(r"bonferroni|multipletests|\bfdr\b|benjamini|\bholm\b|alpha\s*/\s*\w+", re.I)

DIAGNOSES = ("detected", "saw_not_flagged", "never_looked", "flagged_without_looking", "clean_ok", "false_alarm", "invalid")


def cue_inspected(family: str, code: str) -> bool:
    return bool(CUE_PATTERNS[family].search(code))


def diagnose(variant: str, valid: bool, looked: bool, tp: int, fp: int) -> str:
    if not valid:
        return "invalid"
    if variant == "clean":
        return "false_alarm" if fp > 0 else "clean_ok"
    if tp > 0:
        return "detected" if looked else "flagged_without_looking"
    return "saw_not_flagged" if looked else "never_looked"


def transcript_path(row) -> Path:
    return Path(row["_source"]).parent / f"rep{int(row['rep'])}" / safe_name(row["agent"]) / "transcripts" / f"{row['task_id']}.json"


def outcome_class(row) -> str:
    """Graded outcome used for sampling and tables."""
    if not row["valid"]:
        return "invalid"
    if row["variant"] == "clean":
        return "false_alarm" if row["fp"] > 0 else "clean_ok"
    if row["fn"] == 0:
        return "caught"
    if row["fooled"]:
        return "fooled_missed"
    return "right_wrong_reason" if row["conclusion_correct"] else "missed_wrong"


def behavior_table(scores: pd.DataFrame, private_dir: Path) -> pd.DataFrame:
    """scores: output of study.attach_pairs(study.load_runs(...)). Reads each run's transcript."""
    fam = {}
    for f in Path(private_dir).glob("*.json"):
        t = GroundTruth.load(f)
        fam[t.task_id] = t.family
    rows = []
    for _, r in scores.iterrows():
        tp = transcript_path(r)
        cells = json.loads(tp.read_text())["cells"] if tp.exists() else []
        code = "\n".join(c["code"] for c in cells)
        family = fam[r["task_id"]]
        looked = cue_inspected(family, code)
        rows.append({
            "model": r["model"], "condition": r["condition"], "rep": int(r["rep"]), "task_id": r["task_id"],
            "pair_id": r["pair_id"], "family": family, "variant": r["variant"], "outcome": outcome_class(r),
            "n_cells": len(cells), "n_error_cells": sum(bool(c.get("error")) for c in cells),
            "cue_inspected": looked, "mentioned_correction": bool(CORRECTION.search(code)),
            "flagged_correct": bool(r["tp"] > 0), "flagged_any": bool(r["tp"] + r["fp"] > 0),
            "diagnosis": diagnose(r["variant"], bool(r["valid"]), looked, int(r["tp"]), int(r["fp"])),
        })
    return pd.DataFrame(rows)


def _rate(study: Study, df: pd.DataFrame, num, den, level: float) -> dict:
    idx = df["pair_id"].map(study.index).to_numpy()
    n = np.bincount(idx, weights=np.asarray(num, float), minlength=study.P)
    d = np.bincount(idx, weights=np.asarray(den, float), minlength=study.P)
    return study.estimate((n, d), level).as_dict()


def _decompose(study: Study, t: pd.DataFrame, level: float) -> dict:
    looked, caught = t["cue_inspected"].to_numpy(bool), t["flagged_correct"].to_numpy(bool)
    one = np.ones(len(t))
    return {"looked_rate": _rate(study, t, looked, one, level),
            "flag_given_looked": _rate(study, t, looked & caught, looked, level),
            "flag_given_not_looked": _rate(study, t, ~looked & caught, ~looked, level),
            "miss_never_looked_share": _rate(study, t, ~looked & ~caught, ~caught, level)}


def summarize(beh: pd.DataFrame, study: Study, level: float = 0.95) -> dict:
    """Per model and condition: how often trapped runs looked at the cue, and how often looking led to a flag."""
    out: dict = {}
    trapped = beh[beh.variant == "trapped"]
    for (m, c), g in trapped.groupby(["model", "condition"]):
        out.setdefault(m, {})[c] = {
            "overall": _decompose(study, g, level),
            "by_family": {fam: _decompose(study, gf, level) for fam, gf in g.groupby("family")},
            "diagnosis_counts": {k: int(v) for k, v in g["diagnosis"].value_counts().items()},
        }
    return out
