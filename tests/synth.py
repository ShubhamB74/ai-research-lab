"""Shared synthetic-experiment builders for tests (real on-disk formats, planted rates)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from failurelab import prereg
from failurelab.schema import GroundTruth

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("leakage", "survivorship", "multiple_testing")


def _expit(x):
    return 1 / (1 + np.exp(-x))


def build(tmp, pairs=60, reps=3, models=("m1",), rates=None, seed=0):
    """Private answer keys, scores.jsonl per model, a frozen pre-registration. Returns run_study kwargs."""
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
    plan["design"].update(models=list(models), reps=reps, conditions=list(rates), max_tasks=None)
    plan["analysis"]["bootstrap_B"] = 1000
    pf, doc, lock = tmp / "prereg.json", tmp / "PREREG.md", tmp / "prereg.lock"
    pf.write_text(json.dumps(plan)); doc.write_text("frozen text")
    prereg.freeze(pf, doc, lock)
    return dict(tasks_dir=tmp / "tasks", run_dirs=run_dirs, out_dir=tmp / "report", prereg_path=pf, doc_path=doc, lock_path=lock)


def behavior_frame(pairs=40, models=("m1",), conditions=("none", "checklist"), seed=0):
    """Synthetic behaviour table: looking raises the chance of flagging."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in models:
        for c in conditions:
            p_look = 0.3 if c == "none" else 0.8
            for p in range(pairs):
                looked = bool(rng.random() < p_look)
                caught = bool(rng.random() < (0.85 if looked else 0.05))
                rows.append(dict(model=m, condition=c, rep=0, task_id=f"t{p:03d}a", pair_id=f"t{p:03d}a", family=FAMILIES[p % 3],
                                 variant="trapped", outcome="caught" if caught else "fooled_missed", n_cells=3, n_error_cells=0,
                                 cue_inspected=looked, mentioned_correction=False, flagged_correct=caught, flagged_any=caught,
                                 diagnosis="detected" if (looked and caught) else "saw_not_flagged" if looked else
                                 "flagged_without_looking" if caught else "never_looked"))
    return pd.DataFrame(rows)
