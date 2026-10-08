"""Reproducibility checks: can someone else regenerate every artefact and get the same bytes/numbers?

Four independent checks:
  tasks      regenerate the task set from the pre-registered seed/salt and compare file hashes with a manifest
  manifest   the on-disk task set still matches the manifest written at freeze time
  regrade    re-grade every stored raw submission against the answer keys; compare with scores.jsonl
  analysis   re-run the analysis from scores.jsonl; compare results.json byte for byte

HONEST CAVEAT: numpy's Generator does not promise identical random streams across major versions, so task
regeneration on a different numpy can legitimately differ. The robust fix is to publish the generated task set
(public and private) with its manifest as a release artefact; the `tasks` check then tells you whether
regeneration happens to work on your stack.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .grader import grade, grade_invalid
from .runner import safe_name
from .schema import GroundTruth, SubmissionError, parse_submission
from .study import load_runs, run_study


def file_hashes(tasks_dir: Path) -> dict[str, str]:
    root = Path(tasks_dir)
    out = {}
    for sub in ("public", "private"):
        for f in sorted((root / sub).rglob("*")):
            if f.is_file():
                out[str(f.relative_to(root))] = hashlib.sha256(f.read_bytes()).hexdigest()
    return out


def write_manifest(tasks_dir: Path, out: Path, meta: dict | None = None) -> dict:
    files = file_hashes(tasks_dir)
    rec = {"n_files": len(files), "numpy": np.__version__, "pandas": pd.__version__, "meta": meta or {}, "files": files}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rec, indent=1, sort_keys=True))
    return rec


def _diff(want: dict, got: dict, label: str) -> list[str]:
    bad = [f"{label}: changed {k}" for k in want if k in got and want[k] != got[k]]
    bad += [f"{label}: missing {k}" for k in want if k not in got]
    bad += [f"{label}: unexpected {k}" for k in got if k not in want]
    return bad


def verify_manifest(tasks_dir: Path, manifest: Path) -> list[str]:
    return _diff(json.loads(Path(manifest).read_text())["files"], file_hashes(tasks_dir), "task set")


def regenerate_and_compare(plan: dict, manifest: Path) -> list[str]:
    from .cli import cmd_generate
    ts = plan["design"]["task_set"]
    want = json.loads(Path(manifest).read_text())
    with tempfile.TemporaryDirectory() as tmp:
        cmd_generate(argparse.Namespace(n=ts["n_per_family"], seed=ts["seed"], salt=ts["salt"], out=tmp))
        problems = _diff(want["files"], file_hashes(Path(tmp)), "regenerated tasks")
    if problems and (want.get("numpy") != np.__version__ or want.get("pandas") != pd.__version__):
        problems.append(f"note: manifest written with numpy {want.get('numpy')} / pandas {want.get('pandas')}, "
                        f"now {np.__version__} / {pd.__version__}; random streams are not guaranteed across versions")
    return problems


def regrade(run_dirs, tasks_dir: Path) -> list[str]:
    """Re-grade every stored raw submission; the graded fields must equal scores.jsonl."""
    problems = []
    priv = Path(tasks_dir) / "private"
    for d in run_dirs:
        for _, row in load_runs([d]).iterrows():
            adir = Path(d) / f"rep{int(row['rep'])}" / safe_name(row["agent"])
            truth = GroundTruth.load(priv / f"{row['task_id']}.json")
            raw_path = adir / f"{row['task_id']}.json"
            if raw_path.exists():
                try:
                    s = grade(parse_submission(json.loads(raw_path.read_text()), row["task_id"]), truth)
                except SubmissionError:
                    s = grade_invalid(truth)
            elif (adir / f"{row['task_id']}.error.txt").exists():
                s = grade_invalid(truth)
            else:
                problems.append(f"regrade: no raw submission for {row['agent']} rep{int(row['rep'])} {row['task_id']}")
                continue
            for f in ("valid", "conclusion_correct", "fooled", "right_for_wrong_reason", "tp", "fp", "fn"):
                if bool(getattr(s, f)) != bool(row[f]) if isinstance(getattr(s, f), bool) else getattr(s, f) != row[f]:
                    problems.append(f"regrade: {row['agent']} rep{int(row['rep'])} {row['task_id']} field {f}: stored {row[f]}, regraded {getattr(s, f)}")
            if abs(s.brier - float(row["brier"])) > 1e-9:
                problems.append(f"regrade: {row['agent']} rep{int(row['rep'])} {row['task_id']} field brier differs")
    return problems


def analysis_reproduces(report_dir: Path, **study_kwargs) -> list[str]:
    stored = Path(report_dir) / "results.json"
    if not stored.exists():
        return [f"analysis: {stored} not found"]
    with tempfile.TemporaryDirectory() as tmp:
        run_study(out_dir=tmp, **study_kwargs)
        fresh = (Path(tmp) / "results.json").read_bytes()
    return [] if fresh == stored.read_bytes() else ["analysis: re-running the analysis does not reproduce results.json byte for byte"]


def repro_check(tasks_dir, run_dirs, report_dir, manifest, plan: dict, study_kwargs: dict) -> dict[str, list[str]]:
    return {"manifest": verify_manifest(tasks_dir, manifest),
            "tasks": regenerate_and_compare(plan, manifest),
            "regrade": regrade(run_dirs, tasks_dir),
            "analysis": analysis_reproduces(report_dir, **study_kwargs)}
