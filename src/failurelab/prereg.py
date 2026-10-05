"""Pre-registration: machine-readable plan, validation, and a hash lock.

`freeze` writes experiments/prereg.lock containing SHA-256 hashes of docs/prereg.json and
docs/PREREGISTRATION.md plus the git commit if available. `analyze` re-hashes and labels results
NON-CONFIRMATORY if the files changed or no lock exists.

HONEST LIMIT: a lock file proves the plan did not change after the lock was written. It does not prove the
lock was written BEFORE the data existed. For that, commit the lock (and push, or otherwise timestamp it)
before the first live run, and keep the commit hash in the report.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

from .prompts import CONDITIONS

DEFAULT_RULES = {
    "baseline_condition": "none",
    "treatment_condition": "checklist",
    "H1": {"recall_below": 0.50, "clean_accuracy_above": 0.90, "falsify_if_recall_lower_above": 0.40},
    "H2": {"delta_recall_min": 0.20, "delta_false_flag_max": 0.10, "falsify_if_delta_recall_below": 0.10},
    "H3": {"share_min": 0.25, "support_lower_above": 0.10, "falsify_if_share_below": 0.10},
    "H4": {},
}


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def validate(p: dict) -> list[str]:
    """Return a list of problems; empty means the plan is complete enough to freeze."""
    errs = []
    d, a = p.get("design", {}), p.get("analysis", {})
    models = d.get("models")
    if not models or not all(isinstance(m, str) and m for m in models):
        errs.append("design.models must list the model names exactly as they appear in the runs (e.g. 'anthropic:<id>')")
    if not isinstance(d.get("reps"), int) or d["reps"] < 1:
        errs.append("design.reps must be an integer >= 1")
    conds = d.get("conditions") or []
    if not conds or any(c not in CONDITIONS for c in conds):
        errs.append(f"design.conditions must be a non-empty subset of {CONDITIONS}")
    rules = p.get("rules", {})
    for key in ("baseline_condition", "treatment_condition"):
        if rules.get(key) not in conds:
            errs.append(f"rules.{key} must be one of design.conditions")
    if d.get("max_tasks") is not None and (not isinstance(d["max_tasks"], int) or d["max_tasks"] < 2):
        errs.append("design.max_tasks must be null (all tasks) or an integer >= 2")
    if not isinstance(a.get("bootstrap_B"), int) or a["bootstrap_B"] < 1000:
        errs.append("analysis.bootstrap_B must be an integer >= 1000")
    if not isinstance(a.get("bootstrap_seed"), int):
        errs.append("analysis.bootstrap_seed must be an integer")
    return errs


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        except Exception:
            return None
    return {"git_commit": run("rev-parse", "HEAD"), "git_dirty": bool(run("status", "--porcelain"))}


def freeze(prereg: str | Path, doc: str | Path, lock: str | Path) -> dict:
    errs = validate(load(prereg))
    if errs:
        raise ValueError("cannot freeze an incomplete pre-registration:\n- " + "\n- ".join(errs))
    files = {str(prereg): _sha(Path(prereg)), str(doc): _sha(Path(doc))}
    record = {"frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              "sha256": files, **_git()}
    Path(lock).parent.mkdir(parents=True, exist_ok=True)
    Path(lock).write_text(json.dumps(record, indent=2))
    return record


def check_lock(lock: str | Path, prereg: str | Path, doc: str | Path) -> tuple[str, str]:
    """Returns (state, detail) with state in {'ok', 'missing', 'modified'}."""
    lock = Path(lock)
    if not lock.exists():
        return "missing", "no experiments/prereg.lock: the plan was never frozen"
    rec = json.loads(lock.read_text())
    for path in (prereg, doc):
        want = rec["sha256"].get(str(path))
        if want is None:
            return "modified", f"{path} is not covered by the lock"
        if _sha(Path(path)) != want:
            return "modified", f"{path} changed after it was frozen"
    return "ok", f"frozen {rec['frozen_at_utc']}" + (f", commit {rec['git_commit']}" if rec.get("git_commit") else "")
