"""Multi-condition, multi-repetition LLM experiments with resumable, auditable output.

Layout of --out:
  rep0/<agent>/<task>.json                    raw submission (as graded)
  rep0/<agent>/transcripts/<task>.json        full transcript (prompts, tool calls, outputs, usage)
  scores.jsonl                                one row per (agent, condition, rep, task): graded fields + status

Loop order is rep (outer) then condition, so a run cut short still has every condition for rep 0.
NOTE: repeated runs of the same task are NOT independent. Phase 4 analysis must cluster by task.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

from .agents.llm_agent import LLMAgent
from .llm import ModelClient
from .runner import run_agent, safe_name
from .schema import GroundTruth


def select_tasks(private_root: Path, max_tasks: int | None, seed: int = 0) -> set[str] | None:
    """Pick whole clean/trapped PAIRS, rotating across families first, then across the cells
    (mechanism x difficulty) inside each family, so small subsets still cover every family."""
    if max_tasks is None:
        return None
    truths = [GroundTruth.load(f) for f in sorted(Path(private_root).glob("*.json"))]
    cells: dict[str, dict[tuple, list[tuple[str, str]]]] = {}
    for t in truths:
        if t.variant == "trapped":
            key = (t.params.get("mechanism", ""), t.params.get("difficulty", ""))
            cells.setdefault(t.family, {}).setdefault(key, []).append((t.task_id, t.twin_id))
    rng = random.Random(seed)
    for fam in cells.values():
        for pairs in fam.values():
            rng.shuffle(pairs)
    want = (max_tasks + 1) // 2
    cursor = {fam: 0 for fam in cells}
    chosen: list[tuple[str, str]] = []
    while len(chosen) < want and any(any(v) for fam in cells.values() for v in fam.values()):
        for fam in sorted(cells):
            keys = sorted(cells[fam])
            for _ in range(len(keys)):                      # next non-empty cell of this family
                k = keys[cursor[fam] % len(keys)]
                cursor[fam] += 1
                if cells[fam][k]:
                    chosen.append(cells[fam][k].pop())
                    break
            if len(chosen) >= want:
                break
    return {i for pair in chosen for i in pair}


def plan(tasks_dir: Path, conditions: list[str], reps: int, max_tasks: int | None, max_turns: int, seed: int = 0) -> dict:
    priv = Path(tasks_dir) / "private"
    ids = select_tasks(priv, max_tasks, seed)
    n_tasks = len(ids) if ids is not None else len(list(priv.glob("*.json")))
    n_runs = n_tasks * reps * len(conditions)
    return {"tasks": n_tasks, "conditions": conditions, "reps": reps, "task_runs": n_runs,
            "max_model_calls": n_runs * max_turns, "note": "token cost grows with transcript length; "
            "run a 3-pair smoke test first and measure usage in the transcripts"}


def run_llm_experiment(client: ModelClient, tasks_dir: Path, out_dir: Path, conditions: list[str], reps: int,
                       max_tasks: int | None = None, max_turns: int = 15, seed: int = 0,
                       sandbox_kwargs: dict | None = None) -> pd.DataFrame:
    tasks_dir, out_dir = Path(tasks_dir), Path(out_dir)
    pub, priv = tasks_dir / "public", tasks_dir / "private"
    ids = select_tasks(priv, max_tasks, seed)
    rows = []
    for rep in range(reps):
        for cond in conditions:
            rep_root = out_dir / f"rep{rep}"
            agent = LLMAgent(client, cond, max_turns=max_turns, rep=rep, sandbox_kwargs=sandbox_kwargs,
                             transcript_dir=rep_root / safe_name(f"{client.name}|{cond}") / "transcripts")
            for s in run_agent(agent, pub, priv, rep_root, task_ids=ids):
                tpath = agent.transcript_dir / f"{s.task_id}.json"
                tr = json.loads(tpath.read_text()) if tpath.exists() else {}
                rows.append({"agent": agent.name, "condition": cond, "rep": rep, **s.__dict__,
                             "status": tr.get("status", "missing"), "format_retries": tr.get("format_retries", 0),
                             "n_python_calls": tr.get("n_python_calls", 0),
                             "preregistered_first": tr.get("preregistered_first", False),
                             "tripwire_hits": tr.get("tripwire_hits", []),
                             "input_tokens": tr.get("usage", {}).get("input_tokens", 0),
                             "output_tokens": tr.get("usage", {}).get("output_tokens", 0)})
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scores.jsonl").write_text("\n".join(json.dumps(r, default=str) for r in rows) + "\n")
    return df
