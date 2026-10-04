"""LLM research agent: model <-> tools loop with full transcript capture (Phase 3).

Flow per task: preregister -> run_python (many) -> submit_answer. Everything the model sees and does
is stored in a transcript JSON so numbers can be regenerated and failures analysed later.

Outcomes (transcript `status`):
  submitted      a schema-valid submit_answer was received
  invalid_format submit_answer kept violating the schema (scored as an invalid submission)
  no_submission  the model stopped using tools without submitting
  max_turns      turn budget exhausted
  aborted        RunAborted (budget / API outage): NOT a final outcome; a re-run redoes the task
Resume: a transcript with a final status is reused without calling the model, so interrupted
experiments can simply be re-run.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from ..llm import ModelClient, RunAborted
from ..prompts import TOOLS, build_system_prompt, build_user_prompt
from ..sandbox import Sandbox
from ..schema import DIRECTIONS, SubmissionError, Task, parse_submission
from .base import Agent

FINAL_STATUSES = {"submitted", "invalid_format", "no_submission", "max_turns"}
# A tripwire, not a guarantee: flags code that pokes at the benchmark's own files for HUMAN review.
TRIPWIRE = re.compile(r"private|ground[_ ]?truth|failurelab|\.\./|/mnt/|os\.walk|glob\.glob\(\s*['\"]/", re.I)


def _user_text(text: str) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": text}]}


class LLMAgent(Agent):
    def __init__(self, client: ModelClient, condition: str = "none", *, max_turns: int = 15,
                 transcript_dir: str | Path | None = None, sandbox_kwargs: dict | None = None,
                 max_format_retries: int = 2, max_nudges: int = 2, rep: int = 0):
        self.client, self.condition, self.max_turns = client, condition, max_turns
        self.transcript_dir = Path(transcript_dir) if transcript_dir else None
        self.sandbox_kwargs = sandbox_kwargs or {}
        self.max_format_retries, self.max_nudges, self.rep = max_format_retries, max_nudges, rep
        self.name = f"{client.name}|{condition}"
        build_system_prompt(condition)   # fail fast on a bad condition

    # ------------------------------------------------------------------ main loop
    def run(self, task: Task) -> dict[str, Any]:
        path = self.transcript_dir / f"{task.task_id}.json" if self.transcript_dir else None
        if path and path.exists():
            saved = json.loads(path.read_text())
            if saved.get("status") in FINAL_STATUSES:
                return saved.get("submission_raw") or {"task_id": task.task_id, "error": saved["status"]}

        system = build_system_prompt(self.condition)
        messages = [_user_text(build_user_prompt(task))]
        tr = {"task_id": task.task_id, "agent": self.name, "model": self.client.name,
              "condition": self.condition, "rep": self.rep, "started": time.time(), "status": "max_turns",
              "cells": [], "prereg": None, "preregistered_first": False, "format_retries": 0,
              "tripwire_hits": [], "usage": {"input_tokens": 0, "output_tokens": 0}, "model_calls": 0,
              "system_prompt": system}
        submission_raw: dict | None = None
        nudges = 0
        try:
            with Sandbox(task.data_path, **self.sandbox_kwargs) as sandbox:
                for _ in range(self.max_turns):
                    turn = self.client.complete(system, messages, TOOLS)
                    tr["model_calls"] += 1
                    for k in ("input_tokens", "output_tokens"):
                        tr["usage"][k] += int(turn.usage.get(k, 0))
                    messages.append({"role": "assistant", "content": turn.blocks})
                    if not turn.tool_calls:
                        nudges += 1
                        if nudges > self.max_nudges:
                            tr["status"] = "no_submission"
                            break
                        messages.append(_user_text("Please continue using the tools. When you are finished, call submit_answer."))
                        continue
                    results, done = [], None
                    for call in turn.tool_calls:
                        content, is_error, final = self._handle(call, task, sandbox, tr)
                        res = {"type": "tool_result", "tool_use_id": call.id, "content": content}
                        if is_error:
                            res["is_error"] = True
                        results.append(res)
                        done = done or final
                    messages.append({"role": "user", "content": results})
                    if done:
                        submission_raw, tr["status"] = done
                        break
        except RunAborted:
            tr["status"] = "aborted"
            self._save(path, tr, messages, None)
            raise
        self._save(path, tr, messages, submission_raw)
        return submission_raw or {"task_id": task.task_id, "error": tr["status"]}

    # ------------------------------------------------------------------ tool dispatch
    def _handle(self, call, task: Task, sandbox: Sandbox, tr: dict):
        """Returns (tool_result_text, is_error, final) where final is (raw_submission, status) or None."""
        inp = call.input if isinstance(call.input, dict) else {}
        if call.name == "preregister":
            if inp.get("predicted_direction") not in DIRECTIONS or not str(inp.get("hypothesis", "")).strip():
                return "Invalid preregistration: need a hypothesis and predicted_direction in " + str(DIRECTIONS), True, None
            if tr["prereg"] is None:
                tr["prereg"] = {k: str(inp.get(k, "")) for k in ("hypothesis", "mechanism", "predicted_direction", "falsifier")}
                tr["preregistered_first"] = not tr["cells"]
            return "Preregistration recorded.", False, None
        if call.name == "run_python":
            code = inp.get("code")
            if not isinstance(code, str):
                return "run_python needs a string `code`.", True, None
            hits = sorted({m.group(0) for m in TRIPWIRE.finditer(code)})
            tr["tripwire_hits"].extend(h for h in hits if h not in tr["tripwire_hits"])
            res = sandbox.run(code)
            tr["cells"].append({"code": code, "stdout": res.stdout, "stderr": res.stderr, "error": res.error,
                                "seconds": round(res.seconds, 3), "timed_out": res.timed_out})
            return res.as_tool_text(), bool(res.error or res.timed_out or res.crashed), None
        if call.name == "submit_answer":
            prereg = tr["prereg"] or {}
            raw = {"task_id": task.task_id, **{k: inp.get(k) for k in ("effect_exists", "direction", "confidence", "flags")},
                   "revisions": inp.get("revisions", []),
                   "hypothesis": prereg.get("hypothesis", ""), "mechanism": prereg.get("mechanism", ""),
                   "predicted_direction": prereg.get("predicted_direction", "none"),
                   "experiments": [{"code": c["code"][:1500], "output": (c["stdout"] or c["error"] or "")[:1500]}
                                   for c in tr["cells"]]}
            if raw["flags"] is None:
                raw["flags"] = []
            try:
                parse_submission(raw, task.task_id)
            except SubmissionError as exc:
                tr["format_retries"] += 1
                if tr["format_retries"] > self.max_format_retries:
                    return f"Invalid submission: {exc}", True, (raw, "invalid_format")
                return f"Invalid submission: {exc}. Fix it and call submit_answer again.", True, None
            return "Answer recorded. The task is complete.", False, (raw, "submitted")
        return f"Unknown tool {call.name!r}. Available: preregister, run_python, submit_answer.", True, None

    # ------------------------------------------------------------------ persistence
    def _save(self, path: Path | None, tr: dict, messages: list, submission_raw: dict | None) -> None:
        if not path:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tr = {**tr, "finished": time.time(), "n_python_calls": len(tr["cells"]),
              "submission_raw": submission_raw, "messages": messages}
        path.write_text(json.dumps(tr, indent=1, default=str))
