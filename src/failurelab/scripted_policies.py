"""Scripted stand-ins for an LLM, used to validate the HARNESS (not any model).

`reference_policy("HeuristicCarefulAgent")` behaves like a model that: preregisters, runs a Python cell
inside the sandbox that executes the corresponding baseline analysis, reads the printed result back,
and submits it. If the harness is correct, scoring this through the full tool loop must reproduce the
baseline agent's scores exactly (tests/test_harness_reproduces_baselines.py).
"""
from __future__ import annotations

import json

from .llm import ModelTurn, tool_turn

_CODE = '''
import json
from failurelab.agents import {cls}
from failurelab.schema import Task
task = Task(task_id="x", question="", data_path="data.csv", data_dictionary={{}}, allowed_flags=[])
raw = {cls}().run(task)
print("RESULT=" + json.dumps(raw, default=str))
'''


def _last_tool_text(messages: list[dict]) -> str:
    for block in messages[-1]["content"]:
        if block.get("type") == "tool_result":
            return block["content"]
    return ""


def reference_policy(agent_class_name: str):
    def policy(messages: list[dict], tools: list[dict]) -> ModelTurn:
        step = sum(1 for m in messages if m["role"] == "assistant")
        if step == 0:
            return tool_turn(("preregister", {"hypothesis": "scripted reference analysis", "mechanism": "n/a",
                                              "predicted_direction": "none", "falsifier": "n/a"}))
        if step == 1:
            return tool_turn(("run_python", {"code": _CODE.format(cls=agent_class_name)}))
        text = _last_tool_text(messages)
        raw = json.loads(text.split("RESULT=", 1)[1].splitlines()[0])
        return tool_turn(("submit_answer", {k: raw[k] for k in ("effect_exists", "direction", "confidence", "flags")}))
    return policy
