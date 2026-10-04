"""Prompts and tool schemas. Three conditions, identical except for one appended block:

  none         : protocol + answer definitions only
  generic_hint : + a one-paragraph reminder that results can be misleading
  checklist    : + a checklist covering ALL six taxonomy items (same text for every task, so it never
                 reveals which trap, if any, is present)

Disclosed design choices (they are threats to validity, see README):
  * the base prompt tells the agent how answers are scored (flagging an absent problem counts against it),
    so recall is not inflated by flag-spamming;
  * every task's question already says "flag any methodological problems you find".
"""
from __future__ import annotations

from .schema import DIRECTIONS, Task
from .taxonomy import ALL_FLAG_CODES

CONDITIONS = ("none", "generic_hint", "checklist")

BASE_SYSTEM = f"""You are a quantitative research analyst working on a research question with a dataset.
Follow this protocol:
1. Call `preregister` BEFORE any analysis: state your hypothesis, the mechanism behind it, the predicted direction, and what result would prove you wrong.
2. Analyse the data with `run_python` (a persistent Python session; pandas and numpy are available; the dataset is `data.csv` in the working directory; there is no internet). Run cheap tests first and try hard to break your own result.
3. Call `submit_answer` exactly once when you are done. It ends the task.

Your answer is checked mechanically against a known answer key:
- effect_exists: true only if, after correcting for any problem you identify, the data supports a genuine effect. If problems prevent a reliable conclusion, answer false.
- direction: "positive" or "negative" if effect_exists is true, otherwise "none".
- confidence: your probability (0 to 1) that your conclusion is correct. Overconfidence is penalised.
- flags: the methodological problems you believe are present in THIS dataset, chosen from: {", ".join(ALL_FLAG_CODES)}. Flagging a problem that is not present counts against you, and so does missing one that is present."""

GENERIC_HINT = """

Reminder: real datasets often contain problems that make a naive analysis misleading. Before you conclude, ask what could make this result misleading, and test it."""

CHECKLIST = """

Before concluding, work through this checklist and test each item that could apply:
- leakage: does any column use information that would not have been available at prediction time? Check timestamps and how derived columns are built (e.g. try to reproduce them from raw columns).
- survivorship: does the sample contain everything it claims to, or only entities that survived or were selected? Look for missing entities, gaps, zero attrition, or history recorded before inclusion.
- multiple_testing: how many things were tried to produce this result? Correct for the number of tests searched.
- overfitting: does the result hold out of sample?
- confounding: could a third variable explain the relationship?
- regime_shift: is the relationship stable across subperiods?"""

_SUFFIX = {"none": "", "generic_hint": GENERIC_HINT, "checklist": CHECKLIST}

TOOLS = [
    {"name": "preregister",
     "description": "Record your hypothesis BEFORE analysing. Call once, first.",
     "input_schema": {"type": "object", "properties": {
         "hypothesis": {"type": "string"}, "mechanism": {"type": "string"},
         "predicted_direction": {"type": "string", "enum": list(DIRECTIONS)},
         "falsifier": {"type": "string", "description": "what result would prove you wrong"}},
         "required": ["hypothesis", "mechanism", "predicted_direction", "falsifier"]}},
    {"name": "run_python",
     "description": "Run Python in a persistent session (variables persist). pandas, numpy available; data.csv in the working directory. Returns stdout/stderr, truncated.",
     "input_schema": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}},
    {"name": "submit_answer",
     "description": "Submit your final research conclusion. Ends the task. Call exactly once.",
     "input_schema": {"type": "object", "properties": {
         "effect_exists": {"type": "boolean"},
         "direction": {"type": "string", "enum": list(DIRECTIONS)},
         "confidence": {"type": "number", "minimum": 0, "maximum": 1},
         "flags": {"type": "array", "items": {"type": "string", "enum": list(ALL_FLAG_CODES)}},
         "revisions": {"type": "array", "items": {"type": "string"},
                       "description": "ideas you tried and abandoned, or conclusions you revised"}},
         "required": ["effect_exists", "direction", "confidence", "flags"]}},
]


def build_system_prompt(condition: str) -> str:
    if condition not in _SUFFIX:
        raise ValueError(f"unknown condition {condition!r}; choose from {CONDITIONS}")
    return BASE_SYSTEM + _SUFFIX[condition]


def build_user_prompt(task: Task) -> str:
    cols = "\n".join(f"- {name}: {desc}" for name, desc in task.data_dictionary.items())
    return f"Research question:\n{task.question}\n\nData dictionary (file: data.csv):\n{cols}"
