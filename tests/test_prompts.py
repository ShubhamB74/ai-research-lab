import argparse
import tempfile
from pathlib import Path

from failurelab.cli import cmd_generate
from failurelab.prompts import CONDITIONS, TOOLS, build_system_prompt, build_user_prompt
from failurelab.schema import GroundTruth, Task
from failurelab.taxonomy import ALL_FLAG_CODES


def test_conditions_differ_only_by_an_appended_block():
    base = build_system_prompt("none")
    for cond in ("generic_hint", "checklist"):
        p = build_system_prompt(cond)
        assert p.startswith(base) and len(p) > len(base)
    assert set(CONDITIONS) == {"none", "generic_hint", "checklist"}


def test_none_condition_gives_no_diagnostic_hints():
    base = build_system_prompt("none").lower()
    for phrase in ("checklist", "could make this result misleading", "reproduce them from raw", "number of tests"):
        assert phrase not in base


def test_checklist_covers_all_six_taxonomy_items_and_is_task_independent():
    text = build_system_prompt("checklist")
    for code in ALL_FLAG_CODES:
        assert f"- {code}:" in text
    assert build_system_prompt("checklist") == text   # no per-task content


def test_submit_tool_flags_use_exactly_the_taxonomy():
    submit = next(t for t in TOOLS if t["name"] == "submit_answer")
    assert submit["input_schema"]["properties"]["flags"]["items"]["enum"] == list(ALL_FLAG_CODES)


def test_user_prompts_are_identical_across_twins_and_reveal_nothing():
    d = Path(tempfile.mkdtemp())
    cmd_generate(argparse.Namespace(n=3, seed=0, salt="p", out=str(d)))
    by_id = {t.task_id: Task.load(d / "public" / t.task_id) for t in
             (GroundTruth.load(f) for f in (d / "private").glob("*.json"))}
    for f in (d / "private").glob("*.json"):
        truth = GroundTruth.load(f)
        a, b = build_user_prompt(by_id[truth.task_id]), build_user_prompt(by_id[truth.twin_id])
        assert a == b
        for word in ("trapped", "clean", "twin", truth.task_id):
            assert word not in a.lower()
