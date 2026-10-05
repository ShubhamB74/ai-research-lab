import json
from pathlib import Path

import pytest

from failurelab import prereg

ROOT = Path(__file__).resolve().parents[1]


def _filled(tmp_path):
    plan = json.loads((ROOT / "docs" / "prereg.json").read_text())
    plan["design"]["models"], plan["design"]["reps"] = ["anthropic:x"], 3
    p, d = tmp_path / "prereg.json", tmp_path / "PREREG.md"
    p.write_text(json.dumps(plan)); d.write_text("hypotheses text")
    return p, d, tmp_path / "prereg.lock"


def test_shipped_template_is_incomplete_until_filled_and_rules_match_defaults():
    plan = prereg.load(ROOT / "docs" / "prereg.json")
    errs = prereg.validate(plan)
    assert any("models" in e for e in errs) and any("reps" in e for e in errs)
    assert plan["rules"] == prereg.DEFAULT_RULES


def test_freeze_refuses_an_incomplete_plan(tmp_path):
    with pytest.raises(ValueError):
        prereg.freeze(ROOT / "docs" / "prereg.json", ROOT / "docs" / "PREREGISTRATION.md", tmp_path / "x.lock")


def test_lock_detects_modification_and_absence(tmp_path):
    p, d, lock = _filled(tmp_path)
    assert prereg.check_lock(lock, p, d)[0] == "missing"
    prereg.freeze(p, d, lock)
    assert prereg.check_lock(lock, p, d)[0] == "ok"
    d.write_text("hypotheses text, quietly edited later")
    state, detail = prereg.check_lock(lock, p, d)
    assert state == "modified" and "changed after it was frozen" in detail
    d.write_text("hypotheses text")
    plan = json.loads(p.read_text()); plan["rules"]["H2"]["delta_recall_min"] = 0.05
    p.write_text(json.dumps(plan))
    assert prereg.check_lock(lock, p, d)[0] == "modified"


def test_validate_rejects_bad_values(tmp_path):
    p, d, _ = _filled(tmp_path)
    plan = json.loads(p.read_text())
    plan["design"]["conditions"] = ["none", "made_up"]
    plan["analysis"]["bootstrap_B"] = 10
    plan["rules"]["treatment_condition"] = "generic_hint"
    errs = prereg.validate(plan)
    assert len(errs) >= 3


def test_human_readable_preregistration_states_every_threshold_in_the_json():
    doc = (ROOT / "docs" / "PREREGISTRATION.md").read_text()
    for hyp in ("H1", "H2", "H3"):
        for name, value in prereg.DEFAULT_RULES[hyp].items():
            assert str(value) in doc, (hyp, name, value)
    for hyp in ("H1", "H2", "H3", "H4", "H5"):
        assert f"**{hyp}" in doc
