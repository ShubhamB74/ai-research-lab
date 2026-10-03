import pytest

from failurelab.schema import SubmissionError, parse_submission

GOOD = {"task_id": "t1", "effect_exists": True, "direction": "positive", "confidence": 0.7, "flags": []}


def test_valid_submission_parses():
    sub = parse_submission(GOOD, "t1")
    assert sub.effect_exists and sub.direction == "positive"


@pytest.mark.parametrize("patch", [
    {"task_id": "other"},
    {"effect_exists": "yes"},
    {"direction": "up"},
    {"effect_exists": False, "direction": "positive"},   # inconsistent
    {"effect_exists": True, "direction": "none"},        # inconsistent
    {"confidence": 1.5},
    {"confidence": True},
    {"flags": ["made_up_flag"]},
])
def test_invalid_submissions_rejected(patch):
    with pytest.raises(SubmissionError):
        parse_submission({**GOOD, **patch}, "t1")
