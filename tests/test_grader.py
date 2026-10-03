from failurelab.grader import grade, grade_invalid
from failurelab.schema import GroundTruth, Submission


def truth(trapped=True, exists=False, direction="none"):
    return GroundTruth(task_id="t", family="leakage", variant="trapped" if trapped else "clean",
                       traps_present=["leakage"] if trapped else [], effect_exists=exists,
                       true_direction=direction, twin_id="u")


def sub(exists, direction, flags=(), conf=1.0):
    return Submission(task_id="t", effect_exists=exists, direction=direction,
                      confidence=conf, flags=list(flags))


def test_perfect_answer():
    s = grade(sub(False, "none", ["leakage"]), truth())
    assert s.conclusion_correct and not s.fooled and s.tp == 1 and s.fp == 0 and s.fn == 0
    assert s.brier == 0.0


def test_fooled_by_trap():
    s = grade(sub(True, "positive", [], conf=0.9), truth())
    assert not s.conclusion_correct and s.fooled and s.fn == 1
    assert abs(s.brier - 0.81) < 1e-9


def test_right_for_wrong_reason():
    s = grade(sub(True, "positive", []), truth(exists=True, direction="positive"))
    assert s.conclusion_correct and s.right_for_wrong_reason and not s.fooled


def test_wrong_direction_is_incorrect():
    s = grade(sub(True, "negative"), truth(exists=True, direction="positive"))
    assert not s.conclusion_correct and s.fooled


def test_false_flag_on_clean_task():
    s = grade(sub(False, "none", ["leakage"]), truth(trapped=False))
    assert s.conclusion_correct and s.fp == 1 and not s.fooled


def test_invalid_counts_as_failure():
    s = grade_invalid(truth())
    assert not s.valid and not s.conclusion_correct and s.fn == 1 and s.brier == 1.0
