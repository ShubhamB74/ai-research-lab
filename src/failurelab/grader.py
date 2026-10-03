"""Deterministic grader. No LLM, no text matching: only comparisons against the answer key."""
from __future__ import annotations

from dataclasses import dataclass

from .schema import GroundTruth, Submission


@dataclass(frozen=True)
class TaskScore:
    task_id: str
    family: str
    variant: str
    valid: bool                 # False if the submission failed schema validation
    conclusion_correct: bool    # effect_exists AND direction match the honest ground truth
    fooled: bool                # trapped task, claimed a (wrong) effect: the trap worked on the agent
    right_for_wrong_reason: bool  # trapped task, correct conclusion, but missed the planted flaw
    tp: int                     # planted flaws correctly flagged
    fp: int                     # flags raised that are not actually present
    fn: int                     # planted flaws missed
    brier: float                # (confidence - correct)^2, lower is better calibrated
    mechanism: str = ""          # from the answer key; used only for breakdown tables
    difficulty: str = ""


def grade(sub: Submission, truth: GroundTruth) -> TaskScore:
    truth_flags, sub_flags = set(truth.traps_present), set(sub.flags)
    correct = (sub.effect_exists == truth.effect_exists
               and (not truth.effect_exists or sub.direction == truth.true_direction))
    trapped = bool(truth_flags)
    tp = len(truth_flags & sub_flags)
    return TaskScore(
        task_id=truth.task_id, family=truth.family, variant=truth.variant, valid=True,
        conclusion_correct=correct,
        fooled=trapped and sub.effect_exists and not correct,
        right_for_wrong_reason=trapped and correct and not truth_flags <= sub_flags,
        tp=tp, fp=len(sub_flags - truth_flags), fn=len(truth_flags - sub_flags),
        brier=(sub.confidence - float(correct)) ** 2,
        mechanism=str(truth.params.get("mechanism", "")),
        difficulty=str(truth.params.get("difficulty", "")),
    )


def grade_invalid(truth: GroundTruth) -> TaskScore:
    """An unparseable/invalid submission counts as a failure, never as a crash or a skip."""
    return TaskScore(
        task_id=truth.task_id, family=truth.family, variant=truth.variant, valid=False,
        conclusion_correct=False, fooled=False, right_for_wrong_reason=False,
        tp=0, fp=0, fn=len(truth.traps_present), brier=1.0,
        mechanism=str(truth.params.get("mechanism", "")),
        difficulty=str(truth.params.get("difficulty", "")),
    )
