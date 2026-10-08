import json

import pandas as pd
import pytest

from failurelab.behavior import summarize
from failurelab.inference import Study
from failurelab.onepager import OnePagerError, allowed_numbers, build_onepager, verify_prose
from failurelab.study import run_study
from synth import behavior_frame, build


def _make(tmp, models=("m1",), pairs=40):
    run_study(**build(tmp, pairs=pairs, reps=2, models=models))
    rd = tmp / "report"
    beh = behavior_frame(pairs=pairs, models=models)
    (rd / "behavior.json").write_text(json.dumps(summarize(beh, Study(beh.pair_id, B=500, seed=1))))
    tax = [{"model": m, "condition": "none", "code": c, "n_coded": 12, "weighted_prevalence": p, "lo": p - 0.1, "hi": p + 0.1}
           for m in models for c, p in (("NEVER_LOOKED", 0.55), ("SAW_DISMISSED", 0.25), ("PREMATURE_STOP", 0.1), ("OTHER", 0.05))]
    (rd / "taxonomy.json").write_text(json.dumps(tax))
    return rd


def _build(rd, **kw):
    return build_onepager(rd, rd / "behavior.json", rd / "taxonomy.json", **kw)


def test_draft_one_pager_is_short_complete_and_marks_the_missing_interpretation(tmp_path):
    rd = _make(tmp_path)
    text = _build(rd)
    assert len(text.split()) <= 700
    for part in ("**Question.**", "**Method.**", "**Pre-registered results.**", "**Where agents fail**", "**Failure modes**",
                 "**Limitations.**", "**Reproduce.**", "DRAFT: interpretation not yet written", "`NEVER_LOOKED`"):
        assert part in text, part
    assert text.count("`NEVER_LOOKED`") == 1 and "`OTHER`" not in text          # top three codes per model only


def test_final_requires_a_human_written_interpretation(tmp_path):
    rd = _make(tmp_path)
    with pytest.raises(OnePagerError):
        _build(rd, final=True)


def test_interpretation_numbers_must_exist_in_the_results(tmp_path):
    rd = _make(tmp_path)
    res = json.loads((rd / "results.json").read_text())
    real = f"{res['models']['m1']['hypotheses']['H1']['estimates']['recall_baseline']['point']:.2f}"
    good, bad = rd / "good.md", rd / "bad.md"
    good.write_text(f"Baseline recall was {real}, mostly because agents never looked at the cue.")
    bad.write_text("Baseline recall was 0.123, mostly because agents never looked at the cue.")
    assert real in _build(rd, interpretation_path=good, final=True)
    with pytest.raises(OnePagerError):
        _build(rd, interpretation_path=bad)
    assert verify_prose("Used `0.123` as a deliberate constant and word six.", allowed_numbers(res)) == []   # backticks exempt


def test_length_and_model_limits_are_enforced(tmp_path):
    rd = _make(tmp_path / "a")
    with pytest.raises(OnePagerError):
        _build(rd, max_words=60)
    rd4 = _make(tmp_path / "b", models=("a", "b", "c", "d"), pairs=12)
    with pytest.raises(OnePagerError):
        _build(rd4)
    assert "`a`" in _build(rd4, models=["a", "b"])
