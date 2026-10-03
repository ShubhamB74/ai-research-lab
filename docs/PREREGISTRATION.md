# Pre-registration (FREEZE THIS BEFORE PHASE 4 RUNS; commit it, then do not edit)

Status: DRAFT. Thresholds below are proposals; confirm or change them, then commit before running
any LLM experiment. Later changes go in an "Amendments" section with a date and reason.

## Setup
- Tasks: paired clean/trapped, families implemented by Phase 2, N seeds per family (set N here: __).
- Conditions: `none`, `generic_hint`, `checklist`. Models: (list here). k runs per task-condition: __.
- Primary analysis: per-condition metrics with Wilson 95% intervals; no model-ranking claims unless
  intervals clearly separate.

## H1 (headline)
Under `none`, flag recall on trapped tasks is below 50%, while accuracy on clean twins is above 90%.
- Mechanism: agents execute analysis well but do not question hidden assumptions unprompted.
- Predicted direction: recall(trapped) << accuracy(clean).
- **Falsified if** recall(trapped, none) >= 50% with the lower CI bound above 40%.

## H2 (prompting)
A `checklist` prompt raises flag recall by at least 20 points over `none` without raising the
false-flag rate on clean twins by more than 10 points.
- Mechanism: failure is attention, not capability.
- **Falsified if** recall gain < 10 points, or false-flag rate rises > 10 points.

## H3 (fooled vs lucky)
Among trapped tasks answered correctly under `none`, at least 25% are "right for the wrong reason"
(conclusion correct, flaw not flagged).
- **Falsified if** that share is below 10%.

## H4 (calibration)
Mean Brier score is worse on trapped than on clean tasks (agents are equally confident when wrong).
- **Falsified if** Brier(trapped) <= Brier(clean).

## H5 (family difficulty)
Detection recall differs by family; the family that requires reasoning about time/ordering
(leakage) is hardest. Exploratory: no falsification threshold; report ordering with intervals.

## Multiple-testing note
H1-H4 are confirmatory (4 tests): report Holm-adjusted conclusions. H5 and anything found later is
exploratory and must be labelled so.

## Amendments
(none yet)
