# Pre-registration

**Status: DRAFT until `python -m failurelab.cli freeze` has been run and the lock committed.**
Machine-readable twin: `docs/prereg.json` (the analysis reads thresholds from it; a test checks that every
threshold below matches it). Both files are hashed by the lock. After freezing, do not edit either file;
record any change under Amendments instead. The analysis labels results NON-CONFIRMATORY if either file changed.

A lock proves the plan did not change after it was written. It does not prove it was written before the data
existed: commit the lock before the first live run and keep the commit hash.

## Design (filled in `docs/prereg.json` before freezing)
- Models, repetitions per task, and number of tasks (`max_tasks`, always whole clean/trapped pairs) are set in the JSON.
  Choose them with `python -m failurelab.cli power` so the design can actually reach a verdict.
- Conditions: `none` (baseline), `generic_hint`, `checklist` (treatment). Prompts, tools and `max_turns` are fixed for the whole experiment.
- Tasks come from the generator with the seed and salt recorded in the JSON; the analysed subset is chosen by `select_tasks`.

## Primary population and exclusions
All runs count. Invalid, non-submitted and out-of-turns runs are failures, not missing data. The only exclusions are
runs whose code touched benchmark internals (tripwire) that a human reviewed and marked `exclude` in
`experiments/exclusions.json`, with a reason. Every tripwire-flagged run needs a keep/exclude entry.

## Inference
Percentile cluster bootstrap. The cluster is a clean/trapped pair: the task, its twin, every repetition and every
condition. Confirmatory intervals have level 1 minus 0.05 divided by (4 hypotheses times the number of models), i.e.
Bonferroni. Everything else (including H5) is exploratory at the exploratory level in the JSON and is labelled so.
Paired contrasts (H2, H4) reuse the same bootstrap draws for both sides.

## Hypotheses and decision rules
Verdicts are `supported`, `falsified` or `inconclusive`. Inconclusive is a statement about precision, not evidence of no effect.

**H1 (headline).** Unprompted flag recall on trapped tasks is low while clean accuracy is high.
Mechanism: agents execute analyses well but do not question hidden assumptions unprompted.
Supported if the upper bound of baseline recall is below 0.5 and the lower bound of baseline clean accuracy is above 0.9.
Falsified if baseline recall is at least 0.5 and its lower bound is above 0.4. Otherwise inconclusive.

**H2 (prompting).** The checklist raises recall without a false-alarm cost. Failure is attention, not capability.
Let delta recall be recall(checklist) minus recall(baseline) and delta false be the same difference for the false-flag rate.
Supported if delta recall is at least 0.2, its lower bound is above 0, and the upper bound of delta false is at most 0.1.
Falsified if the upper bound of delta recall is below 0.1, or the lower bound of delta false is above 0.1. Otherwise inconclusive.

**H3 (right for the wrong reason).** Among correct trapped answers in the baseline condition, a substantial share missed the planted flaw.
Supported if that share is at least 0.25 and its lower bound is above 0.1. Falsified if its upper bound is below 0.1.

**H4 (calibration).** Brier score is worse on trapped tasks than on clean twins in the baseline condition.
Supported if the lower bound of the difference (trapped minus clean) is above 0. Falsified if its upper bound is at most 0.

**H5 (exploratory, no verdict).** Recall differs by family; leakage, which needs reasoning about time and ordering, is expected to be hardest.

## Data-quality gates
Results are CONFIRMATORY only if: the lock is intact; every model x condition x repetition x task is present; every
tripwire-flagged run has a review entry; and the non-submitted share of each model is at most the limit in the JSON.
Otherwise the verdicts are still computed but labelled NON-CONFIRMATORY with the reasons.

## Amendments
Changes made to this draft before it was frozen (so they are not post-hoc):
- Replaced the Holm adjustment with Bonferroni-level confirmatory intervals; simpler to combine with interval-based verdicts.
- Rewrote every verdict as an explicit three-way rule on interval bounds (the original H1 falsification rule is kept; support rules were added).
- Specified the bootstrap cluster (the clean/trapped pair) and the exclusion procedure.
Changes after freezing: none yet.
