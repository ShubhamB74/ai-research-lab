# Building the failure taxonomy from transcripts

The taxonomy is an OUTPUT of reading real transcripts. The tooling samples, renders, measures agreement and weights;
it does not decide what the failure modes are. Nothing here applies until live runs exist.

## 1. Deterministic layer (no judgement)
```bash
python -m failurelab.cli behavior --runs runs_llm/modelA runs_llm/modelB --out report
```
For every run: did the agent's code LOOK at the property that exposes the flaw (the cue), and did it FLAG the flaw?
On trapped tasks: `detected`, `saw_not_flagged` (looked, did not flag), `never_looked`, `flagged_without_looking`.
`report/behavior.json` holds P(looked), P(flag | looked), P(flag | not looked), and the share of misses that never looked,
with task-clustered intervals. "Looked" is a keyword heuristic; step 4 measures its error.

## 2. Sample and read
```bash
python -m failurelab.cli coding-sheet --runs runs_llm/modelA runs_llm/modelB --out coding --per-stratum 3
```
Strata are (model, condition, family, graded outcome). Failures are deliberately oversampled, so every estimate must use
the `weight` column (stratum size divided by sample size); never report raw shares of the coded sample.
`coding/transcripts_md/` has one readable transcript per sampled run. The sheet shows trap family and graded outcome
(failure analysis needs them) and hides the automatic detector's output.

## 3. Code independently (two coders on at least a fifth of the runs)
Fill `codes` with semicolon-separated codes. Start from the seed codes below. If nothing fits, write `NEW:<short_name>`
and describe it in `notes`; put the cell number or quote in `evidence`. Code what the agent DID and SAID, not what you
assume it understood. Seed codes are a-priori guesses from the benchmark design; expect to rename, split and merge them.

| Code | Meaning |
|---|---|
| NEVER_LOOKED | The code never examined the data property that exposes the flaw. |
| SAW_DISMISSED | The agent's own output showed the cue and it explained it away or ignored it. |
| MISDIAGNOSED | Flagged a different problem, or the right one with a wrong explanation. |
| OVERTRUST_STATS | Concluded from a significant statistic without questioning how the sample, feature or search was produced. |
| CORRECTED_BUT_MISJUDGED | Applied a correction or adjustment, then drew the wrong conclusion anyway (or the reverse). |
| GENERIC_HEDGE | Mentioned possible problems in general terms with no specific flag or test. |
| PREMATURE_STOP | Submitted after minimal analysis, without trying to break the result. |
| UNSUPPORTED_FLAG | Flagged a problem on a clean twin with no evidence in its own analysis. |
| TOOLING_FAILURE | Outcome driven by tool or format trouble, not research reasoning. |
| OTHER | Anything else; describe it. |

## 4. Audit the "looked" detector
Fill `cue_inspected_human` (`y` or `n`) for every sampled run: did the agent's code examine the cue for that trap family?
`detector_validity` (in `coding.py`) then gives precision, recall and kappa of the keyword detector against you.
Report these next to every `behavior` result. If kappa is low, say so and rely on the human labels.

## 5. Reconcile and measure agreement
```bash
python -m failurelab.cli agreement coding/sheet_coderA.csv coding/sheet_coderB.csv
```
Per-code percent agreement and Cohen's kappa. Discuss disagreements, revise code definitions, recode, and write the final
codes into one sheet. Record each codebook revision (what changed and why) in `experiments/LOG.md`.

## 6. Weighted prevalence
```bash
python -m failurelab.cli taxonomy --sheet coding/final_sheet.csv --out report
```
Inverse-probability-weighted share of failing runs carrying each code, per model and condition, with a cluster bootstrap by pair.
These are descriptive, exploratory results, not part of the pre-registered tests.

## 7. One page
```bash
python -m failurelab.cli onepager --report report                # draft, interpretation marked missing
python -m failurelab.cli onepager --report report --final         # requires report/interpretation.md
```
Computed numbers are written by code. Your interpretation paragraph is checked: every number in it must appear in the
results files (two decimals, or an integer). Put a deliberate non-result number in backticks to exempt it.

## 8. Reproducibility
```bash
python -m failurelab.cli manifest                                  # at freeze time; commit experiments/task_manifest.json
python -m failurelab.cli repro-check --runs runs_llm/modelA runs_llm/modelB
```
Checks: task set matches the manifest; tasks regenerate from the pre-registered seed (may fail on a different numpy, see below);
every raw submission regrades to the stored scores; the analysis reproduces `results.json` byte for byte.
numpy does not guarantee identical random streams across major versions, so ALSO publish the generated task set and its
manifest as a release artefact; regeneration is a bonus check, not the foundation.
