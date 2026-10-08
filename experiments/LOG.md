# Experiment log

Every experiment goes here, failures included. A dead end documented well is a result.
Add newest entries at the top. Copy the template.

## Template
### EXP-000 | YYYY-MM-DD | <short title>
- **Question / hypothesis:**
- **Mechanism, predicted direction, what would prove me wrong (written BEFORE running):**
- **Setup (seed range, commit hash, config):**
- **Result (numbers, intervals):**
- **Verdict:** survived | killed | inconclusive
- **Why (including bugs found in my own generator/grader):**
- **Next:**

---

### EXP-008 | (date) | Phase 5 tooling: what can be measured without inventing a taxonomy?
- **Design decisions:** (1) the failure taxonomy is not written in advance; only a-priori SEED codes plus open coding (`NEW:` codes) are provided. (2) A deterministic layer separates "did not look at the cue" from "looked but did not flag" using a keyword heuristic over the agent's code; its error must be audited against human labels, so the audit column and `detector_validity` exist. (3) Failures are oversampled for coding, so prevalence uses inverse-probability weights; a test shows weighted and raw shares differ. (4) The one-pager verifies the author's own prose: any number not in the results files is rejected.
- **Verified offline (134 tests):** detector on labelled positive and negative snippets for each family; all diagnosis classes; sampler weights sum to stratum sizes; kappa values; weighted prevalence equals the hand computation; manifest, regrade and analysis reproduction each detect tampering; pipeline runs on real harness transcripts.
- **Known limits:** the cue detector is untested against real agent code; task regeneration may differ across numpy versions (publish the generated tasks); no live transcripts exist, so no failure modes have been identified yet.
- **Next:** run the live experiment, then follow docs/CODING_GUIDE.md.

---

### EXP-007 | (date) | Phase 4 tooling: can the pre-registered rules reach a verdict at a feasible size, and does the pipeline guard itself?
- **Predicted (before running):** with 54 pairs and 3 reps, assumed recall 0.2 (baseline) vs 0.7 (checklist), H1 and H2 would both be supported most of the time; a null checklist effect would almost never be "supported".
- **Result (simulation, 60 runs per scenario):** H2 supported 83 percent when the effect is real, 0 percent when null. H1 supported only 55 percent: the clean-accuracy clause (lower bound above 0.9 at the Bonferroni level) is the bottleneck, not the recall clause.
- **Verdict:** half killed. The design cannot reliably confirm H1 at this size; either more pairs or accept that H1 may be inconclusive. This is why the power planner exists.
- **Bugs found by my own tests:** (1) the report's number-provenance scanner flagged the freeze timestamp as an uncomputed number; fixed by rendering provenance strings as literals, and the same for gate reasons. (2) Excluding a reviewed run made the design look incomplete; completeness is now judged before exclusions. (3) My "planted effect is confirmed" test depended on one lucky seed: on seed 0 the false-flag difference was +0.09 so H2 was correctly inconclusive. Rewrote it to assert properties across seeds (never falsified; supported in most).
- **Next:** pick the design with `power`, fill and freeze `docs/prereg.json`, commit the lock, run the live experiment (yours), review tripwires, run `analyze`.

---

### EXP-006 | (date) | Phase 3: does the harness add or hide anything? (scripted model, no API)
- **Predicted (before running):** a scripted model that preregisters, runs the baseline analysis inside the sandbox, reads the printed result and submits it will reproduce the direct baseline scores EXACTLY for naive and heuristic_careful on all three families; the sandbox hides API keys and answer-key paths; an API outage aborts a run instead of being scored.
- **Result:** exact match on 36 tasks x 2 agents; all sandbox and abort/resume tests passed (88 tests).
- **Bug found on the way:** `select_tasks` rotated through cells in sorted order, so a 12-task subset was all leakage (the first family). Caught by the self-test output, fixed by rotating across families first, test added.
- **Verdict:** survived, for the plumbing only. No real LLM has been run; the Anthropic adapter is tested against a fake SDK.
- **Decisions to remember:** the prompt discloses the scoring rule and the question already asks for flags, so `none` = detection when asked to look. Repeated runs are not independent (cluster by task in Phase 4).
- **Next:** run the live smoke test (docs/RUNNING_EXPERIMENTS.md step 3), freeze docs/PREREGISTRATION.md, then Phase 4.

---

### EXP-005 | (date) | Phase 2 validation battery on survivorship and multiple testing
- **Predicted (before running):** oracle 100%; heuristic_careful solves every cell with zero false alarms; naive fooled in both families; always-flag 100% false alarms; the cheap-checks agent catches only explicit cues.
- **Result (36 pairs per family, 57 tests):** all predictions held. Cheap checks: survivor_only_panel recall 1/1/0 (easy/medium/hard), backfill 1 across cells, visible_candidates 1, hidden_winner 0.
- **Verdict:** survived. Caveat: 6 tasks per cell, and the cues were defined by me, so "solvable" means solvable by someone who knows the trap types.
- **Next:** Phase 3 (LLM agent harness).

### EXP-004 | (date) | Can survivorship tasks plant a negative effect? (killed before coding)
- **Predicted:** if the true mean is negative, survivor-only data still shows a negative mean, so "a negative result survives survivorship bias" gives an identifiable truth.
- **Result (simulation, 120 names, 400 days, delist at -40%):** mu = 0 gives survivors t = +4.1 vs clean t = 0.1; mu = -0.001 gives survivors t = 0.0 (clean t = -9.0). Conditioning on survival erases the negative drift.
- **Verdict:** killed. Survivor-only data cannot identify a negative effect either.
- **Consequence:** adopted an identifiability convention (the supported answer for survivor-only tasks is "no effect") and only plant null/positive worlds there; backfill, where filtering recovers the data, keeps all signs. Documented in the README and generator docstring.
- **Lesson:** simulate the bias before designing the ground truth around it.

---

### EXP-003 | (date) | Phase 1: do the difficulty levels separate kinds of diligence?
- **Predicted (before running):** a "too good to be true" effect-size check (`sanity_check`) catches easy leaks (corr ~0.4), misses subtle ones; the reverse-engineering agent solves every cell.
- **Result (36 pairs, n=12 per difficulty):** sanity_check flag recall 1.00 / 0.17 / 0.00 (easy/medium/hard); heuristic_careful 1.00 across all; naive 0.00.
- **Verdict:** survived. Side finding: on hard leaks the naive agent's *conclusion* is often unchanged (fooled 0.17), so conclusion accuracy alone under-measures failure.
- **Caveat:** detector thresholds were set by me after seeing the mechanisms.
- **Next:** Phase 2 (survivorship, multiple testing) with the same validation battery.

### EXP-002 | (date) | Careful agent failed 2 tasks: a ground-truth validity bug
- **Predicted:** heuristic_careful solves all 36 tasks of the 9-cell design.
- **Result:** 2 failures, both twins of one null world (phi = 0) where an honest test gave t = 3.09 by sampling luck.
- **Verdict:** killed my assumption that "planted truth" = "answer an honest analyst gets". The agent was right; the task was ambiguous.
- **Fix:** deterministic rejection sampling so null worlds have |t|<2 and effect worlds |t|>4.5 with the planted sign; recorded `resample_attempt` and `honest_t`; added a calibration test over 60 seeds. Disclosed as a selection effect in the README.
- **Lesson:** a benchmark can punish correct statistics if truth is defined by the generator rather than by what the realised sample supports.

---

### EXP-001 | (date) | Benchmark validation: baselines on 30 leakage pairs
- **Question:** Does the benchmark discriminate before any LLM is tested?
- **Predicted (before running):** oracle 100%; naive fooled on most trapped tasks; always-flag has 100% false alarms on clean; heuristic_careful solves everything.
- **Result:** matches the README table (naive fooled 0.70, always-flag false-flag 1.00, heuristic_careful 1.00/1.00, random at chance).
- **Verdict:** survived (the benchmark discriminates as designed).
- **Bugs found on the way:** my first "variant is not revealed" test failed because `allowed_flags` legitimately lists all trap codes; fixed the test, not the generator, and added a check that the list is identical across variants.
- **Next:** Phase 1 (more leakage mechanisms, difficulty levels).
