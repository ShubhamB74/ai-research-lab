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
