# Running LLM experiments (Phase 3 harness)

The harness was built and tested with a **scripted stand-in model** (no API access was available).
`AnthropicClient` is tested only against a fake SDK object. **Do step 3 (smoke test) before any real run.**

## 1. Install and generate tasks
```bash
pip install -e ".[dev,llm]"
make test
make generate            # 216 tasks in ./tasks (public/ for agents, private/ = answer key)
```

## 2. Verify the plumbing without an API key
```bash
python -m failurelab.cli selftest --tasks tasks --policy careful --max-tasks 12
python -m failurelab.cli selftest --tasks tasks --policy naive   --max-tasks 12
```
Expected: `careful` scores ~1.00 accuracy with 0 false alarms; `naive` is fooled on trapped tasks.
(This is a scripted model driving the real sandbox/tool loop/grader.)

## 3. Smoke test with a real model (small, cheap, read the transcripts)
```bash
export ANTHROPIC_API_KEY=...          # never commit this
python -m failurelab.cli llm-run --model <MODEL_ID> --conditions none \
    --reps 1 --max-tasks 6 --max-calls 60 --out runs_smoke
```
`<MODEL_ID>`: copy the current id from Anthropic's model docs. Then check, by hand:
- `runs_smoke/scores.jsonl`: `status` should be mostly `submitted`; look at `format_retries`, `n_python_calls`.
- Open 2 transcripts (`runs_smoke/rep0/*/transcripts/*.json`). Did the model call tools, read outputs, preregister first?
- `input_tokens` / `output_tokens` per run: multiply by the number of runs in your full plan for a cost estimate.
- `tripwire_hits`: any non-empty list means the model's code mentioned benchmark internals; read it.
If the adapter fails against the real API (for example a changed response shape or parameter), fix `AnthropicClient` in `src/failurelab/llm.py`; it is ~40 lines.

## 4. Plan, then run
```bash
python -m failurelab.cli llm-run --model <MODEL_ID> --dry-run --reps 3 --max-tasks 36 --out runs_llm
python -m failurelab.cli llm-run --model <MODEL_ID> --reps 3 --max-tasks 36 --max-calls 3000 --out runs_llm
```
- `--max-calls` is a **hard cap**; hitting it stops the run cleanly. API outages also stop the run (they are never scored as agent failures).
- **Resume**: re-run the same command. Tasks with a final transcript are reused with no model call.
- Loop order is rep, then condition, so an interrupted run still has all three conditions for rep 0.
- Conditions: `none`, `generic_hint`, `checklist` (see `src/failurelab/prompts.py`).
- Do not change prompts/tools/`max_turns` between reps of one experiment. Record the model id and git commit in `experiments/LOG.md`.

## 5. Isolation and its limits (read this)
The sandbox runs agent code in a separate Python process with a scrubbed environment (no API keys), a workdir
containing only `data.csv`, a timeout, a memory cap (Linux) and blocked `failurelab` imports. It is **not a
security boundary**: the process runs as your OS user and can read other files you can read, including
`tasks/private/` if it guesses the path, and it has network access unless you restrict it.
- Linux, drop network for the sandbox: `--sandbox-prefix "unshare -rn"` (starts fine in my tests; verify that
  `socket.create_connection(("1.1.1.1", 53))` fails inside it on your machine before trusting it).
- Stronger: run on a machine or container that holds no sensitive files, and keep `tasks/private/` somewhere
  the agent process cannot read (for example run the harness as a different user than the sandbox).
- The tripwire (`tripwire_hits`) is a human-review aid, not enforcement. Review flagged transcripts before
  trusting a result; exclude and report any run in which the model actually tried to read answer keys.

## 6. Reading the output
`scores.jsonl` has one row per (agent, condition, rep, task) with all graded fields, `status`, token usage and
tripwire hits. Repeated runs of the same task are **not independent**; the pooled intervals printed by the CLI
are optimistic. Phase 4 analyses cluster by task. Report non-`submitted` statuses; do not silently drop them.
