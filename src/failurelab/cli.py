"""CLI:  python -m failurelab.cli generate | baselines | selftest | llm-run | freeze | analyze | review | power | behavior | coding-sheet | agreement | taxonomy | onepager | manifest | repro-check"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .agents import (AlwaysFlagAgent, HeuristicCarefulAgent, NaiveAgent, OracleAgent,
                     RandomAgent, SanityCheckAgent)
from .generators import REGISTRY
from .generators.base import write_task
from .runner import run_agent
from .scorecard import breakdown, summarize, to_frame, to_markdown
from .taxonomy import Variant


def opaque_id(family: str, variant: str, seed: int, salt: str) -> str:
    """Task IDs must reveal nothing about family/variant."""
    return "t" + hashlib.sha1(f"{salt}|{family}|{variant}|{seed}".encode()).hexdigest()[:10]


def cmd_generate(args: argparse.Namespace) -> None:
    out = Path(args.out)
    for family, gen in REGISTRY.items():
        for i in range(args.n):
            seed = args.seed + i
            ids = {v: opaque_id(family.value, v.value, seed, args.salt) for v in Variant}
            for v in Variant:
                twin = ids[Variant.TRAPPED if v is Variant.CLEAN else Variant.CLEAN]
                write_task(ids[v], twin, family, v, gen.generate(seed, v),
                           out / "public", out / "private")
    print(f"wrote {args.n * len(Variant) * len(REGISTRY)} tasks to {out}/(public|private)")


def cmd_baselines(args: argparse.Namespace) -> None:
    tasks = Path(args.tasks)
    pub, priv, runs = tasks / "public", tasks / "private", Path(args.out)
    agents = [OracleAgent(priv), HeuristicCarefulAgent(), SanityCheckAgent(), NaiveAgent(),
              AlwaysFlagAgent(), RandomAgent(seed=args.seed)]
    by_agent = {a.name: run_agent(a, pub, priv, runs) for a in agents}
    df = to_frame(by_agent)
    parts = [("Overall (family x variant)", summarize(df)),
             ("Difficulty ladder (trapped tasks)", breakdown(df, "difficulty")),
             ("By mechanism (trapped tasks)", breakdown(df, "mechanism")),
             ("By cell: mechanism/difficulty (trapped tasks)", breakdown(df, "cell"))]
    md = "\n\n".join(f"### {title}\n\n{to_markdown(t)}" for title, t in parts)
    print(md)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/baseline_scorecard.md").write_text(md + "\n")


def _print_scorecard(df) -> None:
    print(to_markdown(summarize(df)))
    print("\nNOTE: repeated runs of a task are not independent; these intervals are optimistic. "
          "Phase 4 uses task-clustered intervals.")


def cmd_selftest(args: argparse.Namespace) -> None:
    """Run a SCRIPTED model through the full harness (sandbox, tool loop, transcripts, grading).
    Needs no API key. Verifies plumbing only, not any LLM."""
    from .agents.llm_agent import LLMAgent
    from .experiment import select_tasks
    from .llm import ScriptedClient
    from .scripted_policies import reference_policy
    cls = {"naive": "NaiveAgent", "careful": "HeuristicCarefulAgent"}[args.policy]
    tasks, runs = Path(args.tasks), Path(args.out)
    client = ScriptedClient(f"scripted-{args.policy}", reference_policy(cls))
    agent = LLMAgent(client, "none", sandbox_kwargs={"allow_failurelab": True},
                     transcript_dir=runs / f"scripted-{args.policy}_none" / "transcripts")
    ids = select_tasks(tasks / "private", args.max_tasks, args.seed)
    scores = run_agent(agent, tasks / "public", tasks / "private", runs, task_ids=ids)
    _print_scorecard(to_frame({agent.name: scores}))


def cmd_llm_run(args: argparse.Namespace) -> None:
    from .experiment import plan, run_llm_experiment
    conditions = args.conditions.split(",")
    print(plan(Path(args.tasks), conditions, args.reps, args.max_tasks, args.max_turns, args.seed))
    if args.dry_run:
        return
    from .llm import AnthropicClient, BudgetedClient, RunAborted
    client = BudgetedClient(AnthropicClient(args.model, temperature=args.temperature, max_tokens=args.max_tokens),
                            args.max_calls)
    try:
        df = run_llm_experiment(client, Path(args.tasks), Path(args.out), conditions, args.reps,
                                args.max_tasks, args.max_turns, args.seed,
                                sandbox_kwargs={"cmd_prefix": args.sandbox_prefix.split()} if args.sandbox_prefix else None)
    except RunAborted as exc:
        print(f"\nRun stopped: {exc}\nCompleted tasks are saved; re-run the same command to resume.")
        return
    _print_scorecard(df)


def cmd_freeze(args: argparse.Namespace) -> None:
    from .prereg import freeze
    rec = freeze(args.prereg, args.doc, args.lock)
    print(json.dumps(rec, indent=2))
    print("\nNow COMMIT docs/prereg.json, docs/PREREGISTRATION.md and the lock file BEFORE the first live run.")


def cmd_analyze(args: argparse.Namespace) -> None:
    from .study import StudyError, run_study
    try:
        res = run_study(args.tasks, args.runs, args.out, args.prereg, args.doc, args.lock, args.exclusions,
                        args.allow_incomplete, args.allow_unreviewed)
    except StudyError as exc:
        raise SystemExit(f"analysis stopped: {exc}")
    print(f"{res['status']}" + (f": {'; '.join(res['reasons'])}" if res["reasons"] else ""))
    for m, e in res["models"].items():
        print(f"  {m}: " + ", ".join(f"{h}={v['verdict']}" for h, v in e["hypotheses"].items()))
    print(f"wrote {args.out}/report.md, results.json, provenance.json")


def cmd_review(args: argparse.Namespace) -> None:
    from .study import tripwire_review
    items = tripwire_review(args.runs)
    print(f"{len(items)} tripwire-flagged run(s)")
    for it in items:
        print(f"\n[{it['model']} | {it['condition']} | rep{it['rep']} | {it['task_id']}] hits={it['hits']}")
        for ln in it["lines"][:8]:
            print("    " + ln[:200])
    if items:
        print('\nRecord a decision for each in the exclusions file: '
              '{"model","condition","rep","task_id","action":"keep"|"exclude","reason":"..."}')


def cmd_power(args: argparse.Namespace) -> None:
    from .power import simulate_power
    res = simulate_power(args.pairs, args.reps, args.recall_none, args.recall_checklist, args.ffr_none,
                         args.ffr_checklist, args.clean_acc, args.trapped_acc, args.pair_sd, args.models,
                         args.sims, seed=args.seed)
    print(f"pairs={args.pairs} reps={args.reps} sims={args.sims} (ASSUMED rates; try a range)")
    for h, t in res.items():
        print(f"  {h}: " + ", ".join(f"{k}={v:.2f}" for k, v in t.items() if k != "not_evaluable"))


def _scores_with_pairs(tasks: str, runs):
    from .study import attach_pairs, load_runs
    return attach_pairs(load_runs(runs), Path(tasks) / "private")


def cmd_behavior(args: argparse.Namespace) -> None:
    from .behavior import behavior_table, summarize
    from .inference import Study
    df = _scores_with_pairs(args.tasks, args.runs)
    beh = behavior_table(df, Path(args.tasks) / "private")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    beh.drop(columns=[]).to_csv(out / "behavior.csv", index=False)
    summ = summarize(beh, Study(beh.pair_id, B=args.bootstrap_B, seed=args.seed))
    (out / "behavior.json").write_text(json.dumps(summ, indent=2, sort_keys=True))
    print(f"wrote {out}/behavior.csv and behavior.json ({len(beh)} runs)")
    for m, conds in summ.items():
        for c, e in conds.items():
            o = e["overall"]
            print(f"  {m} | {c}: looked {o['looked_rate']['point']:.2f}, flag|looked {o['flag_given_looked']['point']:.2f}, "
                  f"flag|not looked {o['flag_given_not_looked']['point']:.2f}")
    print("NOTE: 'looked' is a keyword heuristic; audit a sample by hand (see docs/CODING_GUIDE.md).")


def cmd_coding_sheet(args: argparse.Namespace) -> None:
    from .coding import sample_for_coding, write_coding_sheet
    df = _scores_with_pairs(args.tasks, args.runs)
    sample = sample_for_coding(df, args.per_stratum, args.seed)
    path = write_coding_sheet(sample, Path(args.tasks) / "private", args.out)
    print(f"{len(sample)} runs sampled from {sample.stratum.nunique()} strata -> {path}")


def cmd_agreement(args: argparse.Namespace) -> None:
    import pandas as pd
    from .coding import kappa_table, validate_sheet
    a, b = pd.read_csv(args.sheets[0]), pd.read_csv(args.sheets[1])
    for name, sh in (("A", a), ("B", b)):
        for prob in validate_sheet(sh):
            print(f"coder {name}: {prob}")
    print(kappa_table(a, b).to_string(index=False))
    print("\nReconcile disagreements, write the final codes into one sheet, then run `taxonomy`.")


def cmd_taxonomy(args: argparse.Namespace) -> None:
    import pandas as pd
    from .coding import prevalence_table, validate_sheet
    sheet = pd.read_csv(args.sheet)
    for prob in validate_sheet(sheet):
        print("WARNING " + prob)
    tab = prevalence_table(sheet, B=args.bootstrap_B, seed=args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "taxonomy.json").write_text(json.dumps(tab.to_dict(orient="records"), indent=2))
    print(tab.to_string(index=False))


def cmd_onepager(args: argparse.Namespace) -> None:
    from .onepager import OnePagerError, build_onepager
    rd = Path(args.report)
    try:
        text = build_onepager(rd, rd / "behavior.json", rd / "taxonomy.json", args.interpretation or rd / "interpretation.md",
                              args.final, args.models, args.baseline)
    except OnePagerError as exc:
        raise SystemExit(f"one-pager stopped: {exc}")
    (rd / "one_pager.md").write_text(text)
    print(f"wrote {rd}/one_pager.md ({len(text.split())} words)")


def cmd_manifest(args: argparse.Namespace) -> None:
    from .repro import write_manifest
    rec = write_manifest(Path(args.tasks), args.out)
    print(f"manifest: {rec['n_files']} files (numpy {rec['numpy']}, pandas {rec['pandas']}) -> {args.out}")
    print("Commit it with the pre-registration lock.")


def cmd_repro(args: argparse.Namespace) -> None:
    from .prereg import load
    from .repro import repro_check
    plan = load(args.prereg)
    kw = dict(tasks_dir=args.tasks, run_dirs=args.runs, prereg_path=args.prereg, doc_path=args.doc, lock_path=args.lock,
              exclusions_path=args.exclusions, allow_incomplete=args.allow_incomplete, allow_unreviewed=args.allow_unreviewed)
    results = repro_check(args.tasks, args.runs, args.report, args.manifest, plan, kw)
    failed = False
    for name, problems in results.items():
        print(f"{'PASS' if not problems else 'FAIL'}  {name}")
        for p in problems[:10]:
            print("      " + p)
        failed |= bool(problems)
    raise SystemExit(1 if failed else 0)


def main() -> None:
    p = argparse.ArgumentParser(prog="failurelab")
    sub = p.add_subparsers(required=True)
    g = sub.add_parser("generate")
    g.add_argument("--n", type=int, default=36, help="seeds per family (x2 variants); use a multiple of 9")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--salt", default="failurelab-v0", help="change to re-randomise task ids")
    g.add_argument("--out", default="tasks")
    g.set_defaults(fn=cmd_generate)
    b = sub.add_parser("baselines")
    b.add_argument("--tasks", default="tasks")
    b.add_argument("--out", default="runs")
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(fn=cmd_baselines)
    st = sub.add_parser("selftest", help="scripted model through the full harness (no API key needed)")
    st.add_argument("--tasks", default="tasks")
    st.add_argument("--out", default="runs_selftest")
    st.add_argument("--policy", choices=["naive", "careful"], default="careful")
    st.add_argument("--max-tasks", type=int, default=12)
    st.add_argument("--seed", type=int, default=0)
    st.set_defaults(fn=cmd_selftest)
    lr = sub.add_parser("llm-run", help="run an Anthropic model through the harness (needs ANTHROPIC_API_KEY)")
    lr.add_argument("--tasks", default="tasks")
    lr.add_argument("--out", default="runs_llm")
    lr.add_argument("--model", required=True, help="model id; check the current list in Anthropic's docs")
    lr.add_argument("--conditions", default="none,generic_hint,checklist")
    lr.add_argument("--reps", type=int, default=1)
    lr.add_argument("--max-tasks", type=int, default=None, help="use balanced clean/trapped pairs; omit for all")
    lr.add_argument("--max-turns", type=int, default=15)
    lr.add_argument("--max-calls", type=int, default=200, help="hard cap on total model calls")
    lr.add_argument("--max-tokens", type=int, default=4096)
    lr.add_argument("--temperature", type=float, default=None, help="omit to use the API default")
    lr.add_argument("--seed", type=int, default=0, help="task-subset selection seed")
    lr.add_argument("--sandbox-prefix", default="", help='command prefix for the sandbox, e.g. "unshare -rn"')
    lr.add_argument("--dry-run", action="store_true")
    lr.set_defaults(fn=cmd_llm_run)
    fz = sub.add_parser("freeze", help="validate docs/prereg.json and write the hash lock")
    fz.add_argument("--prereg", default="docs/prereg.json")
    fz.add_argument("--doc", default="docs/PREREGISTRATION.md")
    fz.add_argument("--lock", default="experiments/prereg.lock")
    fz.set_defaults(fn=cmd_freeze)
    an = sub.add_parser("analyze", help="pre-registered analysis and report from one or more run directories")
    an.add_argument("--tasks", default="tasks")
    an.add_argument("--runs", nargs="+", required=True, help="directories containing scores.jsonl (one per model)")
    an.add_argument("--out", default="report")
    an.add_argument("--prereg", default="docs/prereg.json")
    an.add_argument("--doc", default="docs/PREREGISTRATION.md")
    an.add_argument("--lock", default="experiments/prereg.lock")
    an.add_argument("--exclusions", default="experiments/exclusions.json")
    an.add_argument("--allow-incomplete", action="store_true")
    an.add_argument("--allow-unreviewed", action="store_true")
    an.set_defaults(fn=cmd_analyze)
    rv = sub.add_parser("review", help="list tripwire-flagged runs with the offending code lines")
    rv.add_argument("--runs", nargs="+", required=True)
    rv.set_defaults(fn=cmd_review)
    pw = sub.add_parser("power", help="simulate how often the pre-registered rules would reach each verdict")
    pw.add_argument("--pairs", type=int, required=True)
    pw.add_argument("--reps", type=int, required=True)
    pw.add_argument("--recall-none", type=float, required=True)
    pw.add_argument("--recall-checklist", type=float, required=True)
    pw.add_argument("--ffr-none", type=float, default=0.05)
    pw.add_argument("--ffr-checklist", type=float, default=0.05)
    pw.add_argument("--clean-acc", type=float, default=0.95)
    pw.add_argument("--trapped-acc", type=float, default=0.5)
    pw.add_argument("--pair-sd", type=float, default=1.0, help="between-pair heterogeneity (logit scale)")
    pw.add_argument("--models", type=int, default=1)
    pw.add_argument("--sims", type=int, default=200)
    pw.add_argument("--seed", type=int, default=0)
    pw.set_defaults(fn=cmd_power)
    bh = sub.add_parser("behavior", help="did each run's code look at the cue, and did looking lead to a flag?")
    bh.add_argument("--tasks", default="tasks"); bh.add_argument("--runs", nargs="+", required=True)
    bh.add_argument("--out", default="report"); bh.add_argument("--bootstrap-B", type=int, default=5000)
    bh.add_argument("--seed", type=int, default=12345); bh.set_defaults(fn=cmd_behavior)
    cs = sub.add_parser("coding-sheet", help="stratified sample of runs + readable transcripts + blank coding sheet")
    cs.add_argument("--tasks", default="tasks"); cs.add_argument("--runs", nargs="+", required=True)
    cs.add_argument("--out", default="coding"); cs.add_argument("--per-stratum", type=int, default=3)
    cs.add_argument("--seed", type=int, default=0); cs.set_defaults(fn=cmd_coding_sheet)
    ag = sub.add_parser("agreement", help="per-code agreement between two coders' sheets")
    ag.add_argument("sheets", nargs=2); ag.set_defaults(fn=cmd_agreement)
    tx = sub.add_parser("taxonomy", help="weighted prevalence of codes from the final adjudicated sheet")
    tx.add_argument("--sheet", required=True); tx.add_argument("--out", default="report")
    tx.add_argument("--bootstrap-B", type=int, default=5000); tx.add_argument("--seed", type=int, default=12345)
    tx.set_defaults(fn=cmd_taxonomy)
    op = sub.add_parser("onepager", help="one-page summary; numbers computed, interpretation verified")
    op.add_argument("--report", default="report"); op.add_argument("--interpretation", default=None)
    op.add_argument("--models", nargs="*", default=None); op.add_argument("--baseline", default="none")
    op.add_argument("--final", action="store_true"); op.set_defaults(fn=cmd_onepager)
    mf = sub.add_parser("manifest", help="hash the generated task set")
    mf.add_argument("--tasks", default="tasks"); mf.add_argument("--out", default="experiments/task_manifest.json")
    mf.set_defaults(fn=cmd_manifest)
    rc = sub.add_parser("repro-check", help="regenerate tasks, regrade raw submissions, re-run the analysis; compare to stored files")
    rc.add_argument("--tasks", default="tasks"); rc.add_argument("--runs", nargs="+", required=True)
    rc.add_argument("--report", default="report"); rc.add_argument("--manifest", default="experiments/task_manifest.json")
    rc.add_argument("--prereg", default="docs/prereg.json"); rc.add_argument("--doc", default="docs/PREREGISTRATION.md")
    rc.add_argument("--lock", default="experiments/prereg.lock"); rc.add_argument("--exclusions", default="experiments/exclusions.json")
    rc.add_argument("--allow-incomplete", action="store_true"); rc.add_argument("--allow-unreviewed", action="store_true")
    rc.set_defaults(fn=cmd_repro)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
