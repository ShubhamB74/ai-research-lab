"""CLI:  python -m failurelab.cli generate | baselines | selftest | llm-run"""
from __future__ import annotations

import argparse
import hashlib
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
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
