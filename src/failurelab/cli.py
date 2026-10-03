"""CLI:  python -m failurelab.cli generate | baselines"""
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
             ("By mechanism (trapped tasks)", breakdown(df, "mechanism"))]
    md = "\n\n".join(f"### {title}\n\n{to_markdown(t)}" for title, t in parts)
    print(md)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/baseline_scorecard.md").write_text(md + "\n")


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
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
