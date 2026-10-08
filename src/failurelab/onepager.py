"""One-page summary generator (Phase 5).

Everything computed (verdicts, estimates, rates) is written through `Report`, so a number that was not
computed cannot appear. The AUTHOR'S interpretation paragraph is human prose; it is verified too: every
number in it must equal a number present in the results files (formatted to two decimals, or an integer).
Wrap a deliberate non-result number in backticks to exempt it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .study import Report

NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")


class OnePagerError(RuntimeError):
    pass


def allowed_numbers(*objs) -> set[str]:
    out: set[str] = set()

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
        elif isinstance(x, bool) or x is None:
            return
        elif isinstance(x, int):
            out.add(str(x))
        elif isinstance(x, float):
            out.add(f"{x:.2f}")

    for o in objs:
        walk(o)
    return out


def verify_prose(text: str, allowed: set[str]) -> list[str]:
    body = re.sub(r"`[^`]*`", "", text)
    return sorted({t for t in NUMBER.findall(body) if t not in allowed})


def _load(p):
    return json.loads(Path(p).read_text()) if p and Path(p).exists() else None


def build_onepager(report_dir, behavior_json=None, taxonomy_json=None, interpretation_path=None, final=False,
                   models=None, baseline="none", max_words=700) -> str:
    report_dir = Path(report_dir)
    res = _load(report_dir / "results.json")
    if res is None:
        raise OnePagerError(f"{report_dir}/results.json not found: run `analyze` first")
    prov = _load(report_dir / "provenance.json") or {}
    beh, tax = _load(behavior_json), _load(taxonomy_json)
    models = list(models) if models else list(res["models"])
    if len(models) > 3:
        raise OnePagerError("one page fits at most three models; pass --models to choose")
    interp = Path(interpretation_path).read_text().strip() if interpretation_path and Path(interpretation_path).exists() else None
    if final and not interp:
        raise OnePagerError("--final needs an interpretation file written by the author")
    if interp:
        stray = verify_prose(interp, allowed_numbers(res, beh, tax))
        if stray:
            raise OnePagerError(f"interpretation contains numbers that are not in the results files: {stray}")

    d = res["design"]
    top = Report()
    top.add("# AI Research Failure Lab: one-page summary", "",
            f"**Status: {res['status']}**" + (" (" + "; ".join(f"`{x}`" for x in res["reasons"]) + ")" if res["reasons"] else ""), "",
            "**Question.** Do AI research agents notice the methodological flaw that makes a quantitative result meaningless, "
            "and do they avoid false alarms on clean data? Flaws (look-ahead leakage, survivorship, multiple testing) are planted in "
            "synthetic datasets with known ground truth and graded deterministically.", "",
            f"**Method.** {top.i(d['pairs'])} clean/trapped pairs, {top.i(len(d['models']))} model(s), "
            f"{top.i(len(d['conditions']))} prompt condition(s), {top.i(len(d['reps']))} repetition(s); {top.i(d['rows'])} graded runs. "
            f"Intervals are task-clustered bootstrap intervals (confirmatory level {top.n(100 * d['confirmatory_level'])} percent).", "",
            "**Pre-registered results.**", "", "| Model | H1 baseline recall | H2 change in recall | H3 | H4 |", "|---|---|---|---|---|")
    for m in models:
        h = res["models"][m]["hypotheses"]
        g = lambda k, e: top.est(h[k]["estimates"][e]) if e in h[k]["estimates"] else "n/a"
        top.add(f"| `{m}` | {h['H1']['verdict']}: {g('H1', 'recall_baseline')} | {h['H2']['verdict']}: {g('H2', 'delta_recall')} | "
                f"{h['H3']['verdict']} | {h['H4']['verdict']} |")
    top.add("")
    top.add("**Where agents fail** (trapped tasks, baseline condition): did the agent's code look at the property that exposes the flaw, "
            "and did looking lead to a flag? The looking measure is a keyword heuristic with its own error.", "")
    if beh:
        top.add("| Model | Looked at the cue | Flagged given looked | Flagged given not looked | Misses that never looked |", "|---|---|---|---|---|")
        for m in models:
            o = (beh.get(m, {}).get(baseline) or {}).get("overall")
            if o:
                top.add(f"| `{m}` | {top.est(o['looked_rate'])} | {top.est(o['flag_given_looked'])} | "
                        f"{top.est(o['flag_given_not_looked'])} | {top.est(o['miss_never_looked_share'])} |")
    else:
        top.add("Behaviour table not generated yet: run the `behavior` command.")
    top.add("", "**Failure modes** (human-coded, weighted to the full run population):", "")
    rows = [t for t in (tax or []) if t["model"] in models and t["condition"] == baseline]
    if rows:
        for m in models:
            best = sorted((t for t in rows if t["model"] == m), key=lambda t: -t["weighted_prevalence"])[:3]
            for t in best:
                top.add(f"- `{m}`: `{t['code']}` {top.n(t['weighted_prevalence'])} [{top.n(t['lo'])}, {top.n(t['hi'])}] of failures")
    else:
        top.add("Not yet coded: follow the coding workflow in `docs/CODING_GUIDE.md`.")
    top.add("", "**Interpretation.**", "")
    top.verify()

    bottom = Report()
    bottom.add("", "**Limitations.**", "",
               "- Synthetic worlds with planted flaws and stated identifiability conventions; real traps are messier.",
               "- The prompt discloses the scoring rule and each task asks for flags, so the baseline measures detection when asked to look.",
               "- Few clusters means wide intervals; inconclusive is a statement about precision, not about absence of an effect.", "",
               "**Reproduce.** `python -m failurelab.cli repro-check` regenerates the tasks, regrades every raw submission and "
               "re-runs the analysis, comparing each to the stored files.")
    if prov.get("prereg_lock", {}).get("detail"):
        bottom.add("", f"Pre-registration: `{prov['prereg_lock']['detail']}`.")
    bottom.verify()

    middle = interp if interp else "`DRAFT: interpretation not yet written; add it after reading the failure transcripts`"
    text = top.text() + middle + "\n" + bottom.text()
    words = len(text.split())
    if words > max_words:
        raise OnePagerError(f"one-page summary has {words} words, over the limit of {max_words}; choose fewer models or shorten the interpretation")
    return text
