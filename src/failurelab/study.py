"""Phase 4 study analysis: load runs -> quality gates -> pre-registered verdicts -> report.

Everything in report.md is computed here from scores.jsonl; every number is registered when written and the
finished text is scanned for unregistered numbers (`Report.verify`), so the report cannot contain a number
that was typed by hand. Re-running on the same inputs reproduces results.json byte for byte.

Confirmatory status requires ALL of: pre-registration frozen and unmodified, design complete (every model x
condition x rep x task present), every tripwire-flagged run reviewed, non-submitted share within the limit.
Otherwise the verdicts are still computed but labelled NON-CONFIRMATORY with the reasons.
"""
from __future__ import annotations

import hashlib
import json
import platform
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import prereg as prereg_mod
from .experiment import select_tasks
from .inference import Est, Study
from .runner import safe_name
from .schema import GroundTruth

METRIC_COLUMNS = (("clean_accuracy", "accuracy on clean twins"), ("trapped_accuracy", "accuracy on trapped tasks"),
                  ("fooled_rate", "fooled rate"), ("recall", "flag recall"), ("false_flag_rate", "false-flag rate"),
                  ("rfwr_share", "right-for-wrong-reason share"), ("brier_trapped", "Brier (trapped)"),
                  ("brier_clean", "Brier (clean)"))


class StudyError(RuntimeError):
    pass


# ---------------------------------------------------------------------- loading and gates
def load_runs(run_dirs) -> pd.DataFrame:
    frames = []
    for d in run_dirs:
        f = Path(d) / "scores.jsonl"
        rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        frames.append(pd.DataFrame(rows).assign(_source=str(f)))
    df = pd.concat(frames, ignore_index=True)
    df["model"] = df["agent"].str.rsplit("|", n=1).str[0]
    return df


def attach_pairs(df: pd.DataFrame, private_dir: Path) -> pd.DataFrame:
    twin = {}
    for f in Path(private_dir).glob("*.json"):
        t = GroundTruth.load(f)
        twin[t.task_id] = t.twin_id
    df = df.copy()
    df["pair_id"] = df["task_id"].map(lambda t: min(t, twin[t]))
    return df


def _key(r) -> tuple:
    return (r["model"], r["condition"], int(r["rep"]), r["task_id"])


def apply_exclusions(df: pd.DataFrame, path: str | Path | None) -> tuple[pd.DataFrame, set, int]:
    """exclusions file: [{"model","condition","rep","task_id","action":"exclude"|"keep","reason":"..."}]
    Every tripwire-flagged run needs an entry (keep or exclude) so a human has looked at it."""
    if not path or not Path(path).exists():
        return df, set(), 0
    entries = json.loads(Path(path).read_text())
    reviewed = {(e["model"], e["condition"], int(e["rep"]), e["task_id"]) for e in entries}
    drop = {(e["model"], e["condition"], int(e["rep"]), e["task_id"]) for e in entries if e["action"] == "exclude"}
    mask = df.apply(lambda r: _key(r) in drop, axis=1)
    return df[~mask].reset_index(drop=True), reviewed, int(mask.sum())


def check_design(df: pd.DataFrame, plan: dict, private_dir: Path) -> list[str]:
    d = plan["design"]
    ids = select_tasks(Path(private_dir), d.get("max_tasks"), d.get("task_selection_seed", 0))
    expected = set(ids) if ids is not None else {f.stem for f in Path(private_dir).glob("*.json")}
    problems = []
    extra_models = sorted(set(df["model"]) - set(d["models"]))
    if extra_models:
        problems.append(f"unexpected models in runs: {extra_models}")
    for model in d["models"]:
        for cond in d["conditions"]:
            for rep in range(d["reps"]):
                got = set(df[(df.model == model) & (df.condition == cond) & (df.rep == rep)].task_id)
                if got != expected:
                    problems.append(f"{model}|{cond}|rep{rep}: {len(expected - got)} task(s) missing, {len(got - expected)} unexpected")
    return problems


# ---------------------------------------------------------------------- hypotheses
def evaluate_hypotheses(df: pd.DataFrame, rules: dict, study: Study, level: float,
                        which: tuple = ("H1", "H2", "H3", "H4")) -> dict:
    """Apply the pre-registered decision rules to ONE model's runs. Verdict in
    {supported, falsified, inconclusive, not_evaluable}."""
    base_c, treat_c = rules["baseline_condition"], rules["treatment_condition"]
    sub = {c: df[df.condition == c] for c in df.condition.unique()}
    vec = lambda cond, metric: study.vectors(sub[cond], metric)
    out: dict = {}

    if "H1" in which:
        if base_c not in sub:
            out["H1"] = {"verdict": "not_evaluable", "estimates": {}}
        else:
            h = rules["H1"]
            rec, acc = study.estimate(vec(base_c, "recall"), level), study.estimate(vec(base_c, "clean_accuracy"), level)
            if not (rec.defined and acc.defined):
                v = "not_evaluable"
            elif rec.hi < h["recall_below"] and acc.lo > h["clean_accuracy_above"]:
                v = "supported"
            elif rec.point >= h["recall_below"] and rec.lo > h["falsify_if_recall_lower_above"]:
                v = "falsified"
            else:
                v = "inconclusive"
            out["H1"] = {"verdict": v, "estimates": {"recall_baseline": rec.as_dict(), "clean_accuracy_baseline": acc.as_dict()}}

    if "H2" in which:
        if base_c not in sub or treat_c not in sub:
            out["H2"] = {"verdict": "not_evaluable", "estimates": {}}
        else:
            h = rules["H2"]
            dr = study.contrast(vec(treat_c, "recall"), vec(base_c, "recall"), level)
            df_ = study.contrast(vec(treat_c, "false_flag_rate"), vec(base_c, "false_flag_rate"), level)
            if not (dr.defined and df_.defined):
                v = "not_evaluable"
            elif dr.point >= h["delta_recall_min"] and dr.lo > 0 and df_.hi <= h["delta_false_flag_max"]:
                v = "supported"
            elif dr.hi < h["falsify_if_delta_recall_below"] or df_.lo > h["delta_false_flag_max"]:
                v = "falsified"
            else:
                v = "inconclusive"
            out["H2"] = {"verdict": v, "estimates": {"delta_recall": dr.as_dict(), "delta_false_flag_rate": df_.as_dict()}}

    if "H3" in which:
        if base_c not in sub:
            out["H3"] = {"verdict": "not_evaluable", "estimates": {}}
        else:
            h = rules["H3"]
            sh = study.estimate(vec(base_c, "rfwr_share"), level)
            if not sh.defined:
                v = "not_evaluable"
            elif sh.point >= h["share_min"] and sh.lo > h["support_lower_above"]:
                v = "supported"
            elif sh.hi < h["falsify_if_share_below"]:
                v = "falsified"
            else:
                v = "inconclusive"
            out["H3"] = {"verdict": v, "estimates": {"right_for_wrong_reason_share": sh.as_dict()}}

    if "H4" in which:
        if base_c not in sub:
            out["H4"] = {"verdict": "not_evaluable", "estimates": {}}
        else:
            d = study.contrast(vec(base_c, "brier_trapped"), vec(base_c, "brier_clean"), level)
            v = "not_evaluable" if not d.defined else "supported" if d.lo > 0 else "falsified" if d.hi <= 0 else "inconclusive"
            out["H4"] = {"verdict": v, "estimates": {"delta_brier_trapped_minus_clean": d.as_dict()}}
    return out


# ---------------------------------------------------------------------- report with number provenance
class Report:
    def __init__(self):
        self.lines: list[str] = []
        self.reg: set[str] = set()

    def n(self, x, fmt: str = ".2f") -> str:
        s = "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else format(x, fmt)
        self.reg.add(s)
        return s

    def i(self, x) -> str:
        return self.n(int(x), "d")

    def est(self, e: dict | Est) -> str:
        e = e.as_dict() if isinstance(e, Est) else e
        return f"{self.n(e['point'])} [{self.n(e['lo'])}, {self.n(e['hi'])}]"

    def add(self, *lines: str) -> None:
        self.lines.extend(lines)

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"

    def verify(self) -> None:
        """Every number in the body must have been produced by n()/i()/est(). Text inside backticks is treated as a
        literal identifier or provenance string (model names, timestamps, commit hashes) and is not scanned."""
        body = re.sub(r"`[^`]*`", "", self.text())
        stray = sorted({t for t in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", body) if t not in self.reg})
        if stray:
            raise AssertionError(f"report contains numbers that were not computed: {stray}")


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------------- main entry
def run_study(tasks_dir, run_dirs, out_dir, prereg_path="docs/prereg.json", doc_path="docs/PREREGISTRATION.md",
              lock_path="experiments/prereg.lock", exclusions_path=None, allow_incomplete=False,
              allow_unreviewed=False) -> dict:
    tasks_dir, out_dir = Path(tasks_dir), Path(out_dir)
    plan = prereg_mod.load(prereg_path)
    plan_errors = prereg_mod.validate(plan)
    lock_state, lock_detail = prereg_mod.check_lock(lock_path, prereg_path, doc_path)

    df = attach_pairs(load_runs(run_dirs), tasks_dir / "private")
    df_all = df                       # completeness is judged BEFORE reviewed exclusions are removed
    df, reviewed, n_excluded = apply_exclusions(df, exclusions_path)
    reasons: list[str] = []

    if plan_errors:
        reasons.append("pre-registration is incomplete (" + "; ".join(plan_errors) + ")")
        design_problems: list[str] = []
        models = sorted(df.model.unique())
    else:
        design_problems = check_design(df_all, plan, tasks_dir / "private")
        models = list(plan["design"]["models"])
        if lock_state != "ok":
            reasons.append(f"pre-registration lock: {lock_detail}")
    if design_problems:
        if not allow_incomplete:
            raise StudyError("design incomplete (use --allow-incomplete to analyse anyway):\n- " + "\n- ".join(design_problems[:10]))
        reasons.append(f"design incomplete ({len(design_problems)} problem(s))")

    flagged = df_all[df_all["tripwire_hits"].map(lambda h: len(h) > 0)]
    unreviewed = [r for _, r in flagged.iterrows() if _key(r) not in reviewed]
    if unreviewed:
        if not allow_unreviewed:
            raise StudyError(f"{len(unreviewed)} tripwire-flagged run(s) have no review entry in the exclusions file "
                             "(run `review`, then record keep/exclude; or pass --allow-unreviewed)")
        reasons.append(f"{len(unreviewed)} tripwire-flagged run(s) not reviewed")

    max_ns = plan.get("analysis", {}).get("max_nonsubmitted_share", 0.10)
    ns_share = {m: float((df[df.model == m].status != "submitted").mean()) for m in models if (df.model == m).any()}
    for m, s in ns_share.items():
        if s > max_ns:
            reasons.append(f"non-submitted share for {m} exceeds the limit")
    confirmatory = not reasons

    a = plan.get("analysis", {})
    rules = plan.get("rules", prereg_mod.DEFAULT_RULES)
    models = [m for m in models if (df.model == m).any()]
    n_models = max(1, len(models))
    level_conf = 1 - a.get("confirmatory_alpha", 0.05) / (a.get("n_confirmatory_hypotheses", 4) * n_models)
    level_exp = a.get("exploratory_level", 0.95)
    study = Study(df.pair_id, B=a.get("bootstrap_B", 10000), seed=a.get("bootstrap_seed", 12345))
    conditions = [c for c in ("none", "generic_hint", "checklist") if c in set(df.condition)]

    results: dict = {"status": "CONFIRMATORY" if confirmatory else "NON-CONFIRMATORY", "reasons": reasons,
                     "design": {"models": models, "conditions": conditions, "pairs": study.P,
                                "reps": sorted(int(r) for r in df.rep.unique()), "rows": int(len(df)),
                                "bootstrap_B": study.B, "confirmatory_level": level_conf,
                                "exploratory_level": level_exp, "excluded_runs": n_excluded},
                     "models": {}}
    for m in models:
        dm = df[df.model == m]
        entry: dict = {"hypotheses": evaluate_hypotheses(dm, rules, study, level_conf), "conditions": {}, "quality": {}}
        for c in conditions:
            dc = dm[dm.condition == c]
            if dc.empty:
                continue
            metrics = {k: study.estimate(study.vectors(dc, k), level_exp).as_dict() for k, _ in METRIC_COLUMNS}
            sub_ok = dc[dc.status == "submitted"]
            by_family = {}
            for fam in sorted(dc.family.unique()):
                df_ = dc[dc.family == fam]
                by_family[fam] = {k: study.estimate(study.vectors(df_, k), level_exp).as_dict() for k in ("recall", "false_flag_rate")}
            entry["conditions"][c] = {
                "metrics": metrics, "by_family": by_family,
                "submitted_only": {k: study.estimate(study.vectors(sub_ok, k), level_exp).as_dict() for k in ("recall", "false_flag_rate")} if len(sub_ok) else {}}
            entry["quality"][c] = {"runs": int(len(dc)), "submitted_share": float((dc.status == "submitted").mean()),
                                   "mean_python_calls": float(dc.n_python_calls.mean()),
                                   "preregistered_first_share": float(dc.preregistered_first.mean()),
                                   "tripwire_flagged": int(dc["tripwire_hits"].map(lambda h: len(h) > 0).sum()),
                                   "mean_input_tokens": float(dc.input_tokens.mean()), "mean_output_tokens": float(dc.output_tokens.mean())}
        results["models"][m] = entry

    report = _build_report(results, rules, plan, lock_detail)
    report.verify()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(_clean(results), indent=2, sort_keys=True))
    (out_dir / "report.md").write_text(report.text())
    prov = {"scores_files_sha256": {str(Path(d) / "scores.jsonl"): _sha_file(Path(d) / "scores.jsonl") for d in run_dirs},
            "prereg_lock": {"state": lock_state, "detail": lock_detail},
            "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
            "bootstrap": {"B": study.B, "seed": a.get("bootstrap_seed", 12345)}}
    (out_dir / "provenance.json").write_text(json.dumps(_clean(prov), indent=2, sort_keys=True))
    return results


def _build_report(res: dict, rules: dict, plan: dict, lock_detail: str) -> Report:
    r = Report()
    d = res["design"]
    r.add("# AI Research Failure Lab: results", "",
          f"**Status: {res['status']}**" + (" (" + "; ".join(f"`{x}`" for x in res["reasons"]) + ")" if res["reasons"] else ""), "",
          f"Pre-registration: `{lock_detail}`.", "",
          "## Design", "",
          f"- Models: {', '.join('`' + m + '`' for m in d['models'])}",
          f"- Conditions: {', '.join('`' + c + '`' for c in d['conditions'])}",
          f"- Clean/trapped pairs (bootstrap clusters): {r.i(d['pairs'])}; runs analysed: {r.i(d['rows'])}; "
          f"runs excluded after review: {r.i(d['excluded_runs'])}",
          f"- Intervals: cluster bootstrap over pairs, {r.i(d['bootstrap_B'])} draws; confirmatory level {r.n(100 * d['confirmatory_level'])} "
          f"percent (Bonferroni over the confirmatory hypotheses and models), exploratory level {r.n(100 * d['exploratory_level'])} percent.",
          "- Invalid or non-submitted runs count as failures. Repeated runs of a task are kept inside their pair cluster.", "")
    h1, h2, h3 = rules["H1"], rules["H2"], rules["H3"]
    claims = {
        "H1": f"unprompted flag recall on trapped tasks is below {r.n(h1['recall_below'])} while clean accuracy is above {r.n(h1['clean_accuracy_above'])}",
        "H2": f"the checklist raises recall by at least {r.n(h2['delta_recall_min'])} without raising the false-flag rate by more than {r.n(h2['delta_false_flag_max'])}",
        "H3": f"at least {r.n(h3['share_min'])} of correct trapped answers under the baseline condition miss the planted flaw",
        "H4": "Brier score is worse on trapped tasks than on clean twins (baseline condition)"}
    r.add("## Confirmatory hypotheses", "")
    for m, e in res["models"].items():
        r.add(f"### `{m}`", "", "| Hypothesis | Verdict | Evidence |", "|---|---|---|")
        for h, v in e["hypotheses"].items():
            ev = "; ".join(f"{k.replace('_', ' ')} {r.est(x)}" for k, x in v["estimates"].items()) or "n/a"
            r.add(f"| {h}: {claims[h]} | {v['verdict']} | {ev} |")
        r.add("")
    r.add("## Exploratory results", "", f"Intervals are {r.n(100 * d['exploratory_level'])} percent cluster-bootstrap intervals. "
          "Nothing in this section was pre-registered as a test.", "")
    for m, e in res["models"].items():
        r.add(f"### `{m}`: metrics by condition", "",
              "| Condition | " + " | ".join(lbl for _, lbl in METRIC_COLUMNS) + " |", "|---|" + "---|" * len(METRIC_COLUMNS))
        for c, ce in e["conditions"].items():
            r.add(f"| `{c}` | " + " | ".join(r.est(ce["metrics"][k]) for k, _ in METRIC_COLUMNS) + " |")
        r.add("", f"### `{m}`: recall and false-flag rate by family", "", "| Condition | Family | Recall | False-flag rate |", "|---|---|---|---|")
        for c, ce in e["conditions"].items():
            for fam, fe in ce["by_family"].items():
                r.add(f"| `{c}` | `{fam}` | {r.est(fe['recall'])} | {r.est(fe['false_flag_rate'])} |")
        r.add("")
    r.add("## Data quality", "", "| Model | Condition | Runs | Submitted share | Mean Python calls | Preregistered first | Tripwire-flagged | Mean input tokens |",
          "|---|---|---|---|---|---|---|---|")
    for m, e in res["models"].items():
        for c, q in e["quality"].items():
            r.add(f"| `{m}` | `{c}` | {r.i(q['runs'])} | {r.n(q['submitted_share'])} | {r.n(q['mean_python_calls'], '.1f')} | "
                  f"{r.n(q['preregistered_first_share'])} | {r.i(q['tripwire_flagged'])} | {r.n(q['mean_input_tokens'], '.0f')} |")
    r.add("", "### Sensitivity: submitted runs only", "", "| Model | Condition | Recall | False-flag rate |", "|---|---|---|---|")
    for m, e in res["models"].items():
        for c, ce in e["conditions"].items():
            so = ce["submitted_only"]
            if so:
                r.add(f"| `{m}` | `{c}` | {r.est(so['recall'])} | {r.est(so['false_flag_rate'])} |")
    r.add("", "## Limitations", "",
          "- Synthetic worlds with planted flaws; real traps are messier. Ground truth follows the identifiability conventions in the README.",
          "- The prompt discloses the scoring rule and each task asks the agent to flag problems, so the baseline condition measures detection when asked to look.",
          "- Few clusters means wide intervals; an inconclusive verdict is a finding about precision, not evidence of no effect.",
          "- Hypotheses were fixed before data collection only if the pre-registration lock above is intact and was committed before the first live run.", "",
          "## Reproduce", "",
          "`python -m failurelab.cli analyze --tasks tasks --runs <run dirs> --out report` regenerates this file and `results.json` from `scores.jsonl`.")
    return r


# ---------------------------------------------------------------------- tripwire review helper
def tripwire_review(run_dirs, context: int = 0) -> list[dict]:
    """List tripwire-flagged runs with the offending code lines, for the human review step."""
    from .agents.llm_agent import TRIPWIRE
    out = []
    for df_ in (load_runs([d]).assign(_dir=str(d)) for d in run_dirs):
        for _, row in df_[df_["tripwire_hits"].map(lambda h: len(h) > 0)].iterrows():
            tpath = Path(row["_dir"]) / f"rep{int(row['rep'])}" / safe_name(row["agent"]) / "transcripts" / f"{row['task_id']}.json"
            lines = []
            if tpath.exists():
                for cell in json.loads(tpath.read_text())["cells"]:
                    lines += [ln.strip() for ln in cell["code"].splitlines() if TRIPWIRE.search(ln)]
            out.append({"model": row["model"], "condition": row["condition"], "rep": int(row["rep"]),
                        "task_id": row["task_id"], "hits": list(row["tripwire_hits"]), "lines": lines})
    return out
