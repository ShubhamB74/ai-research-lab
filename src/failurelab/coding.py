"""Human coding pipeline (Phase 5, layer B): stratified sampling, coding sheets, agreement, weighted prevalence.

The failure taxonomy is built by READING transcripts. This module supports that, it does not invent codes.
SEED_CODES are a-priori hypotheses from the benchmark design; revise them as you read (add NEW:<name> codes,
then fold stable ones into the codebook and record the revision in experiments/LOG.md).
Sampling is deliberately non-proportional (failures are oversampled), so prevalence is reported with
inverse-probability weights, never as raw shares of the coded sample.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from .behavior import outcome_class, transcript_path
from .inference import Study
from .runner import safe_name
from .schema import GroundTruth

SEED_CODES = {
    "NEVER_LOOKED": "The agent's code never examined the data property that exposes the flaw.",
    "SAW_DISMISSED": "The agent's own output showed the cue (e.g. zero attrition, a loading on future returns) and it explained it away or ignored it.",
    "MISDIAGNOSED": "Flagged a different problem than the one present, or right problem with a wrong explanation.",
    "OVERTRUST_STATS": "Concluded from a significant statistic without questioning how the sample, feature or search that produced it was built.",
    "CORRECTED_BUT_MISJUDGED": "Applied a correction or adjustment but still drew the wrong conclusion from it (or the reverse).",
    "GENERIC_HEDGE": "Mentioned possible problems in general terms but made no specific flag or test.",
    "PREMATURE_STOP": "Submitted after minimal analysis, without trying to break its own result.",
    "UNSUPPORTED_FLAG": "Flagged a problem on a clean twin without evidence in its own analysis.",
    "TOOLING_FAILURE": "Outcome driven by tool or format trouble (errors, timeouts, bad submission), not by research reasoning.",
    "OTHER": "Anything else; describe it in notes.",
}
OUTCOME_ORDER = ("fooled_missed", "right_wrong_reason", "missed_wrong", "false_alarm", "invalid", "caught", "clean_ok")
SHEET_COLUMNS = ["run_id", "pair_id", "model", "condition", "rep", "task_id", "family", "mechanism", "difficulty", "variant",
                 "outcome_class", "final_conclusion", "final_flags", "stratum", "stratum_size", "sample_size", "weight",
                 "transcript_md", "codes", "evidence", "notes", "cue_inspected_human"]


def run_id(model, condition, rep, task_id) -> str:
    return f"{model}|{condition}|rep{int(rep)}|{task_id}"


# ---------------------------------------------------------------------- sampling
def sample_for_coding(scores: pd.DataFrame, per_stratum: int, seed: int = 0) -> pd.DataFrame:
    """Stratify by (model, condition, family, outcome class); take up to `per_stratum` runs from each, spreading
    over distinct pairs first. Adds stratum size, sample size and the inverse-probability weight."""
    df = scores.copy()
    df["outcome_class"] = df.apply(outcome_class, axis=1)
    df["stratum"] = df[["model", "condition", "family", "outcome_class"]].astype(str).agg("/".join, axis=1)
    rng = random.Random(seed)
    picks = []
    for _, g in df.groupby("stratum", sort=True):
        order = list(g.index)
        rng.shuffle(order)
        seen, first, rest = set(), [], []
        for i in order:                                    # distinct pairs first
            (rest if df.at[i, "pair_id"] in seen else first).append(i)
            seen.add(df.at[i, "pair_id"])
        chosen = (first + rest)[:per_stratum]
        picks.append(df.loc[chosen].assign(stratum_size=len(g), sample_size=len(chosen)))
    out = pd.concat(picks)
    out["weight"] = out["stratum_size"] / out["sample_size"]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------- transcript rendering and sheet
def render_transcript_md(tpath: Path, header: list[str], max_out: int = 1200) -> str:
    tr = json.loads(Path(tpath).read_text())
    names: dict[str, str] = {}
    lines = ["# Transcript", "", *header, ""]
    cell = 0
    for msg in tr["messages"][1:]:                          # message 0 is the task prompt
        for b in msg["content"]:
            if b["type"] == "text" and msg["role"] == "assistant" and b["text"].strip():
                lines += ["**Agent:** " + b["text"].strip(), ""]
            elif b["type"] == "tool_use":
                names[b["id"]] = b["name"]
                if b["name"] == "run_python":
                    cell += 1
                    lines += [f"### Cell {cell}", "```python", b["input"].get("code", ""), "```", ""]
                else:
                    lines += [f"### Tool call: {b['name']}", "```json", json.dumps(b["input"], indent=1), "```", ""]
            elif b["type"] == "tool_result":
                txt = b["content"] if isinstance(b["content"], str) else json.dumps(b["content"])
                if len(txt) > max_out:
                    txt = txt[:max_out] + f"\n... [{len(txt) - max_out} characters cut]"
                lines += [f"Output ({names.get(b['tool_use_id'], 'tool')}):", "```", txt, "```", ""]
    return "\n".join(lines)


def write_coding_sheet(sample: pd.DataFrame, private_dir: Path, out_dir: Path) -> Path:
    """Writes coding_sheet.csv and one readable transcript per sampled run. The sheet shows the graded outcome
    and trap family (failure analysis needs them) but NOT the automatic cue detector's output, so the human
    `cue_inspected_human` label stays independent of it."""
    out_dir = Path(out_dir)
    (out_dir / "transcripts_md").mkdir(parents=True, exist_ok=True)
    truth = {}
    for f in Path(private_dir).glob("*.json"):
        t = GroundTruth.load(f)
        truth[t.task_id] = t
    rows = []
    for _, r in sample.iterrows():
        rid, t = run_id(r["model"], r["condition"], r["rep"], r["task_id"]), truth[r["task_id"]]
        md = out_dir / "transcripts_md" / (safe_name(rid) + ".md")
        tp = transcript_path(r)
        sub = json.loads(tp.read_text()).get("submission_raw") if tp.exists() else None
        header = [f"- run: `{rid}`", f"- trap family: {t.family} / mechanism: {t.params.get('mechanism', '')} / "
                  f"difficulty: {t.params.get('difficulty', '')} / variant: {t.variant}",
                  f"- graded outcome: {r['outcome_class']}"]
        if tp.exists():
            md.write_text(render_transcript_md(tp, header))
        else:
            md.write_text("\n".join(["# Transcript", "", *header, "", "(no transcript on disk)"]))
        concl = (f"effect_exists={sub.get('effect_exists')} direction={sub.get('direction')} confidence={sub.get('confidence')}"
                 if sub and "effect_exists" in sub else "no valid submission")
        rows.append({"run_id": rid, "pair_id": r["pair_id"], "model": r["model"], "condition": r["condition"],
                     "rep": int(r["rep"]), "task_id": r["task_id"], "family": t.family,
                     "mechanism": t.params.get("mechanism", ""), "difficulty": t.params.get("difficulty", ""),
                     "variant": t.variant, "outcome_class": r["outcome_class"], "final_conclusion": concl,
                     "final_flags": ";".join(sub.get("flags", [])) if sub and "flags" in sub else "",
                     "stratum": r["stratum"], "stratum_size": int(r["stratum_size"]), "sample_size": int(r["sample_size"]),
                     "weight": float(r["weight"]), "transcript_md": str(md), "codes": "", "evidence": "", "notes": "",
                     "cue_inspected_human": ""})
    path = out_dir / "coding_sheet.csv"
    pd.DataFrame(rows, columns=SHEET_COLUMNS).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------- codes, agreement, validity
def parse_codes(cell) -> list[str]:
    return sorted({c.strip() for c in str(cell if isinstance(cell, str) else "").split(";") if c.strip()})


def validate_sheet(sheet: pd.DataFrame, allowed=None) -> list[str]:
    allowed = set(allowed or SEED_CODES)
    problems = []
    for _, r in sheet.iterrows():
        for c in parse_codes(r["codes"]):
            if c not in allowed and not c.startswith("NEW:"):
                problems.append(f"{r['run_id']}: unknown code {c!r} (use a codebook code or NEW:<name>)")
    return problems


def cohens_kappa(a, b) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return 1.0 if pe == 1.0 and po == 1.0 else (po - pe) / (1 - pe) if pe != 1.0 else float("nan")


def kappa_table(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Per-code agreement between two coders on the runs both coded (present/absent per run)."""
    m = a.merge(b, on="run_id", suffixes=("_a", "_b"))
    ca, cb = m["codes_a"].map(parse_codes), m["codes_b"].map(parse_codes)
    rows = []
    for code in sorted({c for cs in list(ca) + list(cb) for c in cs}):
        x, y = ca.map(lambda cs: code in cs), cb.map(lambda cs: code in cs)
        rows.append({"code": code, "n_runs": len(m), "coder_a": int(x.sum()), "coder_b": int(y.sum()),
                     "percent_agreement": float((x == y).mean()), "kappa": cohens_kappa(x, y)})
    return pd.DataFrame(rows)


def detector_validity(sheet: pd.DataFrame, beh: pd.DataFrame) -> dict:
    """Precision, recall and kappa of the automatic cue detector against the human `cue_inspected_human` labels."""
    m = sheet.merge(beh[["model", "condition", "rep", "task_id", "cue_inspected"]],
                    on=["model", "condition", "rep", "task_id"])
    m = m[m["cue_inspected_human"].astype(str).str.lower().isin(["y", "n"])]
    if m.empty:
        return {"n": 0}
    h, d = m["cue_inspected_human"].str.lower().eq("y").to_numpy(), m["cue_inspected"].to_numpy(bool)
    tp = int((h & d).sum())
    return {"n": int(len(m)), "precision": tp / d.sum() if d.sum() else float("nan"),
            "recall": tp / h.sum() if h.sum() else float("nan"), "kappa": cohens_kappa(h, d)}


# ---------------------------------------------------------------------- weighted prevalence
def prevalence_table(sheet: pd.DataFrame, by=("model", "condition"), failures_only: bool = True,
                     B: int = 5000, seed: int = 12345, level: float = 0.95) -> pd.DataFrame:
    """Inverse-probability-weighted share of runs carrying each code, within each domain (cluster bootstrap by pair)."""
    s = sheet.copy()
    if failures_only:
        s = s[s["outcome_class"].isin(["fooled_missed", "right_wrong_reason", "missed_wrong", "false_alarm", "invalid"])]
    study = Study(s["pair_id"], B=B, seed=seed)
    idx_all = s["pair_id"].map(study.index).to_numpy()
    s["_codes"] = s["codes"].map(parse_codes)
    rows = []
    for key, g in s.groupby(list(by)):
        key = key if isinstance(key, tuple) else (key,)
        idx = g["pair_id"].map(study.index).to_numpy()
        w = g["weight"].to_numpy(float)
        den = np.bincount(idx, weights=w, minlength=study.P)
        for code in sorted({c for cs in g["_codes"] for c in cs}):
            has = g["_codes"].map(lambda cs: code in cs).to_numpy(float)
            num = np.bincount(idx, weights=w * has, minlength=study.P)
            e = study.estimate((num, den), level)
            rows.append({**dict(zip(by, key)), "code": code, "n_coded": int(len(g)), "weighted_prevalence": e.point,
                         "lo": e.lo, "hi": e.hi})
    return pd.DataFrame(rows)
