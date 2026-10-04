import json

import numpy as np
import pandas as pd

from failurelab import analysis
from failurelab.generators.base import write_task
from failurelab.generators.survivorship import (BACKFILL_DAYS, DIFFICULTIES, EFFECT_T_MIN, MECHANISMS,
                                                NULL_T_MAX, SurvivorshipGenerator, cell_for_seed)
from failurelab.stats import mean_t
from failurelab.taxonomy import ALL_FLAG_CODES, Trap, Variant

GEN = SurvivorshipGenerator()
A, B = MECHANISMS
CELLS = [(m, d) for m in MECHANISMS for d in DIFFICULTIES]


def _seed_for(mech, diff, k=0):
    """k-th seed that lands in a given cell (cells cycle with period 6)."""
    return next(s for s in range(6) if cell_for_seed(s) == (mech, diff)) + 6 * k


def test_deterministic_and_balanced_cells():
    assert GEN.generate(3, Variant.TRAPPED).data.equals(GEN.generate(3, Variant.TRAPPED).data)
    assert sorted(cell_for_seed(s) for s in range(6)) == sorted(CELLS)


def test_clean_twin_is_point_in_time_in_every_cell():
    for mech, diff in CELLS:
        g = GEN.generate_cell(_seed_for(mech, diff), Variant.CLEAN, mech, diff)
        df = g.data
        assert g.traps_present == []
        if mech == A:
            assert g.params["attrition"] >= 0.12
            assert df.groupby("asset").size().nunique() > 1          # names drop out over time
            if diff == "easy":
                assert (df["status"] == "delisted").sum() == round(g.params["attrition"] * 100)
            if diff in ("easy", "medium"):
                assert not analysis._survivor_cues(df)["id_gaps"]
        else:
            assert not (pd.to_datetime(df["date"]) < pd.to_datetime(df["index_add_date"])).any()


def test_trapped_twin_carries_a_detectable_cue_in_every_cell():
    for mech, diff in CELLS:
        g = GEN.generate_cell(_seed_for(mech, diff, 1), Variant.TRAPPED, mech, diff)
        df = g.data
        assert g.traps_present == [Trap.SURVIVORSHIP.value]
        if mech == A:
            assert analysis._balanced(df)                              # zero attrition, every cell
            cues = analysis._survivor_cues(df)
            if diff == "easy":
                assert cues["status_has_no_delistings"] and cues["id_gaps"]
            elif diff == "medium":
                assert cues["id_gaps"]
            else:
                assert cues == {}                                      # only the balanced panel remains
        else:
            early = pd.to_datetime(df["date"]) < pd.to_datetime(df["index_add_date"])
            assert early.any()
            gap = (pd.to_datetime(df["index_add_date"]) - pd.to_datetime(df["date"])).dt.days[early]
            assert gap.max() <= BACKFILL_DAYS[diff] * 7 // 5 + 4      # business days -> calendar days


def test_twins_share_the_same_world():
    for diff in DIFFICULTIES:
        s = _seed_for(A, diff)
        c, t = GEN.generate_cell(s, Variant.CLEAN, A, diff).data, GEN.generate_cell(s, Variant.TRAPPED, A, diff).data
        merged = t.merge(c, on=["date", "asset"], suffixes=("_t", "_c"))
        assert len(merged) == len(t) and np.array_equal(merged["ret_t"], merged["ret_c"])   # survivors' rows identical
        assert set(t["asset"]) < set(c["asset"])                                              # strictly fewer names
        s = _seed_for(B, diff)
        c, t = GEN.generate_cell(s, Variant.CLEAN, B, diff).data, GEN.generate_cell(s, Variant.TRAPPED, B, diff).data
        point_in_time = t[pd.to_datetime(t["date"]) >= pd.to_datetime(t["index_add_date"])].reset_index(drop=True)
        assert point_in_time.equals(c.reset_index(drop=True))     # honest filtering recovers the clean twin


def test_survivor_only_trap_biases_naive_analysis_upward():
    ts = []
    for k in range(8):
        g = GEN.generate_cell(_seed_for(A, "medium", k), Variant.CLEAN, A, "medium")
        if g.params["mu"] == 0.0:
            ts.append(g.params["survivor_t"])
    assert ts and min(ts) > 2.5, ts


def test_backfill_bias_shrinks_with_shorter_backfill():
    mean_naive = {}
    for diff in DIFFICULTIES:
        vals = []
        for k in range(10):
            g = GEN.generate_cell(_seed_for(B, diff, k), Variant.TRAPPED, B, diff)
            if g.params["mu"] == 0.0:
                vals.append(mean_t(g.data["ret"].to_numpy()))
        mean_naive[diff] = float(np.mean(vals))
    assert mean_naive["easy"] > mean_naive["medium"] > mean_naive["hard"] > 0.5, mean_naive


def test_ground_truth_is_calibrated():
    for seed in range(24):
        mech, _ = cell_for_seed(seed)
        p = GEN.generate(seed, Variant.CLEAN).params
        t, mu = p["honest_t"], p["mu"]
        assert (abs(t) < NULL_T_MAX) if mu == 0 else (np.sign(t) == np.sign(mu) and abs(t) > EFFECT_T_MIN)


def test_truth_conventions():
    for seed in range(24):
        mech, diff = cell_for_seed(seed)
        c, t = GEN.generate(seed, Variant.CLEAN), GEN.generate(seed, Variant.TRAPPED)
        mu = c.params["mu"]
        if mech == A:      # survivor-only data can never SUPPORT a positive effect (bias is upward)
            assert (not t.effect_exists) and t.true_direction == "none"
            assert c.effect_exists == (mu > 0)
            assert mu >= 0                                          # negative worlds are not identifiable here
        else:              # backfill is fully correctable: the true mu is recoverable, any sign
            assert c.effect_exists == t.effect_exists == (mu != 0)
            assert c.true_direction == t.true_direction


def test_public_files_do_not_reveal_variant(tmp_path):
    for mech, diff in CELLS:
        pubs = {}
        for variant, tid in ((Variant.TRAPPED, "tabc"), (Variant.CLEAN, "tdef")):
            g = GEN.generate_cell(_seed_for(mech, diff), variant, mech, diff)
            write_task(tid, "x", Trap.SURVIVORSHIP, variant, g, tmp_path / "pub", tmp_path / "priv")
            pubs[variant] = json.loads((tmp_path / "pub" / tid / "task.json").read_text())
        t, c = pubs[Variant.TRAPPED], pubs[Variant.CLEAN]
        assert t["allowed_flags"] == c["allowed_flags"] == list(ALL_FLAG_CODES)
        assert t["question"] == c["question"] and t["data_dictionary"] == c["data_dictionary"]
        text = json.dumps({k: v for k, v in t.items() if k != "allowed_flags"}).lower()
        for word in ("trapped", "clean", "survivorship", "bias", "backfill", "survivor"):
            assert word not in text, (mech, diff, word)
