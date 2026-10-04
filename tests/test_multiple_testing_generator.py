import json

import numpy as np

from failurelab import analysis
from failurelab.generators.base import write_task
from failurelab.generators.multiple_testing import (DIFFICULTIES, EFFECT_T_MIN, M_BY_DIFF, MECHANISMS,
                                                   PRESPEC_NULL_T_MAX, MARGIN, MultipleTestingGenerator,
                                                   cell_for_seed)
from failurelab.stats import corr_t
from failurelab.taxonomy import ALL_FLAG_CODES, Trap, Variant

GEN = MultipleTestingGenerator()
VIS, HID = MECHANISMS
CELLS = [(m, d) for m in MECHANISMS for d in DIFFICULTIES]


def _seed_for(mech, diff, k=0):
    return next(s for s in range(6) if cell_for_seed(s) == (mech, diff)) + 6 * k


def test_deterministic_and_balanced_cells():
    assert GEN.generate(2, Variant.TRAPPED).data.equals(GEN.generate(2, Variant.TRAPPED).data)
    assert sorted(cell_for_seed(s) for s in range(6)) == sorted(CELLS)


def test_clean_has_one_candidate_and_trapped_has_m():
    for mech, diff in CELLS:
        s, m = _seed_for(mech, diff), M_BY_DIFF[diff]
        c, t = GEN.generate_cell(s, Variant.CLEAN, mech, diff), GEN.generate_cell(s, Variant.TRAPPED, mech, diff)
        assert c.traps_present == [] and t.traps_present == [Trap.MULTIPLE_TESTING.value]
        if mech == VIS:
            assert len(analysis._mt_cols(c.data)) == 1 and len(analysis._mt_cols(t.data)) == m
        else:
            assert c.data["library_size"].iloc[0] == 1 and t.data["library_size"].iloc[0] == m
            assert [col for col in t.data.columns if col == "signal"] == ["signal"]    # only the winner ships
        assert c.data["fwd_ret"].equals(t.data["fwd_ret"])
        assert (c.effect_exists, c.true_direction) == (t.effect_exists, t.true_direction)


def test_ground_truth_is_calibrated():
    for seed in range(36):
        mech, diff = cell_for_seed(seed)
        p = GEN.generate(seed, Variant.CLEAN).params
        assert p["max_null_t"] < p["bonferroni_crit"] - MARGIN + 1e-9    # a corrected test says 'no effect' for noise
        if p["effect_world"]:
            assert abs(p["prespecified_t"]) > EFFECT_T_MIN and np.sign(p["prespecified_t"]) == p["sign"]
        else:
            assert abs(p["prespecified_t"]) < PRESPEC_NULL_T_MAX


def test_true_signal_position_is_not_fixed():
    positions = set()
    for k in range(40):
        s = _seed_for(VIS, "medium", k)
        g = GEN.generate_cell(s, Variant.TRAPPED, VIS, "medium")
        if g.params["effect_world"]:
            cols = analysis._mt_cols(g.data)
            ts = corr_t(g.data[cols].to_numpy(), g.data["fwd_ret"].to_numpy())
            positions.add(int(np.argmax(np.abs(ts))))
    assert len(positions) >= 4, positions


def test_hidden_winner_is_the_best_looking_candidate():
    for k in range(10):
        g_t = GEN.generate_cell(_seed_for(HID, "hard", k), Variant.TRAPPED, HID, "hard")
        g_c = GEN.generate_cell(_seed_for(HID, "hard", k), Variant.CLEAN, HID, "hard")
        t_win = abs(float(corr_t(g_t.data["signal"].to_numpy(), g_t.data["fwd_ret"].to_numpy())[0]))
        assert t_win >= abs(g_c.params["prespecified_t"]) - 1e-3   # param is rounded to 3 dp


def test_naive_failure_grows_with_search_size():
    """Among NULL worlds, a naive 'best signal, p<.05' analysis is fooled more often as M grows."""
    fooled = {}
    for diff in DIFFICULTIES:
        flags = []
        for k in range(40):
            mech = MECHANISMS[k % 2]
            g = GEN.generate_cell(_seed_for(mech, diff, k // 2), Variant.TRAPPED, mech, diff)
            if not g.effect_exists:
                df = g.data
                res = analysis.naive(df, analysis.kind(df))
                flags.append(res.exists)
        fooled[diff] = float(np.mean(flags))
    assert fooled["easy"] >= fooled["medium"] - 0.1 and fooled["medium"] > fooled["hard"], fooled
    assert fooled["easy"] - fooled["hard"] >= 0.25, fooled


def test_public_files_do_not_reveal_variant(tmp_path):
    for mech, diff in CELLS:
        pubs = {}
        for variant, tid in ((Variant.TRAPPED, "tabc"), (Variant.CLEAN, "tdef")):
            g = GEN.generate_cell(_seed_for(mech, diff), variant, mech, diff)
            write_task(tid, "x", Trap.MULTIPLE_TESTING, variant, g, tmp_path / "pub", tmp_path / "priv")
            pubs[variant] = json.loads((tmp_path / "pub" / tid / "task.json").read_text())
        t, c = pubs[Variant.TRAPPED], pubs[Variant.CLEAN]
        assert t["allowed_flags"] == c["allowed_flags"] == list(ALL_FLAG_CODES)
        assert t["question"] == c["question"] and t["data_dictionary"] == c["data_dictionary"]
        text = json.dumps({k: v for k, v in t.items() if k != "allowed_flags"}).lower()
        for word in ("trapped", "clean", "multiple", "bonferroni", "correction", "p-hack"):
            assert word not in text, (mech, diff, word)
