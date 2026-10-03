import json

import numpy as np
import pandas as pd

from failurelab.generators.base import write_task
from failurelab.generators.leakage import (DIFFICULTIES, MECHANISMS, LeakageGenerator, cell_for_seed)
from failurelab.taxonomy import ALL_FLAG_CODES, Trap, Variant

GEN = LeakageGenerator()
CELLS = [(m, d) for m in MECHANISMS for d in DIFFICULTIES]


def _corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1])


def test_deterministic():
    a = GEN.generate(3, Variant.TRAPPED).data
    b = GEN.generate(3, Variant.TRAPPED).data
    assert a.equals(b)


def test_cells_are_balanced_over_nine_consecutive_seeds():
    assert sorted(cell_for_seed(s) for s in range(9)) == sorted(CELLS)


def test_clean_twin_is_an_exact_negative_control_in_every_cell():
    """The clean signal must be EXACTLY the honest trailing mean of ret (and as-of == date)."""
    for mech, diff in CELLS:
        c = GEN.generate_cell(11, Variant.CLEAN, mech, diff)
        df = c.data.sort_values(["asset", "date"])
        # honest trailing mean needs history before the trimmed window, so check the interior
        got = df.groupby("asset")["ret"].transform(lambda s: s.rolling(5).mean())
        ok = got.notna()
        assert np.allclose(df.loc[ok, "signal"], got[ok], atol=1e-12), (mech, diff)
        assert c.traps_present == []
        if "signal_asof" in df.columns:
            assert (df["signal_asof"] == df["date"]).all()


def test_twins_share_world_but_not_signal_in_every_cell():
    for mech, diff in CELLS:
        c = GEN.generate_cell(5, Variant.CLEAN, mech, diff)
        t = GEN.generate_cell(5, Variant.TRAPPED, mech, diff)
        assert c.data["ret"].equals(t.data["ret"])
        assert not c.data["signal"].equals(t.data["signal"]), (mech, diff)
        assert c.params["phi"] == t.params["phi"] and c.effect_exists == t.effect_exists
        assert t.traps_present == [Trap.LEAKAGE.value]
        assert c.data.shape == t.data.shape and not c.data.isna().any().any()
        assert list(c.data.columns) == list(t.data.columns)


def test_leak_strength_ladder_is_monotone_in_effect_size():
    """easy leaks look absurd (corr ~0.4); harder leaks look like plausible weak signals."""
    for mech in MECHANISMS:
        rs = []
        for diff in DIFFICULTIES:
            vals = []
            for seed in range(6):
                d = GEN.generate_cell(seed, Variant.TRAPPED, mech, diff).data
                vals.append(_corr(d["signal"].to_numpy(), d["fwd_ret_1d"].to_numpy()))
            rs.append(float(np.mean(vals)))
        assert rs[0] > rs[1] > rs[2] - 0.02, (mech, rs)
        assert rs[0] > 0.3, (mech, rs)


def test_asof_column_flags_exactly_the_restated_rows():
    t = GEN.generate_cell(2, Variant.TRAPPED, "asof_timestamp", "medium").data
    late = pd.to_datetime(t["signal_asof"]) > pd.to_datetime(t["date"])
    assert 0.2 < late.mean() < 0.4          # ~30% restated
    h = GEN.generate_cell(2, Variant.TRAPPED, "asof_timestamp", "hard").data
    assert 0.03 < (pd.to_datetime(h["signal_asof"]) > pd.to_datetime(h["date"])).mean() < 0.15


def test_public_files_do_not_reveal_variant(tmp_path):
    for mech, diff in CELLS:
        pubs = {}
        for variant, tid in ((Variant.TRAPPED, "tabc"), (Variant.CLEAN, "tdef")):
            g = GEN.generate_cell(1, variant, mech, diff)
            write_task(tid, "x", Trap.LEAKAGE, variant, g, tmp_path / "pub", tmp_path / "priv")
            pubs[variant] = json.loads((tmp_path / "pub" / tid / "task.json").read_text())
        t, c = pubs[Variant.TRAPPED], pubs[Variant.CLEAN]
        assert t["allowed_flags"] == c["allowed_flags"] == list(ALL_FLAG_CODES)
        for k in ("question", "data_dictionary"):
            assert t[k] == c[k]
        text = json.dumps({k: v for k, v in t.items() if k != "allowed_flags"}).lower()
        for word in ("trapped", "clean", "leak", "centered", "look-ahead", "phi", "restated", "shift"):
            assert word not in text, (mech, diff, word)


def test_ground_truth_is_calibrated_so_an_honest_analyst_agrees():
    """Every world: null => |t| < 2; effect => |t| > 4.5 with the planted sign (no ambiguous truths)."""
    from failurelab.generators.leakage import EFFECT_T_MIN, NULL_T_MAX
    for seed in range(60):
        g = GEN.generate(seed, Variant.CLEAN)
        t, phi = g.params["honest_t"], g.params["phi"]
        if phi == 0.0:
            assert abs(t) < NULL_T_MAX
        else:
            assert np.sign(t) == np.sign(phi) and abs(t) > EFFECT_T_MIN
        assert g.params == GEN.generate(seed, Variant.TRAPPED).params | {"strength": g.params["strength"]}
