import numpy as np
import pandas as pd

from failurelab.inference import Study, row_contributions


def _df(rows):
    return pd.DataFrame(rows, columns=["pair_id", "variant", "tp", "fn", "fp", "conclusion_correct", "fooled",
                                       "right_for_wrong_reason", "brier"])


def test_point_estimates_match_hand_computation():
    df = _df([("a", "trapped", 1, 0, 0, True, False, False, 0.1), ("a", "trapped", 0, 1, 0, False, True, False, 0.9),
              ("b", "trapped", 1, 0, 0, True, False, True, 0.2), ("b", "clean", 0, 0, 1, True, False, False, 0.3),
              ("a", "clean", 0, 0, 0, True, False, False, 0.0)])
    st = Study(df.pair_id, B=2000, seed=1)
    pt = lambda m: st.estimate(st.vectors(df, m), 0.95).point
    assert pt("recall") == 2 / 3                    # 2 flags caught out of 3 planted
    assert pt("false_flag_rate") == 1 / 2           # 1 of 2 clean tasks flagged something
    assert pt("clean_accuracy") == 1.0 and pt("trapped_accuracy") == 2 / 3
    assert pt("fooled_rate") == 1 / 3
    assert pt("rfwr_share") == 1 / 2                # 1 of 2 correct trapped answers missed the flaw
    assert abs(pt("brier_trapped") - (0.1 + 0.9 + 0.2) / 3) < 1e-12 and pt("brier_clean") == 0.15


def test_row_contributions_zero_outside_their_variant():
    df = _df([("a", "clean", 0, 0, 1, True, False, False, 0.3), ("a", "trapped", 1, 0, 0, True, False, False, 0.1)])
    c = row_contributions(df)
    assert c["recall"][1][0] == 0 and c["false_flag_rate"][1][1] == 0


def test_cluster_bootstrap_is_wider_than_naive_when_clusters_differ():
    """If pairs differ a lot, treating the runs as independent would be far too narrow."""
    rng = np.random.default_rng(0)
    rows = []
    for p in range(30):
        pr = rng.choice([0.05, 0.95])                           # strongly pair-dependent outcomes
        for _ in range(6):
            tp = int(rng.random() < pr)
            rows.append((f"p{p}", "trapped", tp, 1 - tp, 0, True, False, False, 0.1))
    df = _df(rows)
    st = Study(df.pair_id, B=3000, seed=2)
    e = st.estimate(st.vectors(df, "recall"), 0.95)
    p = e.point
    naive_half = 1.96 * np.sqrt(p * (1 - p) / len(df))
    assert (e.hi - e.lo) / 2 > 2 * naive_half


def test_interval_coverage_with_clustered_data():
    rng = np.random.default_rng(3)
    cover, n_rep = 0, 80
    for _ in range(n_rep):
        rows = []
        for p in range(40):
            pr = rng.beta(2, 3)                                  # cluster-level rate, mean 0.4
            for _r in range(3):
                tp = int(rng.random() < pr)
                rows.append((f"p{p}", "trapped", tp, 1 - tp, 0, True, False, False, 0.1))
        df = _df(rows)
        st = Study(df.pair_id, B=800, seed=int(rng.integers(1 << 30)))
        e = st.estimate(st.vectors(df, "recall"), 0.95)
        cover += e.lo <= 0.4 <= e.hi
    assert cover / n_rep >= 0.86, cover / n_rep


def test_paired_contrast_uses_common_draws_and_has_the_right_sign():
    rng = np.random.default_rng(4)
    rows_a, rows_b = [], []
    for p in range(40):
        u = rng.normal()
        for _ in range(3):
            pa, pb = 1 / (1 + np.exp(-(1.0 + u))), 1 / (1 + np.exp(-(-1.0 + u)))
            ta, tb = int(rng.random() < pa), int(rng.random() < pb)
            rows_a.append((f"p{p}", "trapped", ta, 1 - ta, 0, True, False, False, 0.1))
            rows_b.append((f"p{p}", "trapped", tb, 1 - tb, 0, True, False, False, 0.1))
    a, b = _df(rows_a), _df(rows_b)
    st = Study(a.pair_id, B=3000, seed=5)
    paired = st.contrast(st.vectors(a, "recall"), st.vectors(b, "recall"), 0.95)
    assert paired.point > 0.3 and paired.lo > 0.1
    # pair-level noise is shared by both conditions, so the paired interval is tighter than two independent ones
    ea, eb = st.estimate(st.vectors(a, "recall"), 0.95), st.estimate(st.vectors(b, "recall"), 0.95)
    indep_half = np.sqrt(((ea.hi - ea.lo) / 2) ** 2 + ((eb.hi - eb.lo) / 2) ** 2)
    assert (paired.hi - paired.lo) / 2 < indep_half


def test_undefined_estimates_are_nan_not_crashes():
    df = _df([("a", "clean", 0, 0, 0, True, False, False, 0.1)])
    st = Study(df.pair_id, B=1000, seed=1)
    e = st.estimate(st.vectors(df, "recall"), 0.95)          # no trapped rows, denominator is zero
    assert not e.defined
