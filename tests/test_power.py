from failurelab.power import simulate_power


def test_probabilities_are_a_distribution():
    res = simulate_power(20, 2, 0.2, 0.6, sims=30, seed=1)
    for h, t in res.items():
        assert abs(sum(t.values()) - 1.0) < 1e-9, h


def test_more_pairs_means_more_power_for_a_real_effect():
    small = simulate_power(10, 3, 0.15, 0.75, sims=60, seed=2)["H2"]["supported"]
    large = simulate_power(60, 3, 0.15, 0.75, sims=60, seed=2)["H2"]["supported"]
    assert large > small and large > 0.6


def test_null_effect_is_almost_never_declared_supported():
    res = simulate_power(40, 3, 0.2, 0.2, sims=80, seed=3)["H2"]
    assert res["supported"] <= 0.05


def test_a_false_flag_cost_blocks_support():
    """Checklist raises recall a lot but also makes the agent flag clean tasks: H2 must not be supported."""
    res = simulate_power(60, 3, 0.2, 0.8, ffr_none=0.05, ffr_checklist=0.45, sims=60, seed=4)["H2"]
    assert res["supported"] <= 0.05
