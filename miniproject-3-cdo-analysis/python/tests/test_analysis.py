import json
import math

import numpy as np
import pytest
from scipy.stats import binom

from cdo import BASE_DEAL, SEED, build_results, load_fixed_normals, sensitivity, simulate
from cdo.analysis import (
    exact_default_count_distribution,
    exact_shortfall_probability,
    expected_pool_cash_flows,
    safe_lgd_limits,
)
from cdo.defaults import default_times


@pytest.fixture(scope="module")
def base(normals):
    return simulate(normals, BASE_DEAL)


def brute_force_case(z_row, p):
    """One case with plain loops and the math module, written independently of the package."""
    n = len(z_row)
    n_periods = p["years"] * p["freq"]
    chol = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            target = 1.0 if i == j else p["rho"]
            s = sum(chol[i][k] * chol[j][k] for k in range(j))
            chol[i][j] = math.sqrt(target - s) if i == j else (target - s) / chol[j][j]
    quarters = []
    pool = [0.0] * n_periods
    for i in range(n):
        x = sum(chol[i][k] * z_row[k] for k in range(i + 1))
        u = 0.5 * math.erfc(-x / math.sqrt(2.0))
        years = math.log(1.0 - u) / math.log(1.0 - p["pd"])
        quarter = min(n_periods + 1, max(1, math.ceil(years * p["freq"])))
        quarters.append(quarter)
        for k in range(1, n_periods + 1):
            promised = p["face"] * p["coupon"] / p["freq"] + (p["face"] if k == n_periods else 0.0)
            pool[k - 1] += promised * ((1.0 - p["lgd"]) if k >= quarter else 1.0)
    a, b, equity = [], [], []
    for k in range(1, n_periods + 1):
        last = k == n_periods
        a_due = p["a_notional"] * p["a_coupon"] / p["freq"] + (p["a_notional"] if last else 0.0)
        b_due = p["b_notional"] * p["b_coupon"] / p["freq"] + (p["b_notional"] if last else 0.0)
        cash = pool[k - 1]
        to_a = min(cash, a_due)
        to_b = min(cash - to_a, b_due)
        a.append(to_a)
        b.append(to_b)
        equity.append(cash - to_a - to_b)
    return quarters, pool, a, b, equity


def test_fixed_random_numbers_have_the_expected_shape(normals):
    assert normals.shape == (1000, 10)


def test_fixed_random_numbers_are_the_seeded_draws(tmp_path, raw_normals):
    table, source = load_fixed_normals(tmp_path / "fresh.csv", 1000, 10, SEED)
    assert source.startswith("generated")
    assert np.array_equal(table.values, raw_normals)


def test_loading_a_file_with_the_wrong_shape_is_rejected(tmp_path):
    path = tmp_path / "short.csv"
    path.write_text("case,Bond 1,Bond 2\n1,0.1,0.2\n")
    with pytest.raises(ValueError):
        load_fixed_normals(path, 1000, 10, SEED)


def test_base_case_default_counts(base):
    n_defaults = base["defaulted"].sum(axis=1)
    assert np.bincount(n_defaults, minlength=11).tolist() == [257, 253, 200, 131, 77, 49, 16, 11, 5, 1, 0]
    assert n_defaults.mean() == pytest.approx(1.821)


def test_base_case_pool_and_equity_totals(base):
    pool = base["pool_cf"].sum(axis=1)
    equity = base["eq_cf"].sum(axis=1)
    assert pool.mean() == pytest.approx(117.283, abs=1e-3)
    assert pool.std(ddof=1) == pytest.approx(12.032, abs=1e-3)
    assert pool.min() == pytest.approx(64.48)
    assert equity.mean() == pytest.approx(83.283, abs=1e-3)
    assert np.percentile(equity, 5) == pytest.approx(60.6855, abs=1e-3)
    assert int(np.argmin(equity)) + 1 == 760


def test_case_five_defaults(base):
    quarters = base["Q"][4]
    assert quarters[[0, 3, 4, 5, 6, 8]].tolist() == [4, 20, 5, 13, 12, 7]
    assert (quarters[[1, 2, 7, 9]] == 21).all()
    assert base["pool_cf"][4].sum() == pytest.approx(88.15)


def test_classes_are_never_short_in_the_base_case(base):
    assert np.allclose(base["a_cf"], base["a_due"])
    assert np.allclose(base["b_cf"], base["b_due"])


def test_recovery_floor_covers_both_classes_in_the_base_case(base):
    floor = base["pool_promised"] * (1 - BASE_DEAL["lgd"])
    assert floor[0] == pytest.approx(0.60) and floor[-1] == pytest.approx(40.60)
    assert (floor >= base["a_due"] + base["b_due"]).all()
    limits = safe_lgd_limits(base)
    assert limits["class_b"] == pytest.approx(1 - 30.2 / 101.5)
    assert limits["class_a"] == pytest.approx(1 - 20.1 / 101.5)


def test_simulated_mean_is_close_to_the_exact_expectation(base):
    expected = expected_pool_cash_flows(BASE_DEAL)
    gap = base["pool_cf"].mean(axis=0) / expected - 1
    assert np.abs(gap).max() < 0.005
    assert expected[-1] == pytest.approx(101.5 * (0.4 + 0.6 * 0.96**5))


def test_exact_default_distribution_is_a_proper_distribution_with_the_right_mean():
    p_default = 1 - 0.96**5
    exact = exact_default_count_distribution(10, p_default, 0.2)
    assert exact.sum() == pytest.approx(1.0, abs=1e-12)
    assert (exact * np.arange(11)).sum() == pytest.approx(10 * p_default, abs=1e-10)


def test_exact_default_distribution_without_correlation_is_the_binomial():
    p_default = 1 - 0.96**5
    assert np.allclose(exact_default_count_distribution(10, p_default, 0.0), binom.pmf(np.arange(11), 10, p_default))


def test_exact_default_distribution_agrees_with_a_large_simulation():
    # The quadrature uses one common factor, the simulation uses the Cholesky factor: same model.
    p_default = 1 - 0.96**5
    Z = np.random.default_rng(6).standard_normal((400_000, 10))
    defaults = (default_times(Z, 0.04, 0.2) <= 5.0).sum(axis=1)
    simulated = np.bincount(defaults, minlength=11) / len(defaults)
    assert np.allclose(simulated, exact_default_count_distribution(10, p_default, 0.2), atol=0.003)


def test_fixed_sample_is_consistent_with_the_exact_distribution(base):
    exact = exact_default_count_distribution(10, 1 - 0.96**5, 0.2)
    simulated = np.bincount(base["defaulted"].sum(axis=1), minlength=11) / 1000
    assert np.abs(simulated - exact).max() < 0.04
    assert exact[8:].sum() == pytest.approx(0.0050, abs=5e-4)
    assert exact[9:].sum() == pytest.approx(0.0013, abs=3e-4)


def test_exact_shortfall_probability_in_the_base_and_stressed_deal(base):
    classes_due = base["a_due"] + base["b_due"]
    assert exact_shortfall_probability(BASE_DEAL, classes_due) == 0.0
    stressed = {**BASE_DEAL, "lgd": 1.0}
    exact = exact_default_count_distribution(10, 1 - 0.96**5, 0.2)
    # With nothing recovered Class B needs 3 survivors and Class A needs 2.
    assert exact_shortfall_probability(stressed, classes_due) == pytest.approx(exact[8:].sum())
    assert exact_shortfall_probability(stressed, base["a_due"]) == pytest.approx(exact[9:].sum())


def test_exact_shortfall_probability_edge_cases(base):
    classes_due = base["a_due"] + base["b_due"]
    no_loss = {**BASE_DEAL, "lgd": 0.0}
    assert exact_shortfall_probability(no_loss, classes_due) == 0.0
    # a class owed more than the pool even promises is always short, whatever the LGD
    assert exact_shortfall_probability(no_loss, base["pool_promised"] + 1.0) == 1.0
    # when a coupon quarter binds before maturity there is no closed form, and the function says so
    coupon_heavy = base["pool_promised"] * np.where(np.arange(20) < 19, 0.95, 0.05)
    assert exact_shortfall_probability(BASE_DEAL, coupon_heavy) is None


def test_exact_default_distribution_stays_accurate_at_high_correlation():
    p_default = 1 - 0.96**5
    Z = np.random.default_rng(7).standard_normal((400_000, 10))
    defaults = (default_times(Z, 0.04, 0.9) <= 5.0).sum(axis=1)
    simulated = np.bincount(defaults, minlength=11) / len(defaults)
    exact = exact_default_count_distribution(10, p_default, 0.9)
    assert exact.sum() == pytest.approx(1.0, abs=1e-10)
    assert np.allclose(simulated, exact, atol=0.003)


def test_bad_random_numbers_are_rejected(normals):
    with_gap = normals.copy()
    with_gap[0, 0] = np.nan
    with pytest.raises(ValueError):
        simulate(with_gap, BASE_DEAL)
    with pytest.raises(ValueError):
        simulate(normals[0], BASE_DEAL)


def test_vectorized_model_matches_a_loop_implementation(normals):
    stressed = {**BASE_DEAL, "pd": 0.12, "lgd": 1.0, "rho": 0.6, "b_notional": 30.0}
    for deal in (BASE_DEAL, stressed):
        r = simulate(normals, deal)
        for i in range(normals.shape[0]):
            quarters, pool, a, b, equity = brute_force_case(normals[i], deal)
            assert quarters == r["Q"][i].tolist()
            assert np.allclose(pool, r["pool_cf"][i], atol=1e-9)
            assert np.allclose(a, r["a_cf"][i], atol=1e-9)
            assert np.allclose(b, r["b_cf"][i], atol=1e-9)
            assert np.allclose(equity, r["eq_cf"][i], atol=1e-9)


def test_higher_default_probability_lowers_equity(normals):
    table = sensitivity(normals, BASE_DEAL, "pd", (0.01, 0.04, 0.12))
    assert table["equity mean"].is_monotonic_decreasing
    assert table["avg defaults"].is_monotonic_increasing


def test_lgd_changes_losses_but_not_who_defaults(normals):
    table = sensitivity(normals, BASE_DEAL, "lgd", (0.4, 0.6, 1.0))
    assert table["avg defaults"].nunique() == 1
    assert table["equity mean"].is_monotonic_decreasing
    assert table.loc[0.4, "P(B shortfall)"] == 0
    assert table.loc[1.0, "P(B shortfall)"] > 0


def test_correlation_widens_the_distribution_without_moving_the_mean_much(normals):
    table = sensitivity(normals, BASE_DEAL, "rho", (0.0, 0.2, 0.8))
    assert table["equity std"].is_monotonic_increasing
    assert table["equity 5th pct"].is_monotonic_decreasing
    assert np.ptp(table["equity mean"]) < 1.0


@pytest.mark.parametrize(
    "key, value",
    [("pd", -0.01), ("pd", 1.0), ("lgd", 1.2), ("lgd", -0.1), ("rho", 1.0), ("rho", -0.1), ("face", -1.0)],
)
def test_invalid_inputs_are_rejected(normals, key, value):
    with pytest.raises(ValueError):
        simulate(normals, {**BASE_DEAL, key: value})


def test_wrong_number_of_bonds_is_rejected(normals):
    with pytest.raises(ValueError):
        simulate(normals[:, :5], BASE_DEAL)


def test_simulate_does_not_modify_its_inputs(normals):
    before = normals.copy()
    deal = dict(BASE_DEAL)
    simulate(normals, deal)
    assert np.array_equal(normals, before)
    assert deal == BASE_DEAL


def test_results_are_json_ready_and_consistent(normals):
    results = build_results(normals, BASE_DEAL, SEED)
    json.dumps(results, allow_nan=False)
    checks = results["checks"]
    assert checks["slide_value"] == pytest.approx(984.73, abs=0.005)
    assert checks["class_a_shortfall_cases"] == 0
    assert checks["class_b_shortfall_cases"] == 0
    assert checks["worst_case"] == 760
    assert len(results["cases"]) == 1000
    assert len(results["quarterly"]) == 20
    equity = next(row for row in results["summary"] if row["series"] == "equity")
    assert equity["no-default amount"] == pytest.approx(96.0)
    assert equity["mean"] == pytest.approx(83.283, abs=1e-3)
    assert equity["std error"] == pytest.approx(equity["std dev"] / 1000**0.5)
