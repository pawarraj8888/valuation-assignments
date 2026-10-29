import numpy as np
import pytest

from cdo import MARKET_YTM, PART2_DEAL, PART2_RISK_FREE, simulate
from cdo.bonds import promised_cash_flows
from cdo.valuation import (
    bond_market_value,
    break_even_size,
    cost_of_risk,
    discount_factors,
    fair_coupon,
    optimise_class_b,
    structure_values,
    tranche_cash_flows,
    value_cdo,
)


@pytest.fixture(scope="module")
def pool_cf(normals):
    return simulate(normals, PART2_DEAL)["pool_cf"]


@pytest.fixture(scope="module")
def base_value(pool_cf):
    return value_cdo(pool_cf, PART2_DEAL, PART2_RISK_FREE, MARKET_YTM)


def test_bonds_market_value_is_the_promised_cash_flows_at_9_percent():
    promised = 10 * promised_cash_flows(10.0, 0.06, 20, 4)
    by_hand = sum(cf / (1 + 0.09 / 4) ** k for k, cf in enumerate(promised, start=1))
    assert bond_market_value(PART2_DEAL, 0.09) == pytest.approx(by_hand, abs=1e-12)
    assert by_hand == pytest.approx(88.027216, abs=1e-6)


def test_values_add_up_to_the_bonds_market_value(pool_cf, base_value):
    assert base_value["value"].sum() == pytest.approx(base_value["market_value"], abs=1e-9)
    other = value_cdo(pool_cf, {**PART2_DEAL, "b_notional": 60.0, "b_coupon": 0.07}, PART2_RISK_FREE, MARKET_YTM)
    assert other["value"].sum() == pytest.approx(other["market_value"], abs=1e-9)


def test_cost_per_unit_risk_follows_lecture_6(base_value):
    pool = base_value["pv"].sum(axis=1)
    expected = (pool.mean() - base_value["market_value"]) / pool.var(ddof=1)
    assert base_value["cost_per_unit_risk"] == pytest.approx(expected, rel=1e-12)
    # pinned to the Valuation sheet of the workbook (cell L8)
    assert base_value["cost_per_unit_risk"] == pytest.approx(0.10533475038549751, rel=1e-9)


def test_lintner_values_match_the_workbook(base_value):
    # Valuation!B8:D8 of the Part B workbook
    assert base_value["value"] == pytest.approx([15.338354118117447, 32.03012470870477, 40.65873689569592], abs=1e-9)


def test_values_do_not_depend_on_the_covariance_divisor(base_value):
    pv = base_value["pv"]
    centred = pv - pv.mean(axis=0)
    pool = centred.sum(axis=1)
    population_lambda = (pv.sum(axis=1).mean() - base_value["market_value"]) / (pool @ pool / len(pv))
    population_values = pv.mean(axis=0) - population_lambda * (centred.T @ pool / len(pv))
    assert population_values == pytest.approx(base_value["value"], abs=1e-9)


def test_a_class_that_is_never_short_is_worth_its_cash_flows_at_the_risk_free_rate(base_value):
    assert base_value["defaults"].sum() == 0
    factors = discount_factors(PART2_RISK_FREE, 4, 20)
    a_due = promised_cash_flows(15.0, 0.045, 20, 4)
    assert base_value["value"][0] == pytest.approx(a_due @ factors, abs=1e-9)
    assert base_value["risk_charge"][:2] == pytest.approx([0.0, 0.0], abs=1e-12)


def test_default_indicator_flags_cases_with_a_quarterly_shortfall(normals):
    stressed = {**PART2_DEAL, "lgd": 1.0}
    pool_cf = simulate(normals, stressed)["pool_cf"]
    v = value_cdo(pool_cf, stressed, PART2_RISK_FREE, MARKET_YTM)
    flows = tranche_cash_flows(pool_cf, stressed)
    short_b = (flows["B"] < flows["b_due"] - 1e-9).any(axis=1)
    assert short_b.sum() > 0
    assert np.array_equal(v["defaults"][:, 1], short_b.astype(int))


def test_cost_of_risk_needs_some_risk():
    with pytest.raises(ValueError):
        cost_of_risk(np.full(1000, 95.0), 90.0)


def test_fair_coupon_makes_a_class_worth_par(pool_cf, base_value):
    lam = base_value["cost_per_unit_risk"]
    for size in (30.0, 60.0):
        p = {**PART2_DEAL, "b_notional": size}
        coupon = fair_coupon(pool_cf, p, PART2_RISK_FREE, lam)
        value = structure_values(pool_cf, {**p, "b_coupon": coupon}, PART2_RISK_FREE, lam)["value"][1]
        assert value == pytest.approx(size, abs=1e-8)
    assert fair_coupon(pool_cf, PART2_DEAL, PART2_RISK_FREE, lam, "a") == pytest.approx(0.04, abs=1e-9)
    assert fair_coupon(pool_cf, {**PART2_DEAL, "b_notional": 80.0}, PART2_RISK_FREE, lam) is None


def test_break_even_size_is_where_the_target_rate_is_fair(pool_cf, base_value):
    lam = base_value["cost_per_unit_risk"]
    size = break_even_size(pool_cf, PART2_DEAL, PART2_RISK_FREE, lam, np.arange(0.0, 101.0, 1.0))
    value = structure_values(pool_cf, {**PART2_DEAL, "b_notional": size}, PART2_RISK_FREE, lam)["value"][1]
    assert value == pytest.approx(size, abs=1e-7)
    assert 60 < size < 70


def test_bonus_literal_answer_and_flat_gain_at_fair_coupons(pool_cf):
    bonus = optimise_class_b(pool_cf, PART2_DEAL, PART2_RISK_FREE, MARKET_YTM, np.arange(0.0, 81.0, 5.0),
                             np.array([0.0, 0.04, 0.055, 0.08]))
    assert bonus["literal"]["b_notional"] == 0.0
    assert bonus["literal"]["value_residual"] == pytest.approx(bonus["residual_with_no_class_b"], abs=1e-12)
    gains = [row["bank_gain"] for row in bonus["fair_coupon_curve"]]
    assert np.ptp(gains) < 1e-6
    assert gains[0] == pytest.approx(15.0 - 15.338354118117447, abs=1e-6)
    # a little above the bonds less Class A: the no-default cases carry a negative Lintner weight
    assert bonus["bonds_less_class_a"] == pytest.approx(bonus["market_value"] - 15.338354118117447, abs=1e-9)
    assert bonus["bonds_less_class_a"] < bonus["largest_class_b_at_par"] < bonus["bonds_less_class_a"] + 0.5
    assert bonus["min_residual_value_in_grid"] < 0


def test_fair_coupon_finds_the_lowest_coupon_when_value_is_not_monotonic(pool_cf, base_value):
    lam = base_value["cost_per_unit_risk"]
    size = 72.75   # only very high coupons reach par here
    coupon = fair_coupon(pool_cf, {**PART2_DEAL, "b_notional": size}, PART2_RISK_FREE, lam)
    assert coupon is not None and coupon > 0.2
    value = structure_values(pool_cf, {**PART2_DEAL, "b_notional": size, "b_coupon": coupon}, PART2_RISK_FREE, lam)
    assert value["value"][1] == pytest.approx(size, abs=1e-7)
    assert value["value"][2] < 0


def test_bank_gain_at_the_target_rate_turns_positive_after_the_break_even_size(pool_cf):
    bonus = optimise_class_b(pool_cf, PART2_DEAL, PART2_RISK_FREE, MARKET_YTM, np.arange(60.0, 71.0, 1.0),
                             np.array([0.055]), default_limits=(0.2,))
    assert bonus["break_even_size"] < bonus["gain_turns_positive"] < bonus["break_even_size"] + 2
    assert 66.0 < bonus["gain_turns_positive"] < 67.0
