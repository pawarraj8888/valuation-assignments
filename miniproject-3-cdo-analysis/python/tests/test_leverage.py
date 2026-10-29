import numpy as np
import pytest

from cdo import MARKET_YTM, PART2_DEAL, PART2_RISK_FREE, simulate
from cdo.leverage import GOAL_SEEK, bank_position, irr, leverage_analysis, leverage_point, max_class_b_for_risk
from cdo.valuation import structure_values, value_cdo

# The Leverage sheet as Michael saved it: Class B $65.25 MM at the coupon Goal Seek found (Inputs!B16:B17),
# and what the sheet shows for it (Leverage!B6, B24, B29, B13, B12, B16).
SHEET_B, SHEET_COUPON = 65.25, 0.054706907821240935
SHEET = {"investment": 7.438861604400913, "expected_irr": 0.26141365881089973, "relative_risk": 0.9997762574704259,
         "risk_charge_b": 3.0637584976862255, "expected_credit_loss_b": 1.265485761014375, "p_default_b": 0.159}


@pytest.fixture(scope="module")
def pool_cf(normals):
    return simulate(normals, PART2_DEAL)["pool_cf"]


@pytest.fixture(scope="module")
def market(pool_cf):
    v = value_cdo(pool_cf, PART2_DEAL, PART2_RISK_FREE, MARKET_YTM)
    return v["market_value"], v["cost_per_unit_risk"]


@pytest.fixture(scope="module")
def analysis(pool_cf):
    return leverage_analysis(pool_cf, PART2_DEAL, PART2_RISK_FREE, MARKET_YTM, np.arange(0.0, 72.01, 1.0))


def test_irr_of_a_bond_bought_at_par_is_its_coupon_rate():
    assert irr(100.0, np.array([5.0, 5.0, 105.0])) == pytest.approx(0.05, abs=1e-12)
    with pytest.raises(ValueError):
        irr(0.0, np.array([1.0, 1.0]))


def test_bank_investment_is_the_lintner_value_of_the_residual(pool_cf, market):
    for p in (PART2_DEAL, {**PART2_DEAL, "b_notional": 60.0, "b_coupon": 0.07}):
        x = bank_position(pool_cf, p, PART2_RISK_FREE, *market)
        assert x["investment"] == pytest.approx(x["value_residual"], abs=1e-9)


def test_matches_the_leverage_sheet_at_its_saved_inputs(pool_cf, market):
    x = bank_position(pool_cf, {**PART2_DEAL, "b_notional": SHEET_B, "b_coupon": SHEET_COUPON}, PART2_RISK_FREE, *market)
    for name, sheet_value in SHEET.items():
        assert x[name] == pytest.approx(sheet_value, abs=1e-9), name
    assert x["value_b"] == pytest.approx(SHEET_B, abs=1e-9)


def test_risk_and_expected_irr_both_rise_with_class_b(analysis):
    risk = [row["relative_risk"] for row in analysis["profile"]]
    irrs = [row["expected_irr"] for row in analysis["profile"]]
    assert np.all(np.diff(risk) > 0) and np.all(np.diff(irrs) > 0)
    assert analysis["no_class_b"]["relative_risk"] < risk[0]


def test_largest_class_b_meets_the_cap_and_one_step_more_breaks_it(pool_cf, market, analysis):
    for row in analysis["risk_limits"]:
        cap = row["max_relative_risk"]
        assert row["relative_risk"] == pytest.approx(cap, abs=1e-8)
        assert leverage_point(pool_cf, PART2_DEAL, PART2_RISK_FREE, *market, row["b_notional"] + 0.01)["relative_risk"] > cap
        value_b = structure_values(pool_cf, {**PART2_DEAL, "b_notional": row["b_notional"], "b_coupon": row["b_coupon"]},
                                   PART2_RISK_FREE, market[1])["value"][1]
        assert value_b == pytest.approx(row["b_notional"], abs=1e-8)
        assert abs(row["b_notional"] - dict(GOAL_SEEK)[cap]) < 0.2


def test_no_class_b_fits_under_a_cap_below_the_bonds_own_risk(pool_cf, market, analysis):
    floor = analysis["no_class_b"]["relative_risk"]
    assert max_class_b_for_risk(pool_cf, PART2_DEAL, PART2_RISK_FREE, *market, floor / 2, 1.0, 70.0) is None
