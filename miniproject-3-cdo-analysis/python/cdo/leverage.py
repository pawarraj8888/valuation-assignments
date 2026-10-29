"""Bonus, the bank's side: how much of the bonds to fund with Class B (the Leverage sheet of the workbook).

Each class is sold for what it is worth: Class A for its Lintner value and Class B at par with its
fair coupon. The bank pays the rest of the bonds' price, which is always exactly the Lintner value of
the residual it keeps, so a bigger Class B adds no value. What it changes is the bank's risk and
expected return: less of the bank's own money, a higher expected IRR and a riskier stake. The risk
measure is the standard deviation of the residual's PV across the cases divided by the bank's
investment (the relative equity risk). For a cap on that measure we find the largest Class B, which
is also the highest expected IRR, since both rise with Class B's size.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .valuation import discount_factors, fair_coupon, structure_values, value_cdo

# The Goal Seek answers on the Leverage sheet (Leverage!A51:I62): cap on the relative equity risk
# and the largest Class B principal ($MM) found for it.
GOAL_SEEK = ((0.20, 22.83), (0.25, 32.83), (0.30, 39.5), (0.35, 44.75), (0.40, 48.5), (0.45, 51.75),
             (0.50, 54.15), (0.55, 56.15), (0.60, 58.0), (0.65, 59.47), (0.70, 60.64), (1.00, 65.25))
RISK_LIMITS = tuple(limit for limit, _ in GOAL_SEEK)


def irr(investment: float, cash_flows: np.ndarray) -> float:
    """Rate per period at which cash_flows, received at the end of periods 1..n, are worth the investment."""
    if not investment > 0:
        raise ValueError(f"the IRR needs a positive investment, got {investment}")
    periods = np.arange(1, len(cash_flows) + 1)

    def npv(rate):
        return cash_flows @ (1.0 + rate) ** -periods - investment

    return brentq(npv, -0.99, 10.0, xtol=1e-14)


def bank_position(pool_cf: np.ndarray, p: dict, rate: float, market_value: float, cost_per_unit_risk: float) -> dict:
    """The bank's investment, expected IRR and relative equity risk for the structure in p, with each
    class sold for its Lintner value.

    The expected IRR is the IRR of the investment against the residual's average cash flow in each
    quarter, compounded to a year. The relative equity risk is the standard deviation of the
    residual's PV at the risk-free rate across the cases, divided by the investment.
    """
    s = structure_values(pool_cf, p, rate, cost_per_unit_risk)
    value_a, value_b, value_residual = (float(x) for x in s["value"])
    investment = market_value - value_a - value_b
    periods_per_year = p["freq"]
    expected_irr = (1.0 + irr(investment, s["flows"]["Residual"].mean(axis=0))) ** periods_per_year - 1.0
    promised_b = float(s["flows"]["b_due"] @ discount_factors(rate, periods_per_year, pool_cf.shape[1]))
    sd = float(s["pv"][:, 2].std(ddof=1))
    return {"b_notional": float(p["b_notional"]), "b_coupon": float(p["b_coupon"]), "value_a": value_a,
            "value_b": value_b, "value_residual": value_residual, "investment": investment,
            "expected_credit_loss_b": promised_b - float(s["pv"][:, 1].mean()),
            "risk_charge_b": float(s["risk_charge"][1]), "p_default_b": float(s["p_default"][1]),
            "expected_irr": expected_irr,
            "excess_irr": expected_irr - ((1.0 + rate / periods_per_year) ** periods_per_year - 1.0),
            "sd_residual_pv": sd, "relative_risk": sd / investment}


def leverage_point(pool_cf: np.ndarray, p: dict, rate: float, market_value: float, cost_per_unit_risk: float,
                   size: float) -> dict | None:
    """The bank's position with a Class B of the given principal sold at par at its fair coupon,
    or None if no coupon makes a Class B that large worth par."""
    coupon = fair_coupon(pool_cf, {**p, "b_notional": size}, rate, cost_per_unit_risk)
    if coupon is None:
        return None
    return bank_position(pool_cf, {**p, "b_notional": size, "b_coupon": coupon}, rate, market_value, cost_per_unit_risk)


def max_class_b_for_risk(pool_cf: np.ndarray, p: dict, rate: float, market_value: float, cost_per_unit_risk: float,
                         limit: float, low: float, high: float) -> dict | None:
    """Largest Class B (at its fair coupon) whose relative equity risk is at most the limit.

    The relative risk rises with Class B's size, so the answer is the size at which it equals the
    limit, found by root finding between low (under the limit) and high (over it). Returns None when
    the risk is over the limit even at low.
    """
    def excess(size):
        return leverage_point(pool_cf, p, rate, market_value, cost_per_unit_risk, size)["relative_risk"] - limit

    if excess(low) > 0:
        return None
    if excess(high) <= 0:
        raise ValueError(f"the relative risk is still under {limit} at a Class B of {high}")
    size = brentq(excess, low, high, xtol=1e-9)
    return {"max_relative_risk": limit, **leverage_point(pool_cf, p, rate, market_value, cost_per_unit_risk, size)}


def leverage_analysis(pool_cf: np.ndarray, p: dict, rate: float, ytm: float, sizes: np.ndarray,
                      limits: tuple = RISK_LIMITS) -> dict:
    """The Leverage sheet in Python: the bank's position at the structure in p, with no Class B, at each
    Class B size (fair coupon), and the largest Class B for each cap on the relative equity risk."""
    base = value_cdo(pool_cf, p, rate, ytm)
    lam, market_value = base["cost_per_unit_risk"], base["market_value"]
    profile = []
    for size in sizes:
        if not size > 0:
            continue
        point = leverage_point(pool_cf, p, rate, market_value, lam, float(size))
        if point is None:
            break
        profile.append(point)
    low, high = profile[0]["b_notional"], profile[-1]["b_notional"]
    goal_seek = dict(GOAL_SEEK)
    limited = []
    for limit in limits:
        row = max_class_b_for_risk(pool_cf, p, rate, market_value, lam, limit, low, high)
        limited.append(None if row is None else {**row, "goal_seek_b_notional": goal_seek.get(limit)})
    return {
        "market_value": market_value,
        "cost_per_unit_risk": lam,
        "risk_free_annual": (1.0 + rate / p["freq"]) ** p["freq"] - 1.0,
        "as_given": bank_position(pool_cf, p, rate, market_value, lam),
        "no_class_b": bank_position(pool_cf, {**p, "b_notional": 0.0}, rate, market_value, lam),
        "profile": profile,
        "risk_limits": limited,
    }
