"""Part 2: value Class A, Class B and the residual with the Lintner CAPM (Lecture 6).

In every case the cash flows of each tranche are discounted at the risk-free rate. The
bonds' market value (their promised cash flows at the 9% market yield) fixes the cost of
risk, and each tranche is charged for risk in proportion to the covariance of its PV with
the PV of the whole pool.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .bonds import promised_cash_flows
from .model import BASE_DEAL, simulate
from .random_numbers import moment_match
from .waterfall import waterfall

# Part 2 assumptions from the assignment. Dollar amounts are in $ millions. The target
# rates are used as the classes' coupons, so a class sold at par pays its target rate.
PART2_DEAL = {**BASE_DEAL, "a_notional": 15.0, "a_coupon": 0.045, "b_notional": 30.0, "b_coupon": 0.055}
PART2_RISK_FREE = 0.04
TRANCHES = ("Class A", "Class B", "Residual")


def discount_factors(rate: float, freq: int, n_periods: int) -> np.ndarray:
    """Factors that bring a payment at the end of quarter k = 1..n_periods back to today."""
    return (1.0 + rate / freq) ** -np.arange(1, n_periods + 1)


def bond_market_value(p: dict, ytm: float) -> float:
    """Market value of the bonds: the pool's promised cash flows discounted at the market yield."""
    n_periods = p["years"] * p["freq"]
    promised = p["n_bonds"] * promised_cash_flows(p["face"], p["coupon"], n_periods, p["freq"])
    return float(promised @ discount_factors(ytm, p["freq"], n_periods))


def tranche_cash_flows(pool_cf: np.ndarray, p: dict) -> dict:
    """Class A, Class B and residual cash flows (cases x quarters) for the structure in p."""
    n_periods = pool_cf.shape[1]
    a_due = promised_cash_flows(p["a_notional"], p["a_coupon"], n_periods, p["freq"])
    b_due = promised_cash_flows(p["b_notional"], p["b_coupon"], n_periods, p["freq"])
    paid = waterfall(pool_cf, a_due, b_due)
    return {"A": paid["A"], "B": paid["B"], "Residual": paid["Equity"], "a_due": a_due, "b_due": b_due}


def tranche_pvs(flows: dict, rate: float, freq: int) -> np.ndarray:
    """PV at the risk-free rate of each tranche in each case: a cases x 3 table (A, B, residual)."""
    factors = discount_factors(rate, freq, flows["A"].shape[1])
    return np.column_stack([flows[name] @ factors for name in ("A", "B", "Residual")])


def default_indicators(flows: dict, tol: float = 1e-9) -> np.ndarray:
    """1 where a class is paid less than it is due in some quarter, else 0 (cases x 2: A, B)."""
    short_a = (flows["A"] < flows["a_due"] - tol).any(axis=1)
    short_b = (flows["B"] < flows["b_due"] - tol).any(axis=1)
    return np.column_stack([short_a, short_b]).astype(int)


def cost_of_risk(pool_pv: np.ndarray, market_value: float) -> dict:
    """The bonds' implicit cost of risk and the cost per unit of risk (variance), Lecture 6 slide 44."""
    expected = float(pool_pv.mean())
    variance = float(pool_pv.var(ddof=1))
    if not variance > 0:
        raise ValueError("the pool's PV is the same in every case, so there is no risk to price")
    cost = expected - market_value
    return {"market_value": market_value, "expected_pool_pv": expected, "cost_of_risk": cost,
            "pool_variance": variance, "cost_per_unit_risk": cost / variance}


def lintner_values(pv: np.ndarray, cost_per_unit_risk: float) -> dict:
    """Value of each tranche: expected PV less the cost per unit risk times its covariance with the pool."""
    covariance = np.cov(pv, rowvar=False, ddof=1)
    cov_with_pool = covariance.sum(axis=1)
    expected = pv.mean(axis=0)
    charge = cost_per_unit_risk * cov_with_pool
    return {"expected_pv": expected, "covariance": covariance, "cov_with_pool": cov_with_pool,
            "risk_charge": charge, "value": expected - charge}


def value_cdo(pool_cf: np.ndarray, p: dict, rate: float, ytm: float) -> dict:
    """All the Part 2 steps for one structure: PVs, default indicators, cost of risk, Lintner values."""
    flows = tranche_cash_flows(pool_cf, p)
    pv = tranche_pvs(flows, rate, p["freq"])
    risk = cost_of_risk(pv.sum(axis=1), bond_market_value(p, ytm))
    return {"pv": pv, "defaults": default_indicators(flows), **risk,
            **lintner_values(pv, risk["cost_per_unit_risk"])}


def structure_values(pool_cf: np.ndarray, p: dict, rate: float, cost_per_unit_risk: float) -> dict:
    """Lintner values and default probabilities for another structure on the same pool.

    The cost per unit risk comes from the bonds alone, so it stays the same whatever the
    classes look like. Only each tranche's covariance with the pool has to be recomputed.
    """
    flows = tranche_cash_flows(pool_cf, p)
    pv = tranche_pvs(flows, rate, p["freq"])
    pool = pv.sum(axis=1)
    cov_with_pool = (pv - pv.mean(axis=0)).T @ (pool - pool.mean()) / (len(pool) - 1)
    return {"value": pv.mean(axis=0) - cost_per_unit_risk * cov_with_pool,
            "p_default": default_indicators(flows).mean(axis=0)}


def fair_coupon(pool_cf: np.ndarray, p: dict, rate: float, cost_per_unit_risk: float, cls: str = "b",
                coupons: np.ndarray | None = None) -> float | None:
    """Lowest coupon at which Class A or B is worth its principal, or None if no coupon up to 100% does it.

    A class that is too large stays below par whatever its coupon. The value is not always rising
    in the coupon (the Lintner weights of the no-default cases are negative), so the coupons are
    scanned upwards (0% to 100% in 1% steps unless coupons are given) to find the first one that
    reaches par before the root is refined.
    """
    if coupons is None:
        coupons = np.linspace(0.0, 1.0, 101)
    if cls not in ("a", "b"):
        raise ValueError(f"class must be 'a' or 'b', got {cls!r}")
    principal = p[f"{cls}_notional"]
    if not principal > 0:
        raise ValueError(f"Class {cls.upper()} has no principal to price")
    column = 0 if cls == "a" else 1

    def gap(coupon):
        return structure_values(pool_cf, {**p, f"{cls}_coupon": coupon}, rate, cost_per_unit_risk)["value"][column] - principal

    previous = None
    for coupon in coupons:
        if gap(coupon) >= 0:
            return float(coupon) if previous is None else brentq(gap, previous, coupon, xtol=1e-12)
        previous = coupon
    return None


def break_even_size(pool_cf: np.ndarray, p: dict, rate: float, cost_per_unit_risk: float,
                    sizes: np.ndarray) -> float | None:
    """Largest Class B principal at which its coupon in p is still worth par (value = principal).

    Below this size the coupon pays Class B more than its risk needs; above it Class B is
    worth less than par. Returns None if Class B is worth at least par at every size given.
    """
    def gap(size):
        return structure_values(pool_cf, {**p, "b_notional": size}, rate, cost_per_unit_risk)["value"][1] - size

    positive = [float(size) for size in sizes if size > 0]
    gaps = [gap(size) for size in positive]
    for (size_lo, gap_lo), (size_hi, gap_hi) in zip(zip(positive, gaps), zip(positive[1:], gaps[1:])):
        if gap_lo >= 0 > gap_hi:
            return brentq(gap, size_lo, size_hi, xtol=1e-10)
    return None


def gain_turns_positive(at_target: list) -> float | None:
    """Class B principal at which the bank's gain at the target coupon first crosses zero (linear
    interpolation between the sizes searched), or None if it never does."""
    for low, high in zip(at_target, at_target[1:]):
        if low["bank_gain"] < 0 <= high["bank_gain"]:
            t = -low["bank_gain"] / (high["bank_gain"] - low["bank_gain"])
            return low["b_notional"] + t * (high["b_notional"] - low["b_notional"])
    return None


def largest_class_b_at_par(pool_cf: np.ndarray, p: dict, rate: float, cost_per_unit_risk: float,
                           low: float, high: float, tol: float = 0.005) -> float:
    """Largest Class B principal that some coupon (up to 100%) can make worth par, by bisection
    between a size that can (low) and one that cannot (high)."""
    def can(size):
        return fair_coupon(pool_cf, {**p, "b_notional": size}, rate, cost_per_unit_risk) is not None

    if not can(low) or can(high):
        raise ValueError("the search for the largest Class B needs a size that can reach par and one that cannot")
    while high - low > tol:
        middle = (low + high) / 2
        low, high = (middle, high) if can(middle) else (low, middle)
    return low


def optimise_class_b(pool_cf: np.ndarray, p: dict, rate: float, ytm: float, sizes: np.ndarray,
                     coupons: np.ndarray, default_limits: tuple = (0.0, 0.01, 0.05, 0.10, 0.20)) -> dict:
    """The bonus: search Class B's principal and coupon with Class A held as in p.

    Returns the literal answer (the structure that gives the residual the highest Lintner
    value), the bank's position at Class B's target rate (cash from selling both classes at
    par plus the residual's value), Class B's fair coupon at each size, and the best size at
    the target rate when Class B's default probability is capped.
    """
    base = value_cdo(pool_cf, p, rate, ytm)
    lam, market_value = base["cost_per_unit_risk"], base["market_value"]

    grid = []
    for size in sizes:
        for coupon in coupons:
            s = structure_values(pool_cf, {**p, "b_notional": float(size), "b_coupon": float(coupon)}, rate, lam)
            grid.append({"b_notional": float(size), "b_coupon": float(coupon), "value_a": float(s["value"][0]),
                         "value_b": float(s["value"][1]), "value_residual": float(s["value"][2]),
                         "p_default_b": float(s["p_default"][1])})
    literal = max(grid, key=lambda row: row["value_residual"])

    at_target = []
    for size in sizes:
        s = structure_values(pool_cf, {**p, "b_notional": float(size)}, rate, lam)
        bank = p["a_notional"] + float(size) + float(s["value"][2])
        at_target.append({"b_notional": float(size), "value_a": float(s["value"][0]), "value_b": float(s["value"][1]),
                          "value_residual": float(s["value"][2]), "p_default_b": float(s["p_default"][1]),
                          "bank_position": bank, "bank_gain": bank - market_value})

    fair = []
    for size in sizes:
        if size <= 0:
            continue
        coupon = fair_coupon(pool_cf, {**p, "b_notional": float(size)}, rate, lam)
        if coupon is None:
            break
        s = structure_values(pool_cf, {**p, "b_notional": float(size), "b_coupon": coupon}, rate, lam)
        bank = p["a_notional"] + float(size) + float(s["value"][2])
        fair.append({"b_notional": float(size), "fair_coupon": coupon, "value_residual": float(s["value"][2]),
                     "p_default_b": float(s["p_default"][1]), "bank_gain": bank - market_value})

    break_even = break_even_size(pool_cf, p, rate, lam, sizes)
    capped = []
    for limit in default_limits:
        allowed = [row for row in at_target if row["p_default_b"] <= limit + 1e-12]
        best = max(allowed, key=lambda row: row["bank_gain"])
        capped.append({"max_p_default_b": limit, "b_notional": best["b_notional"],
                       "p_default_b": best["p_default_b"], "bank_gain": best["bank_gain"]})

    no_b = structure_values(pool_cf, {**p, "b_notional": 0.0}, rate, lam)
    bonds_less_a = market_value - float(no_b["value"][0])
    return {
        "a_notional": p["a_notional"], "a_coupon": p["a_coupon"], "target_b_coupon": p["b_coupon"],
        "market_value": market_value, "cost_per_unit_risk": lam,
        "literal": literal,
        "residual_with_no_class_b": float(no_b["value"][2]),
        "bonds_less_class_a": bonds_less_a,
        "largest_class_b_at_par": largest_class_b_at_par(pool_cf, p, rate, lam, bonds_less_a - 2, bonds_less_a + 2),
        "min_residual_value_in_grid": min(row["value_residual"] for row in grid),
        "break_even_size": break_even,
        "gain_turns_positive": gain_turns_positive(at_target),
        "p_default_b_at_break_even": (float(structure_values(pool_cf, {**p, "b_notional": break_even}, rate,
                                                             lam)["p_default"][1]) if break_even is not None else None),
        "at_target_rate": at_target,
        "fair_coupon_curve": fair,
        "default_limits": capped,
        "grid": grid,
    }


def value_sensitivity(Z: np.ndarray, p: dict, rate: float, ytm: float, name: str, values: tuple) -> list:
    """Re-value the structure with one model input changed. The bonds' market value stays at its
    price at the market yield, so the cost of risk moves with the model's expected cash flows."""
    rows = []
    for x in values:
        q = {**p, name: x}
        v = value_cdo(simulate(Z, q)["pool_cf"], q, rate, ytm)
        rows.append({name: x, "expected_pool_pv": v["expected_pool_pv"], "cost_of_risk": v["cost_of_risk"],
                     "cost_per_unit_risk": v["cost_per_unit_risk"], "value_a": float(v["value"][0]),
                     "value_b": float(v["value"][1]), "value_residual": float(v["value"][2]),
                     "p_default_a": float(v["defaults"][:, 0].mean()), "p_default_b": float(v["defaults"][:, 1].mean())})
    return rows


def valuation_precision(p: dict, rate: float, ytm: float, n_cases: int, n_sets: int = 500, seed: int = 6105) -> dict:
    """How much the Lintner values move from one set of random numbers to another.

    Draws n_sets fresh tables of n_cases x n_bonds normals and values the structure on each,
    once as drawn and once after moment matching.
    """
    rng = np.random.default_rng(seed)
    as_drawn, matched = [], []
    for _ in range(n_sets):
        Z = rng.standard_normal((n_cases, p["n_bonds"]))
        for store, table in ((as_drawn, Z), (matched, moment_match(Z))):
            v = value_cdo(simulate(table, p)["pool_cf"], p, rate, ytm)
            store.append(list(v["value"]) + [v["cost_per_unit_risk"], v["expected_pool_pv"], v["expected_pv"][2]])
    as_drawn, matched = np.array(as_drawn), np.array(matched)
    return {"n_sets": n_sets, "seed": seed,
            "columns": ["value_a", "value_b", "value_residual", "cost_per_unit_risk", "expected_pool_pv",
                        "expected_pv_residual"],
            "sd_as_drawn": as_drawn.std(axis=0, ddof=1).tolist(), "sd_matched": matched.std(axis=0, ddof=1).tolist(),
            "mean_as_drawn": as_drawn.mean(axis=0).tolist(), "mean_matched": matched.mean(axis=0).tolist()}
