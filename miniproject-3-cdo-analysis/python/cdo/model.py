"""Deal parameters and the end-to-end simulation for one parameter set."""
from __future__ import annotations

import numpy as np

from .bonds import bis_cash_flows, promised_cash_flows
from .defaults import default_period, default_times
from .waterfall import waterfall

# Base case from the assignment. Dollar amounts are in $ millions.
BASE_DEAL = {
    "n_bonds": 10,
    "years": 5,
    "freq": 4,  # payments per year
    "face": 10.0,  # per bond
    "coupon": 0.06,  # annual, paid quarterly
    "pd": 0.04,  # annual default probability (pi)
    "lgd": 0.60,  # loss given default (lambda)
    "rho": 0.20,  # "correlation" between any two bonds' default dates
    "a_notional": 20.0,  # Class A
    "a_coupon": 0.02,
    "b_notional": 10.0,  # Class B
    "b_coupon": 0.04,
}

# Given in the assignment, needed only for the valuation in Part 2.
MARKET_YTM = 0.09
RISK_FREE = 0.01


def validate_deal(p: dict) -> None:
    """Raise ValueError if an input is outside the range the model can handle."""
    if not 0.0 <= p["pd"] < 1.0:
        raise ValueError(f"annual default probability must be in [0, 1), got {p['pd']}")
    if not 0.0 <= p["lgd"] <= 1.0:
        raise ValueError(f"LGD must be in [0, 1], got {p['lgd']}")
    if not 0.0 <= p["rho"] < 1.0:
        raise ValueError(f"correlation must be in [0, 1), got {p['rho']}")
    for key in ("face", "coupon", "a_notional", "a_coupon", "b_notional", "b_coupon"):
        if not p[key] >= 0:
            raise ValueError(f"{key} must be a number that is not negative, got {p[key]}")


def simulate(Z: np.ndarray, p: dict) -> dict:
    """Run the whole model for the parameter set p on the independent normals Z (cases x bonds)."""
    validate_deal(p)
    if Z.ndim != 2 or Z.shape[1] != p["n_bonds"]:
        raise ValueError(f"random numbers must be a cases x {p['n_bonds']} table, got shape {Z.shape}")
    if not np.isfinite(Z).all():
        raise ValueError("random numbers contain values that are not finite")
    n_periods = p["years"] * p["freq"]
    bond_promised = promised_cash_flows(p["face"], p["coupon"], n_periods, p["freq"])
    T = default_times(Z, p["pd"], p["rho"])
    Q = default_period(T, p["freq"], n_periods)
    bond_cf = bis_cash_flows(bond_promised, Q, p["lgd"])
    pool_cf = bond_cf.sum(axis=1)
    a_due = promised_cash_flows(p["a_notional"], p["a_coupon"], n_periods, p["freq"])
    b_due = promised_cash_flows(p["b_notional"], p["b_coupon"], n_periods, p["freq"])
    paid = waterfall(pool_cf, a_due, b_due)
    return {
        "T": T,
        "Q": Q,
        "defaulted": Q <= n_periods,
        "bond_promised": bond_promised,
        "pool_promised": p["n_bonds"] * bond_promised,
        "bond_cf": bond_cf,
        "pool_cf": pool_cf,
        "a_due": a_due,
        "b_due": b_due,
        "a_cf": paid["A"],
        "b_cf": paid["B"],
        "eq_cf": paid["Equity"],
    }
