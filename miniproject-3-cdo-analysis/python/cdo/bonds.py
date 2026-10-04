"""Promised bond cash flows and the BIS defaultable bond model (Lecture 5)."""
from __future__ import annotations

import numpy as np


def promised_cash_flows(face: float, coupon: float, n_periods: int, freq: int) -> np.ndarray:
    """Coupon every period and the face value with the last coupon."""
    cf = np.full(n_periods, face * coupon / freq)
    cf[-1] += face
    return cf


def bis_cash_flows(promised: np.ndarray, default_period: np.ndarray, lgd: float) -> np.ndarray:
    """Cash flows of bonds given the period in which each one defaults.

    BIS model: (1 - lgd) of every promised payment is always paid and the other
    lgd stops in the period of default. default_period can be any shape; the
    result has one more axis for the periods. A default_period beyond the last
    period means the bond never defaults.
    """
    periods = np.arange(1, len(promised) + 1)
    default_period = np.asarray(default_period)[..., None]
    paid_share = np.where(periods >= default_period, 1.0 - lgd, 1.0)
    return promised * paid_share


def bis_expected_value(promised: np.ndarray, pd_per_period: float, lgd: float, rate_per_period: float) -> tuple:
    """Probability-weighted present value over every possible default period.

    This is the table on the BIS slide in Lecture 5. Returns (rows, value) where
    each row is (default period, probability, PV) and period None means no default.
    """
    n = len(promised)
    discount = (1.0 + rate_per_period) ** -np.arange(1, n + 1)
    rows = []
    for k in range(1, n + 2):
        survived = k > n
        prob = (1.0 - pd_per_period) ** n if survived else pd_per_period * (1.0 - pd_per_period) ** (k - 1)
        pv = float((bis_cash_flows(promised, k, lgd) * discount).sum())
        rows.append((None if survived else k, prob, pv))
    value = sum(prob * pv for _, prob, pv in rows)
    return rows, value
