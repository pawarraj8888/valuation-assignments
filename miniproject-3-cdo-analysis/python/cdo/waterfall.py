"""The CDO waterfall: Class A, then Class B, then the bank's equity."""
from __future__ import annotations

import numpy as np


def waterfall(pool_cf: np.ndarray, a_due: np.ndarray, b_due: np.ndarray) -> dict:
    """Split pool cash flows (cases x periods) between Class A, Class B and equity.

    Each period stands alone: Class A is paid first up to what it is due, then
    Class B, and the residual goes to equity. No carry-forwards, so a shortfall
    is not made up later and nothing is held back.
    """
    a_paid = np.minimum(pool_cf, a_due)
    b_paid = np.minimum(pool_cf - a_paid, b_due)
    equity = pool_cf - a_paid - b_paid
    return {"A": a_paid, "B": b_paid, "Equity": equity}
