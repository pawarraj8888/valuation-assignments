"""Correlated default times from fixed normals (Gaussian copula, geometric default time)."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def correlation_matrix(n: int, rho: float) -> np.ndarray:
    """n x n matrix with 1 on the diagonal and rho everywhere else."""
    corr = np.full((n, n), float(rho))
    np.fill_diagonal(corr, 1.0)
    return corr


def correlate(Z: np.ndarray, rho: float) -> np.ndarray:
    """Turn independent normals (cases x bonds) into normals with pairwise correlation rho."""
    chol = np.linalg.cholesky(correlation_matrix(Z.shape[1], rho))
    return Z @ chol.T


def default_times(Z: np.ndarray, pd_annual: float, rho: float) -> np.ndarray:
    """Default time in years for every case and bond.

    u = N(x) for the correlated normals x, then t = ln(1 - u) / ln(1 - pi) as in
    Lecture 5. With pi = 0 nothing ever defaults.
    """
    if pd_annual == 0:
        return np.full(Z.shape, np.inf)
    U = norm.cdf(correlate(Z, rho))
    with np.errstate(divide="ignore"):
        return np.log(1.0 - U) / np.log(1.0 - pd_annual)


def default_period(T: np.ndarray, freq: int, n_periods: int) -> np.ndarray:
    """Period in which default happens (1, 2, ...). n_periods + 1 means the bond survives."""
    period = np.ceil(np.minimum(np.asarray(T, dtype=float) * freq, n_periods + 1))
    return np.maximum(period, 1).astype(int)
