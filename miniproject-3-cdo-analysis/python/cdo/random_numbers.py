"""The fixed random numbers: 1000 cases x 10 bonds of independent standard normals."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 6103
N_CASES = 1000


def moment_match(Z: np.ndarray) -> np.ndarray:
    """De-mean the draws and rescale them so the sample has exactly the moments of independent standard normals.

    As shown in class: subtract each column's mean, take the population covariance of the draws and
    multiply the de-meaned draws by the inverse of its Cholesky factor. Afterwards every bond's
    numbers have mean 0 and variance 1 across the cases, and no two bonds' numbers are correlated.
    """
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2 or Z.shape[0] <= Z.shape[1]:
        raise ValueError(f"moment matching needs a cases x bonds table with more cases than bonds, got shape {Z.shape}")
    if not np.isfinite(Z).all():
        raise ValueError("moment matching needs finite numbers")
    demeaned = Z - Z.mean(axis=0)
    covariance = demeaned.T @ demeaned / len(Z)
    try:
        chol = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError("moment matching needs columns that are not copies or combinations of each other") from error
    return np.linalg.solve(chol, demeaned.T).T      # same as demeaned @ inverse(chol).T


def load_fixed_normals(path: Path, n_cases: int, n_bonds: int, seed: int) -> tuple:
    """Read the fixed random numbers, creating the file from the seed if it is missing.

    Returns (table, source). The values are always the ones stored in the file,
    so every run uses exactly the same numbers. Cases are numbered from 1.
    """
    path = Path(path)
    if path.exists():
        source = "read from file"
    else:
        rng = np.random.default_rng(seed)
        columns = [f"Bond {i}" for i in range(1, n_bonds + 1)]
        index = pd.RangeIndex(1, n_cases + 1, name="case")
        draws = pd.DataFrame(rng.standard_normal((n_cases, n_bonds)), columns=columns, index=index)
        path.parent.mkdir(parents=True, exist_ok=True)
        draws.to_csv(path, float_format="%.10f")
        source = f"generated with seed {seed}"
    table = pd.read_csv(path, index_col="case")
    if table.shape != (n_cases, n_bonds):
        raise ValueError(f"{path} has shape {table.shape}, expected {(n_cases, n_bonds)}")
    if not np.isfinite(table.values).all():
        raise ValueError(f"{path} contains values that are not finite numbers")
    return table, source
