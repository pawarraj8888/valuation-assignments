import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_FILE = ROOT.parent / "data" / "fixed_random_numbers.csv"


@pytest.fixture(scope="session")
def raw_normals() -> np.ndarray:
    """The 1000 x 10 fixed independent normals exactly as stored in the csv file."""
    if not DATA_FILE.exists():
        pytest.skip("fixed random numbers file not available")
    from cdo.random_numbers import N_CASES, SEED, load_fixed_normals

    table, _ = load_fixed_normals(DATA_FILE, N_CASES, 10, SEED)
    return table.values


@pytest.fixture(scope="session")
def normals(raw_normals) -> np.ndarray:
    """The moment-matched numbers: what every result in the project is computed from."""
    from cdo.random_numbers import moment_match

    return moment_match(raw_normals)
