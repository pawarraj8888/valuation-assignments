"""The hand-in notebook is generated from the package. These tests run its cells and compare."""
import os
import shutil
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from build_notebook import build_cells, build_notebook  # noqa: E402
from cdo import BASE_DEAL, sensitivity, simulate  # noqa: E402
from cdo.analysis import B_NOTIONAL_GRID, PD_GRID  # noqa: E402

DATA_FILE = ROOT.parent / "data" / "fixed_random_numbers.csv"


def run_notebook_cells(folder: Path) -> dict:
    """Execute every code cell with folder as the working directory and return the variables."""
    namespace: dict = {}
    previous = os.getcwd()
    os.chdir(folder)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for kind, text in build_cells():
                if kind == "code":
                    exec(compile(text, "<notebook cell>", "exec"), namespace)
            plt.close("all")
    finally:
        os.chdir(previous)
    return namespace


@pytest.fixture(scope="module")
def notebook(tmp_path_factory) -> dict:
    if not DATA_FILE.exists():
        pytest.skip("fixed random numbers file not available")
    folder = tmp_path_factory.mktemp("notebook")
    shutil.copy(DATA_FILE, folder / "fixed_random_numbers.csv")
    namespace = run_notebook_cells(folder)
    namespace["_folder"] = folder
    return namespace


def test_notebook_inputs_are_the_package_base_deal(notebook):
    assert notebook["base"] == BASE_DEAL
    assert notebook["N_CASES"] == 1000


def test_notebook_cash_flows_match_the_package(notebook, normals):
    r = simulate(normals, BASE_DEAL)
    assert np.array_equal(notebook["Q"], r["Q"])
    assert np.array_equal(notebook["pool_cf"], r["pool_cf"])
    assert np.array_equal(notebook["a_cf"], r["a_cf"])
    assert np.array_equal(notebook["b_cf"], r["b_cf"])
    assert np.array_equal(notebook["eq_cf"], r["eq_cf"])


def test_notebook_sensitivities_match_the_package(notebook, normals):
    pd.testing.assert_frame_equal(notebook["sens_pd"], sensitivity(normals, BASE_DEAL, "pd", PD_GRID))
    pd.testing.assert_frame_equal(notebook["sens_b"], sensitivity(normals, BASE_DEAL, "b_notional", B_NOTIONAL_GRID))


def test_notebook_summary_table(notebook):
    summary = notebook["summary"]
    assert summary.loc["Equity (bank)", "No-default amount ($MM)"] == pytest.approx(96.0)
    assert summary.loc["Equity (bank)", "Mean ($MM)"] == pytest.approx(83.283, abs=1e-3)
    assert summary.loc["Class A", "P(below no-default amount)"] == 0
    assert summary.loc["Class B", "P(below no-default amount)"] == 0


def test_notebook_writes_nothing_but_the_random_numbers(notebook):
    assert sorted(p.name for p in notebook["_folder"].iterdir()) == ["fixed_random_numbers.csv"]


def test_notebook_recreates_the_random_numbers_when_the_file_is_missing(tmp_path, raw_normals, normals):
    namespace = run_notebook_cells(tmp_path)
    assert (tmp_path / "fixed_random_numbers.csv").exists()
    assert namespace["source"].startswith("generated")
    assert np.array_equal(namespace["Z_raw"], raw_normals)
    assert np.array_equal(namespace["Z"], normals)


def test_notebook_has_no_stored_outputs_before_execution():
    nb = build_notebook()
    assert all(not cell.get("outputs") for cell in nb.cells if cell.cell_type == "code")
    assert nb.cells[1].source.startswith("# Mini-Project 3")
