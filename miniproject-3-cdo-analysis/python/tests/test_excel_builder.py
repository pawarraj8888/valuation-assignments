"""Checks on the workbook the builder writes, before Excel ever opens it."""
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import build_excel as layout  # noqa: E402
from cdo import BASE_DEAL, N_CASES, SEED, load_fixed_normals  # noqa: E402

DATA_FILE = ROOT.parent / "data" / "fixed_random_numbers.csv"
RESULTS_FILE = ROOT.parent / "output" / "results.json"
# A reference with row 0 (B0, $B$0) or with a number where the column letter should be ($0$6).
# Excel refuses to open a workbook that contains one.
BAD_REFERENCE = re.compile(r"(?<![A-Za-z0-9_.])\$?[A-Z]{1,3}\$?0(?![0-9.])|\$\d+\$\d+")


@pytest.fixture(scope="module")
def workbook():
    if not DATA_FILE.exists() or not RESULTS_FILE.exists():
        pytest.skip("needs the fixed random numbers and output/results.json (run run_analysis.py first)")
    table, _ = load_fixed_normals(DATA_FILE, N_CASES, BASE_DEAL["n_bonds"], SEED)
    return layout.build_workbook(table, json.loads(RESULTS_FILE.read_text()), dict(BASE_DEAL), 5)


def formulas(workbook):
    for ws in workbook:
        for row in ws.iter_rows():
            for cell in row:
                if isinstance(cell.value, str) and cell.value.startswith("="):
                    yield ws.title, cell.coordinate, cell.value


def test_the_reference_check_catches_broken_references():
    assert BAD_REFERENCE.search("=Inputs!$B$25*IF($A20>=$0$6,1-lgd,1)")
    assert BAD_REFERENCE.search("=SUM(B0:B9)")
    assert not BAD_REFERENCE.search('=IF(B5=0,0,C5/B5)+COUNTIF($G$6:$G$15,"<=20")/Number_of_Cases+PERCENTILE(E10:E20,0.05)')


def test_no_formula_has_a_broken_cell_reference(workbook):
    bad = [(sheet, where, text) for sheet, where, text in formulas(workbook) if BAD_REFERENCE.search(text)]
    assert not bad, bad[:5]


def test_defaults_use_the_moment_matched_numbers(workbook):
    first = layout.FIRST_ROW
    formula = workbook["Defaults"].cell(row=first, column=layout.DEF_X_COL).value
    assert f"Random!${layout.col(layout.RAND_MATCHED_COL)}{first}" in formula


def test_the_matched_table_is_formulas_and_the_draws_are_values(workbook):
    ws = workbook["Random"]
    assert ws.cell(row=layout.FIRST_ROW, column=layout.RAND_MATCHED_COL).value.startswith("=SUMPRODUCT(")
    assert ws.cell(row=layout.FIRST_ROW, column=layout.RAND_DEMEANED_COL).value.startswith("=")
    assert isinstance(ws.cell(row=layout.FIRST_ROW, column=layout.RAND_RAW_COL).value, float)


def test_number_of_cases_is_counted_not_typed(workbook):
    assert str(workbook["Inputs"][layout.CASES_CELL].value).startswith("=COUNT(")


def test_only_the_expected_names_are_defined(workbook):
    assert set(workbook.defined_names) == set(layout.INPUT_CELLS) | {"case_number", "Number_of_Cases"}


def test_statistics_table_has_the_standard_error_column(workbook):
    ws = workbook["Statistics"]
    header = [ws.cell(row=layout.SUMMARY_FIRST_ROW - 1, column=c).value for c in range(2, 13)]
    assert header[0] == "No-default amount" and header[3] == "Std error of the mean"
    assert ws.cell(row=layout.SUMMARY_FIRST_ROW, column=5).value == f"=D{layout.SUMMARY_FIRST_ROW}/SQRT(Number_of_Cases)"
    assert not any("romised" in str(label) for label in header)


def test_moment_matching_formulas_are_the_ones_from_class(workbook):
    ws = workbook["Random"]
    assert ws["M18"].value == "=B18-B$15"
    assert ws["B4"].value == "=COVAR($B$18:$B$1017,$B$18:$B$1017)"
    assert ws["X4"].value.ref == "X4:AG13" and ws["X4"].value.text == "=MINVERSE(M4:V13)"
    assert ws["X18"].value == "=SUMPRODUCT($M18:$V18,$X$4:$AG$4)"
    assert ws["Y18"].value == "=SUMPRODUCT($M18:$V18,$X$5:$AG$5)"
