#!/usr/bin/env python3
"""Build the Excel implementation of Miniproject 3 Part 1 (live formulas, case selector).

Layout (the way the workbook would be built by hand in Excel):
    Inputs       deal parameters (yellow cells) and the promised cash flow schedule
    Case         case selector: default times and quarterly cash flows of the chosen case
    Random       the 1000 x 10 fixed independent normals (values), then de-meaned and moment matched (formulas)
    Defaults     Cholesky factor, correlated normals, uniforms, default times and quarters
    CashFlows    pool cash flow and the waterfall for every case and quarter
    Statistics   summary statistics, default-count distribution, quarterly table, histogram
    Sensitivity  sensitivity tables from the Python run next to the workbook's live values
    Notes        the methodology

    python build_excel.py --data ../data/fixed_random_numbers.csv --results ../output/results.json \
        --out ../excel/Miniproject3_CDO_Analysis.xlsx
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter as col
from openpyxl.workbook.defined_name import DefinedName
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.worksheet.formula import ArrayFormula

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cdo import BASE_DEAL, N_CASES, PROJECT, SEED, load_fixed_normals  # noqa: E402
from write_report import assumptions, steps  # noqa: E402

# --- layout constants shared with verify_excel.py ---------------------------------
N_BONDS = BASE_DEAL["n_bonds"]
N_PERIODS = BASE_DEAL["years"] * BASE_DEAL["freq"]
FIRST_ROW = 18                              # case 1 on Random, Defaults and CashFlows
LAST_ROW = FIRST_ROW + N_CASES - 1

INPUT_CELLS = {                             # Inputs sheet: Excel name -> (cell, deal key)
    "face": ("B9", "face"), "coupon": ("B10", "coupon"), "pd_annual": ("B11", "pd"), "lgd": ("B12", "lgd"),
    "rho": ("B13", "rho"), "a_notional": ("B14", "a_notional"), "a_coupon": ("B15", "a_coupon"),
    "b_notional": ("B16", "b_notional"), "b_coupon": ("B17", "b_coupon"),
}
SCHEDULE_FIRST_ROW = 25                     # Inputs!A25:E44 quarter, bond, pool, Class A due, Class B due
CASE_CELL = "B2"                            # Case sheet, named case_number
CASE_BOND_FIRST_ROW = 6                     # Case!A6:H15 one row per bond
CASE_CF_FIRST_ROW = 20                      # Case!A20:P39 one row per quarter

RAND_RAW_COL, RAND_DEMEANED_COL, RAND_MATCHED_COL, RAND_CHECK_COL = 2, 13, 24, 35   # Random: B, M, X, AI
RAND_MATRIX_FIRST_ROW = 4                   # Random rows 4-13: covariance, its Cholesky factor, the inverse, a check
RAND_MEAN_ROW = 15                          # Random row 15: column means
CASES_CELL = "B21"                          # Inputs sheet, named Number_of_Cases
CASE_QUARTER_COL = 7                        # Case!G: default quarter of each bond
CHOL_FIRST_ROW = 4                          # Defaults!B4:K13
DEF_X_COL, DEF_U_COL, DEF_T_COL, DEF_Q_COL, DEF_COUNT_COL = 2, 13, 24, 35, 46
CF_POOL_COL, CF_A_COL, CF_B_COL, CF_EQ_COL, CF_TOTAL_COL = 2, 23, 44, 65, 86
CF_BOND_ROW, CF_A_DUE_ROW, CF_B_DUE_ROW, CF_HEADER_ROW = 14, 15, 16, 17

SUMMARY_FIRST_ROW = 5                       # Statistics!A5:K8 pool, Class A, Class B, equity
DIST_FIRST_ROW = 13                         # Statistics!A13:F23 defaults 0..10
QUARTERLY_FIRST_ROW = 28                    # Statistics!A28:K47 quarters 1..20
HIST_FIRST_ROW = 52                         # Statistics!A52:E75 histogram bins
HIST_BINS = 24
LIVE_ROW = 5                                # Sensitivity!A5:I5 current workbook values

# --- plain Excel styling ------------------------------------------------------------
HEADER_FILL = PatternFill("solid", fgColor="DDEBF7")
INPUT_FILL = PatternFill("solid", fgColor="FFF2CC")
BOLD = Font(bold=True)
TITLE = Font(bold=True, size=14)
THIN = Side(style="thin", color="BFBFBF")
BOTTOM = Border(bottom=THIN)
WRAP = Alignment(wrap_text=True, vertical="top")
CENTER_WRAP = Alignment(wrap_text=True, vertical="center", horizontal="center")
SERIES_COLORS = {"Class A": "EB6834", "Class B": "1BAF7A", "Equity": "2A78D6", "pool": "52514E", "benchmark": "898781"}
INPUT_LIMITS = {"pd_annual": (0, 0.9999), "lgd": (0, 1), "rho": (0, 0.9999)}   # anything else: 0 or more


def header(ws, row: int, labels: list, start_col: int = 1, height: float | None = None) -> None:
    for offset, label in enumerate(labels):
        cell = ws.cell(row=row, column=start_col + offset, value=label)
        cell.font = BOLD
        cell.fill = HEADER_FILL
        cell.border = BOTTOM
        cell.alignment = CENTER_WRAP
    if height:
        ws.row_dimensions[row].height = height


def widths(ws, spec: dict) -> None:
    for column, width in spec.items():
        ws.column_dimensions[column].width = width


def name(wb: Workbook, label: str, ref: str) -> None:
    wb.defined_names[label] = DefinedName(label, attr_text=ref)


def block(first_col: int, row: int | str, width: int, absolute_row: bool = False) -> str:
    """Range reference for `width` cells of one row starting at column index first_col."""
    r = f"${row}" if absolute_row else f"{row}"
    return f"${col(first_col)}{r}:${col(first_col + width - 1)}{r}"


def column_range(column: int) -> str:
    """All 1000 cases of one column, e.g. $CH$18:$CH$1017."""
    return f"${col(column)}${FIRST_ROW}:${col(column)}${LAST_ROW}"


# ----------------------------------------------------------------------------- sheets
def build_inputs(wb: Workbook, deal: dict, results: dict) -> None:
    ws = wb.active
    ws.title = "Inputs"
    ws["A1"] = "Miniproject 3 (Part 1) - Simplified CDO analysis"
    ws["A1"].font = TITLE
    ws["A2"] = f"{PROJECT['course']} | {PROJECT['authors']}"
    ws["A3"] = ("Yellow cells are inputs. Everything else is a formula, except the stored random numbers and the Sensitivity tables. "
                "All amounts are in $ millions. "
                "The case to display is chosen on the Case sheet.")
    header(ws, 5, ["Parameter", "Value", "Note"])
    fixed = [
        (6, "Number of bonds", deal["n_bonds"], "Fixed by the layout of the sheets (10 columns per block)."),
        (7, "Years to maturity", deal["years"], "Fixed by the layout (20 quarterly columns)."),
        (8, "Payments per year", deal["freq"], "Quarterly."),
    ]
    for row, label, value, note in fixed:
        ws.cell(row=row, column=1, value=label)
        ws.cell(row=row, column=2, value=value)
        ws.cell(row=row, column=3, value=note)
    inputs = [
        ("face", "Face value per bond ($MM)", "0.00", "Given."),
        ("coupon", "Coupon rate, annual", "0.00%", "Given. Paid quarterly: coupon / 4 x face."),
        ("pd_annual", "Default probability, annual (pi)", "0.00%", "Given. Default time in years t = ln(1 - u) / ln(1 - pi)."),
        ("lgd", "Loss given default (lambda)", "0.00%", "Given. BIS model: 1 - LGD of every promised payment is always paid."),
        ("rho", "Correlation between default dates", "0.00", "Given. Applied to the normals through the Cholesky factor."),
        ("a_notional", "Class A notional ($MM)", "0.00", "Given."),
        ("a_coupon", "Class A coupon, annual", "0.00%", "Given. Paid quarterly, principal at maturity."),
        ("b_notional", "Class B notional ($MM)", "0.00", "Given."),
        ("b_coupon", "Class B coupon, annual", "0.00%", "Given. Paid quarterly, principal at maturity."),
    ]
    for label, text, fmt, note in inputs:
        ref, key = INPUT_CELLS[label]
        row = int(ref[1:])
        ws.cell(row=row, column=1, value=text)
        ws[ref] = deal[key]
        ws[ref].fill = INPUT_FILL
        ws[ref].number_format = fmt
        ws.cell(row=row, column=3, value=note)
        name(wb, label, f"Inputs!${ref[0]}${row}")
        low, high = INPUT_LIMITS.get(label, (0, 1000000))
        rule = DataValidation(type="decimal", operator="between", formula1=str(low), formula2=str(high),
                              showErrorMessage=True, errorTitle=text, error=f"Enter a number from {low} to {high}.")
        ws.add_data_validation(rule)
        rule.add(ref)
    ws["A18"], ws["B18"], ws["C18"] = "Market YTM on the bonds", results["market_ytm"], "Given. Not used until Part 2."
    ws["A19"], ws["B19"], ws["C19"] = "Risk-free rate", results["risk_free"], "Given. Not used until Part 2."
    ws["B18"].number_format = ws["B19"].number_format = "0.00%"
    ws["A20"], ws["B20"] = "Seed of the fixed random numbers", results["seed"]
    ws["A21"], ws[CASES_CELL] = "Number of cases", f"=COUNT(Random!$A${FIRST_ROW}:$A${LAST_ROW})"
    ws["C21"] = "Given. Counted from the Random sheet: the layout has one row per case, so this is not an input."
    name(wb, "Number_of_Cases", f"Inputs!${CASES_CELL[0]}${CASES_CELL[1:]}")

    ws["A23"] = "Promised cash flows ($MM)"
    ws["A23"].font = BOLD
    header(ws, 24, ["Quarter", "One bond", "Pool (10 bonds)", "Due to Class A", "Due to Class B"])
    for q in range(1, N_PERIODS + 1):
        r = SCHEDULE_FIRST_ROW + q - 1
        ws[f"A{r}"] = q
        ws[f"B{r}"] = f"=face*coupon/{deal['freq']}+IF(A{r}={N_PERIODS},face,0)"
        ws[f"C{r}"] = f"={N_BONDS}*B{r}"
        ws[f"D{r}"] = f"=a_notional*a_coupon/{deal['freq']}+IF(A{r}={N_PERIODS},a_notional,0)"
        ws[f"E{r}"] = f"=b_notional*b_coupon/{deal['freq']}+IF(A{r}={N_PERIODS},b_notional,0)"
        for letter in "BCDE":
            ws[f"{letter}{r}"].number_format = "0.0000"
    total = SCHEDULE_FIRST_ROW + N_PERIODS
    ws[f"A{total}"] = "Total"
    for letter in "BCDE":
        ws[f"{letter}{total}"] = f"=SUM({letter}{SCHEDULE_FIRST_ROW}:{letter}{total - 1})"
        ws[f"{letter}{total}"].number_format = "0.00"
        ws[f"{letter}{total}"].font = BOLD
    widths(ws, {"A": 36, "B": 14, "C": 16, "D": 16, "E": 16})


def build_random(wb: Workbook, table) -> None:
    ws = wb.create_sheet("Random")
    ws["A1"] = "Fixed random numbers and moment matching"
    ws["A1"].font = TITLE
    ws["A2"] = (f"{N_CASES} cases x {N_BONDS} bonds of independent standard normals, generated once in Python "
                f"(numpy default_rng, seed {SEED}) and stored as values, so every recalculation uses the same numbers. "
                "They are then de-meaned and moment matched: multiplied by the inverse of the Cholesky factor of their "
                "covariance, which gives numbers with means of exactly 0, variances of exactly 1 and no correlation.")
    first, last = RAND_MATRIX_FIRST_ROW, RAND_MATRIX_FIRST_ROW + N_BONDS - 1
    raw, demeaned, matched, check = RAND_RAW_COL, RAND_DEMEANED_COL, RAND_MATCHED_COL, RAND_CHECK_COL
    labels = [(raw, "Covariance of the initial numbers (population)"), (demeaned, "Cholesky factor L of that covariance"),
              (matched, "Inverse of L"), (check, "Check: covariance of the moment-matched numbers (identity)")]
    for start, label in labels:
        ws.cell(row=first - 1, column=start, value=label).font = BOLD
    for i in range(1, N_BONDS + 1):
        r = first + i - 1
        for j in range(1, N_BONDS + 1):
            ws.cell(row=r, column=raw + j - 1, value=f"=COVAR({column_range(raw + i - 1)},{column_range(raw + j - 1)})")
            ws.cell(row=r, column=demeaned + j - 1, value=general_cholesky_formula(i, j, raw, demeaned, first))
            ws.cell(row=r, column=check + j - 1,
                    value=f"=COVAR({column_range(matched + i - 1)},{column_range(matched + j - 1)})")
            for start in (raw, demeaned, matched, check):
                ws.cell(row=r, column=start + j - 1).number_format = "0.000000"
    inverse = f"{col(matched)}{first}:{col(matched + N_BONDS - 1)}{last}"
    ws[f"{col(matched)}{first}"] = ArrayFormula(inverse, f"=MINVERSE({col(demeaned)}{first}:{col(demeaned + N_BONDS - 1)}{last})")

    ws.cell(row=RAND_MEAN_ROW, column=1, value="Mean").font = BOLD
    titles = [(raw, "Initial random normals"), (demeaned, "De-meaned (initial number minus its column mean)"),
              (matched, "Moment matched: each case = inverse of L x its de-meaned numbers (what the model uses)")]
    for start, label in titles:
        ws.cell(row=FIRST_ROW - 2, column=start, value=label).font = BOLD
        header(ws, FIRST_ROW - 1, list(table.columns), start_col=start)
        for j in range(N_BONDS):
            ws.cell(row=RAND_MEAN_ROW, column=start + j, value=f"=AVERAGE({column_range(start + j)})").number_format = "0.000000"
    header(ws, FIRST_ROW - 1, ["Case"])
    for i, values in enumerate(table.values):
        r = FIRST_ROW + i
        ws.cell(row=r, column=1, value=i + 1)
        for j, value in enumerate(values):
            ws.cell(row=r, column=raw + j, value=float(value)).number_format = "0.0000000000"
            ws.cell(row=r, column=demeaned + j, value=f"={col(raw + j)}{r}-{col(raw + j)}${RAND_MEAN_ROW}")
            ws.cell(row=r, column=matched + j,
                    value=f"=SUMPRODUCT({block(demeaned, r, N_BONDS)},{block(matched, first + j, N_BONDS, absolute_row=True)})")
    ws.freeze_panes = f"B{FIRST_ROW}"
    widths(ws, {"A": 8, **{col(c): 14 for c in range(2, check + N_BONDS)}})


def general_cholesky_formula(i: int, j: int, cov_col: int, chol_col: int, first_row: int) -> str | int:
    """Cell formula for entry (i, j), counted from 1, of the lower Cholesky factor of the matrix stored at cov_col."""
    row_i, row_j = first_row + i - 1, first_row + j - 1
    if j > i:
        return 0
    target = f"{col(cov_col + j - 1)}{row_i}"
    if j == 1:
        return f"=SQRT({target})" if i == 1 else f"={target}/${col(chol_col)}${first_row}"
    before_i = f"${col(chol_col)}{row_i}:{col(chol_col + j - 2)}{row_i}"
    if j == i:
        return f"=SQRT({target}-SUMSQ({before_i}))"
    before_j = f"${col(chol_col)}{row_j}:{col(chol_col + j - 2)}{row_j}"
    return f"=({target}-SUMPRODUCT({before_i},{before_j}))/{col(chol_col + j - 1)}{row_j}"


def cholesky_formula(i: int, j: int) -> str | int:
    """Cell formula for entry (i, j) of the lower Cholesky factor of the equicorrelation matrix."""
    row_i, row_j = CHOL_FIRST_ROW + i - 1, CHOL_FIRST_ROW + j - 1
    if j > i:
        return 0
    if i == 1:
        return 1
    if j == 1:
        return f"=rho/$B${CHOL_FIRST_ROW}"
    before = f"$B{row_i}:{col(j)}{row_i}"
    if j == i:
        return f"=SQRT(1-SUMSQ({before}))"
    return f"=(rho-SUMPRODUCT({before},$B{row_j}:{col(j)}{row_j}))/{col(j + 1)}{row_j}"


def build_defaults(wb: Workbook) -> None:
    ws = wb.create_sheet("Defaults")
    ws["A1"] = "Correlated default times"
    ws["A1"].font = TITLE
    ws["A2"] = ("Cholesky factor L of the 10 x 10 matrix with 1 on the diagonal and rho elsewhere. Each case: x = L m, "
                "with m the moment-matched numbers from the Random sheet, u = N(x), t = ln(1 - u) / ln(1 - pi) in years, "
                "default quarter = roundup(4 t); 21 means no default.")
    header(ws, CHOL_FIRST_ROW - 1, ["L"] + [f"col {j}" for j in range(1, N_BONDS + 1)])
    for i in range(1, N_BONDS + 1):
        ws.cell(row=CHOL_FIRST_ROW + i - 1, column=1, value=f"Bond {i}").font = BOLD
        for j in range(1, N_BONDS + 1):
            ws.cell(row=CHOL_FIRST_ROW + i - 1, column=1 + j, value=cholesky_formula(i, j)).number_format = "0.000000"

    groups = [(DEF_X_COL, "Correlated normal x"), (DEF_U_COL, "Uniform u = N(x)"),
              (DEF_T_COL, "Default time t (years)"), (DEF_Q_COL, "Default quarter (21 = none)")]
    for first, label in groups:
        ws.cell(row=FIRST_ROW - 2, column=first, value=label).font = BOLD
        header(ws, FIRST_ROW - 1, [f"Bond {i}" for i in range(1, N_BONDS + 1)], start_col=first)
    header(ws, FIRST_ROW - 1, ["Case"])
    header(ws, FIRST_ROW - 1, ["Defaults in 5 years"], start_col=DEF_COUNT_COL, height=30)

    for r in range(FIRST_ROW, LAST_ROW + 1):
        ws.cell(row=r, column=1, value=r - FIRST_ROW + 1)
        for i in range(N_BONDS):
            x, u, t = col(DEF_X_COL + i), col(DEF_U_COL + i), col(DEF_T_COL + i)
            chol_row = CHOL_FIRST_ROW + i
            ws.cell(row=r, column=DEF_X_COL + i, value=f"=SUMPRODUCT(Random!{block(RAND_MATCHED_COL, r, N_BONDS)},$B${chol_row}:$K${chol_row})")
            ws.cell(row=r, column=DEF_U_COL + i, value=f"=NORMSDIST({x}{r})")
            ws.cell(row=r, column=DEF_T_COL + i, value=f'=IF(pd_annual=0,"never",LN(1-{u}{r})/LN(1-pd_annual))')
            ws.cell(row=r, column=DEF_Q_COL + i,
                    value=f"=IF(pd_annual=0,{N_PERIODS + 1},MIN({N_PERIODS + 1},MAX(1,ROUNDUP(4*{t}{r},0))))")
        ws.cell(row=r, column=DEF_COUNT_COL, value=f'=COUNTIF({block(DEF_Q_COL, r, N_BONDS)},"<={N_PERIODS}")')
    ws.freeze_panes = f"B{FIRST_ROW}"
    ws.column_dimensions["A"].width = 9


def build_cash_flows(wb: Workbook) -> None:
    ws = wb.create_sheet("CashFlows")
    ws["A1"] = "Pool cash flows and the waterfall, every case and quarter ($MM)"
    ws["A1"].font = TITLE
    ws["A2"] = ("Pool = promised per bond x (10 - LGD x number of bonds in default by that quarter), the BIS model summed "
                "over the 10 bonds. Class A = min(pool, due to A); Class B = min(pool - A, due to B); equity = the rest. "
                "No carry-forwards: each quarter stands alone.")
    for row, label, source in ((CF_BOND_ROW, "One bond promised", "B"), (CF_A_DUE_ROW, "Due to Class A", "D"),
                               (CF_B_DUE_ROW, "Due to Class B", "E")):
        ws.cell(row=row, column=1, value=label).font = BOLD
        for q in range(N_PERIODS):
            ws.cell(row=row, column=CF_POOL_COL + q, value=f"=Inputs!${source}${SCHEDULE_FIRST_ROW + q}")
    header(ws, CF_HEADER_ROW, ["Case"])
    for first, label in ((CF_POOL_COL, "Pool"), (CF_A_COL, "Class A"), (CF_B_COL, "Class B"), (CF_EQ_COL, "Equity")):
        ws.cell(row=CF_HEADER_ROW - 5, column=first, value=f"{label}, quarter:").font = BOLD
        header(ws, CF_HEADER_ROW, list(range(1, N_PERIODS + 1)), start_col=first)
    header(ws, CF_HEADER_ROW, ["Pool total", "Class A total", "Class B total", "Equity total", "A short (1/0)",
                               "B short (1/0)"], start_col=CF_TOTAL_COL, height=30)

    due_a = f"SUM({block(CF_POOL_COL, CF_A_DUE_ROW, N_PERIODS, absolute_row=True)})"
    due_b = f"SUM({block(CF_POOL_COL, CF_B_DUE_ROW, N_PERIODS, absolute_row=True)})"
    for r in range(FIRST_ROW, LAST_ROW + 1):
        ws.cell(row=r, column=1, value=r - FIRST_ROW + 1)
        quarters_in_default = f"Defaults!{block(DEF_Q_COL, r, N_BONDS)}"
        for q in range(N_PERIODS):
            pool, a, b = col(CF_POOL_COL + q), col(CF_A_COL + q), col(CF_B_COL + q)
            ws.cell(row=r, column=CF_POOL_COL + q,
                    value=f'={pool}${CF_BOND_ROW}*({N_BONDS}-lgd*COUNTIF({quarters_in_default},"<="&{pool}${CF_HEADER_ROW}))')
            ws.cell(row=r, column=CF_A_COL + q, value=f"=MIN({pool}{r},{pool}${CF_A_DUE_ROW})")
            ws.cell(row=r, column=CF_B_COL + q, value=f"=MIN({pool}{r}-{a}{r},{pool}${CF_B_DUE_ROW})")
            ws.cell(row=r, column=CF_EQ_COL + q, value=f"={pool}{r}-{a}{r}-{b}{r}")
        for k, first in enumerate((CF_POOL_COL, CF_A_COL, CF_B_COL, CF_EQ_COL)):
            ws.cell(row=r, column=CF_TOTAL_COL + k, value=f"=SUM({block(first, r, N_PERIODS)})")
        a_total, b_total = col(CF_TOTAL_COL + 1), col(CF_TOTAL_COL + 2)
        ws.cell(row=r, column=CF_TOTAL_COL + 4, value=f"=IF({a_total}{r}<{due_a}-0.000000001,1,0)")
        ws.cell(row=r, column=CF_TOTAL_COL + 5, value=f"=IF({b_total}{r}<{due_b}-0.000000001,1,0)")
    ws.freeze_panes = f"B{FIRST_ROW}"
    ws.column_dimensions["A"].width = 18


def build_case(wb: Workbook, example_case: int) -> None:
    ws = wb.create_sheet("Case", 1)
    ws["A1"] = "Case viewer"
    ws["A1"].font = TITLE
    ws["A2"] = "Case to display (1 to 1000)"
    ws[CASE_CELL] = example_case
    ws[CASE_CELL].fill = INPUT_FILL
    ws[CASE_CELL].font = BOLD
    name(wb, "case_number", "Case!$B$2")
    check = DataValidation(type="whole", operator="between", formula1="1", formula2=str(N_CASES), allow_blank=False,
                           showErrorMessage=True, errorTitle="Case", error=f"Enter a whole number from 1 to {N_CASES}.")
    ws.add_data_validation(check)
    check.add(CASE_CELL)
    ws["C2"] = "Type any case number in the yellow cell. Everything on this sheet follows it."

    first, last = CASE_BOND_FIRST_ROW, CASE_BOND_FIRST_ROW + N_BONDS - 1
    cf_first, cf_last = CASE_CF_FIRST_ROW, CASE_CF_FIRST_ROW + N_PERIODS - 1
    header(ws, first - 1, ["Bond", "Random normal z", "Moment-matched z", "Correlated normal x", "Uniform u",
                           "Default time (years)", "Default quarter (21 = none)", "Status", "Total cash received ($MM)"],
           height=32)
    lookups = [("Random", RAND_RAW_COL, "0.0000"), ("Random", RAND_MATCHED_COL, "0.0000"), ("Defaults", DEF_X_COL, "0.0000"),
               ("Defaults", DEF_U_COL, "0.0000"), ("Defaults", DEF_T_COL, "0.00"), ("Defaults", DEF_Q_COL, "0")]
    quarter = col(CASE_QUARTER_COL)      # column letter of the default quarter
    for i in range(N_BONDS):
        r = first + i
        ws.cell(row=r, column=1, value=f"Bond {i + 1}")
        for k, (sheet, start, fmt) in enumerate(lookups):
            source = f"{sheet}!${col(start)}${FIRST_ROW}:${col(start + N_BONDS - 1)}${LAST_ROW}"
            ws.cell(row=r, column=2 + k, value=f"=INDEX({source},case_number,{i + 1})").number_format = fmt
        ws.cell(row=r, column=CASE_QUARTER_COL + 1, value=f'=IF({quarter}{r}<={N_PERIODS},"defaults in quarter "&{quarter}{r},"survives")')
        ws.cell(row=r, column=CASE_QUARTER_COL + 2,
                value=f"=SUM({col(2 + i)}{cf_first}:{col(2 + i)}{cf_last})").number_format = "0.00"
    ws[f"A{last + 1}"] = "Bonds defaulting within 5 years"
    ws[f"{quarter}{last + 1}"] = f'=COUNTIF({quarter}{first}:{quarter}{last},"<={N_PERIODS}")'
    ws[f"A{last + 1}"].font = ws[f"{quarter}{last + 1}"].font = BOLD

    labels = ["Quarter"] + [f"Bond {i}" for i in range(1, N_BONDS + 1)] + ["Pool", "Class A", "Class B", "Equity",
                                                                           "Pool no-default amount"]
    header(ws, cf_first - 1, labels)
    for q in range(N_PERIODS):
        r, schedule = cf_first + q, SCHEDULE_FIRST_ROW + q
        ws.cell(row=r, column=1, value=q + 1)
        for i in range(N_BONDS):
            ws.cell(row=r, column=2 + i, value=f"=Inputs!$B${schedule}*IF($A{r}>=${quarter}${first + i},1-lgd,1)")
        ws[f"L{r}"] = f"=SUM(B{r}:K{r})"
        ws[f"M{r}"] = f"=MIN(L{r},Inputs!$D${schedule})"
        ws[f"N{r}"] = f"=MIN(L{r}-M{r},Inputs!$E${schedule})"
        ws[f"O{r}"] = f"=L{r}-M{r}-N{r}"
        ws[f"P{r}"] = f"=Inputs!$C${schedule}"
        for c in range(2, 17):
            ws.cell(row=r, column=c).number_format = "0.0000"
    total = cf_last + 1
    ws[f"A{total}"] = "Total"
    for c in range(2, 17):
        cell = ws.cell(row=total, column=c, value=f"=SUM({col(c)}{cf_first}:{col(c)}{cf_last})")
        cell.number_format, cell.font = "0.00", BOLD
    ws[f"A{total + 2}"] = "Check: pool total for this case on the CashFlows sheet minus the total above (should be 0)"
    ws[f"L{total + 2}"] = f"=ROUND(INDEX(CashFlows!{column_range(CF_TOTAL_COL)},case_number)-L{total},9)"

    chart = BarChart()
    chart.type, chart.grouping, chart.overlap = "col", "stacked", 100
    chart.title = "Who receives the pool's cash, quarters 1 to 19 ($MM)"
    chart.y_axis.title, chart.x_axis.title = "$MM", "Quarter"
    chart.add_data(Reference(ws, min_col=13, max_col=15, min_row=cf_first - 1, max_row=cf_last - 1), titles_from_data=True)
    chart.set_categories(Reference(ws, min_col=1, min_row=cf_first, max_row=cf_last - 1))
    for series, label in zip(chart.series, ("Class A", "Class B", "Equity")):
        series.graphicalProperties.solidFill = SERIES_COLORS[label]
    chart.height, chart.width = 8.5, 20
    ws.add_chart(chart, f"R{first - 1}")
    widths(ws, {"A": 27, "B": 15, "C": 16, "D": 16, "E": 12, "F": 13, "G": 14, "H": 22, "I": 15})


def build_statistics(wb: Workbook) -> None:
    ws = wb.create_sheet("Statistics")
    ws["A1"] = "Statistics across the 1000 cases ($MM, undiscounted)"
    ws["A1"].font = TITLE
    ws["A3"] = "Total cash received over the 5 years"
    ws["A3"].font = BOLD
    header(ws, SUMMARY_FIRST_ROW - 1, ["", "No-default amount", "Mean", "Std dev", "Std error of the mean", "Min", "5th pct",
                                       "Median", "95th pct", "Max", "Mean / no-default amount",
                                       "Share of cases below the no-default amount"], height=58)
    ws[f"A{SUMMARY_FIRST_ROW + 4}"] = ("Std error of the mean = std dev / square root of the number of cases, the usual formula for "
                                        "independent cases. Moment matching ties the cases together, so the true sampling error of "
                                        "the pool and equity means is smaller: read this column as an upper bound.")
    ws.merge_cells(start_row=SUMMARY_FIRST_ROW + 4, start_column=1, end_row=SUMMARY_FIRST_ROW + 4, end_column=12)
    ws[f"A{SUMMARY_FIRST_ROW + 4}"].alignment = WRAP
    ws.row_dimensions[SUMMARY_FIRST_ROW + 4].height = 32
    schedule = lambda letter: f"SUM(Inputs!${letter}${SCHEDULE_FIRST_ROW}:${letter}${SCHEDULE_FIRST_ROW + N_PERIODS - 1})"  # noqa: E731
    promised = [f"={schedule('C')}", f"={schedule('D')}", f"={schedule('E')}",
                f"=B{SUMMARY_FIRST_ROW}-B{SUMMARY_FIRST_ROW + 1}-B{SUMMARY_FIRST_ROW + 2}"]
    for k, label in enumerate(("Collateral pool", "Class A", "Class B", "Equity (bank)")):
        r, data = SUMMARY_FIRST_ROW + k, f"CashFlows!{column_range(CF_TOTAL_COL + k)}"
        ws[f"A{r}"] = label
        ws[f"B{r}"] = promised[k]
        formulas = [f"=AVERAGE({data})", f"=STDEV({data})", f"=D{r}/SQRT(Number_of_Cases)", f"=MIN({data})",
                    f"=PERCENTILE({data},0.05)", f"=MEDIAN({data})", f"=PERCENTILE({data},0.95)", f"=MAX({data})",
                    f"=IF(B{r}=0,0,C{r}/B{r})", f'=COUNTIF({data},"<"&(B{r}-0.000000001))/Number_of_Cases']
        for offset, formula in enumerate(formulas):
            ws.cell(row=r, column=3 + offset, value=formula)
        for c in range(2, 11):
            ws.cell(row=r, column=c).number_format = "0.000"
        ws[f"K{r}"].number_format = ws[f"L{r}"].number_format = "0.0%"

    counts = f"Defaults!{column_range(DEF_COUNT_COL)}"
    ws[f"A{DIST_FIRST_ROW - 2}"] = "Number of bonds defaulting within 5 years"
    ws[f"A{DIST_FIRST_ROW - 2}"].font = BOLD
    header(ws, DIST_FIRST_ROW - 1, ["Defaults", "Cases", "Simulated", "If independent (binomial)", "Avg pool cash",
                                    "Avg equity cash"], height=32)
    for k in range(N_BONDS + 1):
        r = DIST_FIRST_ROW + k
        ws[f"A{r}"] = k
        ws[f"B{r}"] = f"=COUNTIF({counts},A{r})"
        ws[f"C{r}"] = f"=B{r}/Number_of_Cases"
        ws[f"D{r}"] = f"=BINOMDIST(A{r},{N_BONDS},1-(1-pd_annual)^{BASE_DEAL['years']},FALSE)"
        ws[f"E{r}"] = f'=IF(B{r}=0,"",AVERAGEIF({counts},A{r},CashFlows!{column_range(CF_TOTAL_COL)}))'
        ws[f"F{r}"] = f'=IF(B{r}=0,"",AVERAGEIF({counts},A{r},CashFlows!{column_range(CF_TOTAL_COL + 3)}))'
        ws[f"C{r}"].number_format = ws[f"D{r}"].number_format = "0.00%"
        ws[f"E{r}"].number_format = ws[f"F{r}"].number_format = "0.00"

    ws[f"A{QUARTERLY_FIRST_ROW - 2}"] = "Quarterly cash flows"
    ws[f"A{QUARTERLY_FIRST_ROW - 2}"].font = BOLD
    header(ws, QUARTERLY_FIRST_ROW - 1, ["Quarter", "Pool no-default amount", "Pool expected (exact)", "Pool mean", "Pool 5th pct",
                                         "Pool 95th pct", "Class A mean", "Class B mean", "Equity mean",
                                         "Equity 5th pct", "Equity 95th pct"], height=32)
    for q in range(N_PERIODS):
        r = QUARTERLY_FIRST_ROW + q
        pool, a, b, eq = (f"CashFlows!{column_range(first + q)}" for first in (CF_POOL_COL, CF_A_COL, CF_B_COL, CF_EQ_COL))
        ws[f"A{r}"] = q + 1
        row = [f"=Inputs!$C${SCHEDULE_FIRST_ROW + q}",
               f"=B{r}*((1-lgd)+lgd*(1-pd_annual)^(A{r}/{BASE_DEAL['freq']}))",
               f"=AVERAGE({pool})", f"=PERCENTILE({pool},0.05)", f"=PERCENTILE({pool},0.95)", f"=AVERAGE({a})",
               f"=AVERAGE({b})", f"=AVERAGE({eq})", f"=PERCENTILE({eq},0.05)", f"=PERCENTILE({eq},0.95)"]
        for offset, formula in enumerate(row):
            ws.cell(row=r, column=2 + offset, value=formula).number_format = "0.0000"

    build_histogram(ws)
    add_statistics_charts(ws)
    widths(ws, {"A": 18, **{col(c): 14 for c in range(2, 13)}})


def build_histogram(ws) -> None:
    """Histogram of total cash as a share of what the pool promised (pool) and of the no-default amount (equity)."""
    ws[f"A{HIST_FIRST_ROW - 2}"] = "Distribution of total 5-year cash (bins are a share of the no-default amount)"
    ws[f"A{HIST_FIRST_ROW - 2}"].font = BOLD
    header(ws, HIST_FIRST_ROW - 1, ["Above", "Up to", "Label", "Pool cases", "Equity cases"])
    low = 1.0 - 0.025 * HIST_BINS
    pool, equity = f"CashFlows!{column_range(CF_TOTAL_COL)}", f"CashFlows!{column_range(CF_TOTAL_COL + 3)}"
    for k in range(HIST_BINS):
        r = HIST_FIRST_ROW + k
        ws[f"A{r}"], ws[f"B{r}"] = low + 0.025 * k, low + 0.025 * (k + 1)
        ws[f"C{r}"] = f'="up to "&TEXT(B{r},"0.0%")' if k == 0 else f'=TEXT(B{r},"0.0%")'
        for letter, data, promised in (("D", pool, f"$B${SUMMARY_FIRST_ROW}"), ("E", equity, f"$B${SUMMARY_FIRST_ROW + 3}")):
            lower = "" if k == 0 else f'{data},">"&({promised}*$A{r}+0.000000001),'
            ws[f"{letter}{r}"] = f'=COUNTIFS({lower}{data},"<="&({promised}*$B{r}+0.000000001))'
        ws[f"A{r}"].number_format = ws[f"B{r}"].number_format = "0.0%"


def add_statistics_charts(ws) -> None:
    defaults = BarChart()
    defaults.type, defaults.title = "col", "Number of defaults: simulated vs independent"
    defaults.y_axis.title, defaults.x_axis.title = "Share of cases", "Bonds defaulting within 5 years"
    defaults.add_data(Reference(ws, min_col=3, max_col=4, min_row=DIST_FIRST_ROW - 1, max_row=DIST_FIRST_ROW + N_BONDS),
                      titles_from_data=True)
    defaults.set_categories(Reference(ws, min_col=1, min_row=DIST_FIRST_ROW, max_row=DIST_FIRST_ROW + N_BONDS))
    defaults.series[0].graphicalProperties.solidFill = SERIES_COLORS["pool"]
    defaults.series[1].graphicalProperties.solidFill = SERIES_COLORS["benchmark"]
    defaults.height, defaults.width = 8.5, 16
    ws.add_chart(defaults, f"M{SUMMARY_FIRST_ROW - 1}")

    histogram = BarChart()
    histogram.type, histogram.title = "col", "Total 5-year cash as a share of the no-default amount"
    histogram.y_axis.title, histogram.x_axis.title = "Number of cases", "Upper edge of the bin"
    histogram.add_data(Reference(ws, min_col=4, max_col=5, min_row=HIST_FIRST_ROW - 1, max_row=HIST_FIRST_ROW + HIST_BINS - 1),
                       titles_from_data=True)
    histogram.set_categories(Reference(ws, min_col=3, min_row=HIST_FIRST_ROW, max_row=HIST_FIRST_ROW + HIST_BINS - 1))
    histogram.series[0].graphicalProperties.solidFill = SERIES_COLORS["pool"]
    histogram.series[1].graphicalProperties.solidFill = SERIES_COLORS["Equity"]
    histogram.height, histogram.width = 8.5, 16
    ws.add_chart(histogram, f"M{QUARTERLY_FIRST_ROW - 1}")


def build_sensitivity(wb: Workbook, results: dict) -> None:
    ws = wb.create_sheet("Sensitivity")
    ws["A1"] = "Sensitivities"
    ws["A1"].font = TITLE
    ws["A2"] = ("The tables are values from the Python run on the same random numbers. To reproduce any row, type its "
                "input on the Inputs sheet: the 'This workbook now' row recalculates and should match it.")
    columns = ["avg defaults", "P(no default)", "equity mean", "equity std", "equity 5th pct", "equity min",
               "P(A shortfall)", "P(B shortfall)"]
    formats = ["0.000", "0.0%", "0.000", "0.000", "0.000", "0.000", "0.0%", "0.0%"]
    header(ws, LIVE_ROW - 1, ["", "Value"] + columns, height=32)
    counts, equity_row = f"Defaults!{column_range(DEF_COUNT_COL)}", SUMMARY_FIRST_ROW + 3
    live = [f"=AVERAGE({counts})", f"=COUNTIF({counts},0)/Number_of_Cases", f"=Statistics!C{equity_row}",
            f"=Statistics!D{equity_row}", f"=Statistics!G{equity_row}", f"=Statistics!F{equity_row}",
            f"=SUM(CashFlows!{column_range(CF_TOTAL_COL + 4)})/Number_of_Cases",
            f"=SUM(CashFlows!{column_range(CF_TOTAL_COL + 5)})/Number_of_Cases"]
    ws[f"A{LIVE_ROW}"] = "This workbook now"
    ws[f"A{LIVE_ROW}"].font = BOLD
    for offset, (formula, fmt) in enumerate(zip(live, formats)):
        ws.cell(row=LIVE_ROW, column=3 + offset, value=formula).number_format = fmt

    tables = [("Annual default probability", "pd", results["sensitivities"]["pd"], "0.0%"),
              ("Loss given default", "lgd", results["sensitivities"]["lgd"], "0.0%"),
              ("Correlation", "rho", results["sensitivities"]["rho"], "0.00"),
              ("Class B notional ($MM)", "b_notional", results["sensitivities"]["b_notional"], "0.00"),
              ("LGD 100%: annual default probability", "pd", results["stress"]["pd"], "0.0%"),
              ("LGD 100%: correlation", "rho", results["stress"]["rho"], "0.00")]
    r = LIVE_ROW + 2
    for title, key, rows, fmt in tables:
        ws.cell(row=r, column=1, value=title).font = BOLD
        header(ws, r + 1, ["", "Value"] + columns, height=32)
        for k, record in enumerate(rows):
            ws.cell(row=r + 2 + k, column=2, value=record[key]).number_format = fmt
            for offset, (column, number_format) in enumerate(zip(columns, formats)):
                ws.cell(row=r + 2 + k, column=3 + offset, value=record[column]).number_format = number_format
        r += len(rows) + 4
    widths(ws, {"A": 36, **{col(c): 13 for c in range(2, 11)}})


def build_notes(wb: Workbook, results: dict) -> None:
    ws = wb.create_sheet("Notes")
    ws["A1"] = "Methodology"
    ws["A1"].font = TITLE
    ws["A2"] = f"{PROJECT['assignment']}: {PROJECT['title']}. {PROJECT['authors']}. Project page: {PROJECT['site_url']}"
    row = 4
    for title, entries in (("Key assumptions", assumptions(results)), ("Implementation steps", steps(results))):
        ws.cell(row=row, column=1, value=title).font = BOLD
        row += 1
        for heading, text in entries:
            ws.cell(row=row, column=1, value=heading).font = BOLD
            ws.cell(row=row, column=2, value=text).alignment = WRAP
            ws.row_dimensions[row].height = 16 * max(3, -(-len(text) // 120))
            row += 1
        row += 1
    widths(ws, {"A": 26, "B": 130})


def build_workbook(table, results: dict, deal: dict, case: int) -> Workbook:
    wb = Workbook()
    build_inputs(wb, deal, results)
    build_random(wb, table)
    build_defaults(wb)
    build_cash_flows(wb)
    build_case(wb, case)
    build_statistics(wb)
    build_sensitivity(wb, results)
    build_notes(wb, results)
    wb.properties.title = f"{PROJECT['assignment']} - {PROJECT['title']}"
    wb.properties.creator = PROJECT["authors"]
    return wb


def parse_overrides(pairs: list) -> dict:
    """--set lgd=1.0 pd=0.12 style overrides of the base deal (used to test the formulas)."""
    deal = dict(BASE_DEAL)
    allowed = sorted(key for _, key in INPUT_CELLS.values())   # the layout fixes n_bonds, years and freq
    for pair in pairs:
        key, _, value = pair.partition("=")
        if key not in allowed or not value:
            raise SystemExit(f"--set expects key=value with one of {allowed}, got {pair!r}")
        deal[key] = float(value)
    return deal


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="../data/fixed_random_numbers.csv")
    parser.add_argument("--results", default="../output/results.json")
    parser.add_argument("--out", default="../excel/Miniproject3_CDO_Analysis.xlsx")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override inputs, e.g. --set lgd=1 pd=0.12")
    parser.add_argument("--case", type=int, default=None, help="case shown on the Case sheet (default: the report's example)")
    args = parser.parse_args(argv)

    table, _ = load_fixed_normals(Path(args.data), N_CASES, N_BONDS, SEED)
    results = json.loads(Path(args.results).read_text())
    case = results["example"]["case"] if args.case is None else args.case
    if not 1 <= case <= N_CASES:
        raise SystemExit(f"--case must be between 1 and {N_CASES}, got {case}")
    wb = build_workbook(table, results, parse_overrides(args.set), case)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    print(f"Workbook written to {out.resolve()} (recalculate in Excel before reading values: ./recalc_excel_mac.sh)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
