#!/usr/bin/env python3
"""Create and execute the Jupyter notebook version of the analysis.

The notebook is self-contained: it needs only numpy, pandas, scipy and matplotlib, so it can be
handed in on its own. Its functions are not written twice. This script copies their source from
the cdo package into the notebook cells, so the notebook and the tested package cannot drift apart.

    python build_notebook.py                              # notebooks/Miniproject3_CDO_Analysis.ipynb
    python build_notebook.py --out ../somewhere/cdo_simulation.ipynb
"""
from __future__ import annotations

import argparse
import ast
import inspect
import pprint
import sys
import textwrap
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cdo import BASE_DEAL, MARKET_YTM, N_CASES, PROJECT, RISK_FREE, SEED  # noqa: E402
from cdo.analysis import (  # noqa: E402
    B_NOTIONAL_GRID,
    LGD_GRID,
    PD_GRID,
    RHO_GRID,
    STRESS_LGD,
    STRESS_PD_GRID,
    STRESS_RHO_GRID,
    average_pairwise_correlation,
    default_count_table,
    describe,
    exact_default_count_distribution,
    expected_pool_cash_flows,
    quarterly_table,
    sensitivity,
    summary_row,
)
from cdo.bonds import bis_cash_flows, bis_expected_value, promised_cash_flows  # noqa: E402
from cdo.defaults import correlate, correlation_matrix, default_period, default_times  # noqa: E402
from cdo.model import simulate, validate_deal  # noqa: E402
from cdo.random_numbers import load_fixed_normals, moment_match  # noqa: E402
from cdo.report import (  # noqa: E402
    COLORS,
    apply_chart_style,
    plot_case,
    plot_default_count,
    plot_quarterly,
    plot_sensitivities,
    plot_total_cash,
    reference_line,
)
from cdo.waterfall import waterfall  # noqa: E402

DEFAULT_OUT = "../notebooks/Miniproject3_CDO_Analysis.ipynb"
EXAMPLE_CASE = 5


def plain_source(function) -> str:
    """Source of a package function with the type hints taken off its signature.

    The package keeps its annotations; the notebook reads better without them. Only the
    `def` line changes, the body (docstring, comments and code) is copied as it is.
    """
    source = textwrap.dedent(inspect.getsource(function)).rstrip()
    first_body_line = ast.parse(source).body[0].body[0].lineno
    signature = inspect.signature(function)
    parameters = [p.replace(annotation=inspect.Parameter.empty) for p in signature.parameters.values()]
    plain = signature.replace(parameters=parameters, return_annotation=inspect.Signature.empty)
    return "\n".join([f"def {function.__name__}{plain}:"] + source.splitlines()[first_body_line - 1:])


def source_of(*functions) -> str:
    """Source code of package functions, ready to paste into a notebook cell."""
    return "\n\n\n".join(plain_source(function) for function in functions)


def code(*parts: str) -> tuple:
    """A code cell made of several blocks (function source, then the lines that use it)."""
    return ("code", "\n\n\n".join(part.strip("\n") for part in parts))


def md(text: str) -> tuple:
    return ("md", text.strip("\n"))


def inputs_cell() -> str:
    """The inputs cell, with the values taken from the package's base deal."""
    d = BASE_DEAL
    return f"""
N_CASES = {N_CASES}
SEED = {SEED}                   # for the fixed random numbers

base = {{
    "n_bonds": {d["n_bonds"]},
    "years": {d["years"]},
    "freq": {d["freq"]},                # payments per year
    "face": {d["face"]},             # $MM per bond
    "coupon": {d["coupon"]},           # annual, paid quarterly
    "pd": {d["pd"]},               # annual default probability (pi)
    "lgd": {d["lgd"]},               # loss given default (lambda)
    "rho": {d["rho"]},               # "correlation" between any two bonds' default dates
    "a_notional": {d["a_notional"]},       # Class A
    "a_coupon": {d["a_coupon"]},
    "b_notional": {d["b_notional"]},       # Class B
    "b_coupon": {d["b_coupon"]},
}}

N_BONDS = base["n_bonds"]
FREQ = base["freq"]
N_PERIODS = base["years"] * base["freq"]      # 20 quarters

MARKET_YTM = {MARKET_YTM}             # Part 2
RISK_FREE = {RISK_FREE}              # Part 2

quarters = np.arange(1, N_PERIODS + 1)
"""


def build_cells() -> list:
    """The notebook as a list of ("md" | "code", text) pairs."""
    return [
        code("""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from scipy.stats import binom, norm
"""),
        md(f"""
# Mini-Project 3 (Part 1): Simplified CDO analysis

{PROJECT["course"]} | {PROJECT["authors"]}

Ten 5-year speculative grade bonds sit in a CDO. We simulate 1000 cases of correlated default
times from a fixed set of random numbers, turn each case into quarterly cash flows for the 10
bonds with the BIS debt model, and push the pooled cash through the waterfall
(Class A, then Class B, then the bank's equity).

To look at a single case, change `CASE` in section 7. Valuation is Part 2 and is not done here.
"""),
        md("""
## 1. Inputs

All dollar amounts are in $ millions. The YTM and the risk-free rate are listed because they are
given, but they are only needed for the valuation in Part 2.
"""),
        code(inputs_cell()),
        md("""
## 2. The BIS debt model for one bond

From Lecture 5: the bond is split into a non-defaultable part, (1 - LGD) of every promised
payment, which is always paid, and a defaultable part, LGD of every promised payment, which
stops in the period of default. So a bond that defaults in period k pays the full promised
amount in periods 1..k-1 and (1 - LGD) x promised from period k through maturity. Principal
is treated the same way as the coupons.

Before using it on the CDO, the function is checked against the example on the lecture slide
(face 1000, 5.5% annual coupon, LGD 60%, pi 3%, r 4%), which should give a value of 984.73.
"""),
        code(source_of(promised_cash_flows, bis_cash_flows, bis_expected_value), """
# Check against the BIS slide in Lecture 5 (annual payments).
slide_promised = promised_cash_flows(1000, 0.055, 5, 1)
rows, value = bis_expected_value(slide_promised, 0.03, 0.60, 0.04)
for year, prob, pv in rows:
    print(f"default year {year if year else 'none'}:  prob {prob:6.2%}   PV {pv:8.2f}")
print(f"Expected PV = {value:.2f}  (slide: 984.73)")
"""),
        md("""
## 3. Fixed random numbers

1000 cases x 10 bonds of independent standard normals. They were drawn once with a fixed seed and
stored in `fixed_random_numbers.csv`. The notebook reads that file, so every run (and Part 2) uses
exactly the same numbers. If the file is missing it is created again from the same seed. Cases are
numbered 1 to 1000.
"""),
        code(source_of(load_fixed_normals), """
candidates = [Path("fixed_random_numbers.csv"), Path("../data/fixed_random_numbers.csv")]
random_file = next((f for f in candidates if f.exists()), candidates[0])

z_table, source = load_fixed_normals(random_file, N_CASES, N_BONDS, SEED)
Z_raw = z_table.values
bond_names = list(z_table.columns)
print(f"Independent normals: {Z_raw.shape[0]} cases x {Z_raw.shape[1]} bonds ({source})")
z_table.head()
"""),
        md("""
### Moment matching

1000 draws never have exactly the moments they are supposed to have. Each bond's numbers have a mean
a little away from 0 and a variance a little away from 1, and two bonds' numbers are slightly
correlated just by chance. As shown in class, we take this out before the numbers are used:

1. subtract each column's mean (de-mean),
2. compute the covariance matrix of the draws and its Cholesky factor L,
3. multiply the de-meaned draws by the inverse of L.

The result `Z` has column means of exactly 0, variances of exactly 1 and no correlation between
columns, and it is what the model uses from here on. The numbers are still fixed: the same file
always gives the same `Z`.
"""),
        code(source_of(moment_match), """
Z = moment_match(Z_raw)


def largest_correlation(A):
    corr = np.corrcoef(A, rowvar=False)
    return np.abs(corr[~np.eye(A.shape[1], dtype=bool)]).max()


check_moments = pd.DataFrame({
    "largest |column mean|": [np.abs(Z_raw.mean(axis=0)).max(), np.abs(Z.mean(axis=0)).max()],
    "smallest variance": [Z_raw.var(axis=0).min(), Z.var(axis=0).min()],
    "largest variance": [Z_raw.var(axis=0).max(), Z.var(axis=0).max()],
    "largest |correlation| between two bonds": [largest_correlation(Z_raw), largest_correlation(Z)],
}, index=["as drawn", "moment matched"])
check_moments.round(4).abs()
"""),
        md("""
## 4. Correlated default times

Steps for every case:

1. Correlate the 10 moment-matched normals with the Cholesky factor of the 10 x 10 correlation
   matrix (1 on the diagonal, 0.20 everywhere else).
2. Convert each correlated normal to a uniform with the normal CDF, u = N(x).
3. Convert the uniform to a default time in years with the formula from class,
   t = ln(1 - u) / ln(1 - pi).
4. The bond defaults in quarter ceil(4t). If that is past quarter 20 the bond survives to maturity
   (stored as quarter 21).

The correlation is applied to the normals (a Gaussian copula). Because the numbers were moment
matched, the correlated normals have a correlation of exactly 0.20 across the 1000 cases. The
correlation of the default times themselves comes out a little below 0.20. Each bond on its own still has the geometric
default time with pi = 4% per year, so P(default by quarter k) = 1 - 0.96^(k/4).
"""),
        code(source_of(correlation_matrix, correlate, default_times, default_period, average_pairwise_correlation), """
X = correlate(Z, base["rho"])
T = default_times(Z, base["pd"], base["rho"])
Q = default_period(T, FREQ, N_PERIODS)
defaulted = Q <= N_PERIODS
n_defaults = defaulted.sum(axis=1)

p_default = 1 - (1 - base["pd"]) ** base["years"]
print(f"Average pairwise correlation of the correlated normals:  {average_pairwise_correlation(X):.3f}  (target {base['rho']:.2f})")
print(f"Average pairwise correlation of the default times:       {average_pairwise_correlation(T):.3f}")
print(f"Average pairwise correlation of the 5-year default flag: {average_pairwise_correlation(defaulted):.3f}")
print()
print(f"Share of bonds defaulting within 5 years: {defaulted.mean():.2%}  (theory 1 - 0.96^5 = {p_default:.2%})")
print(f"Average number of defaults per case:      {n_defaults.mean():.3f}  (theory {N_BONDS * p_default:.3f})")
"""),
        md("""
## 5. Collateral cash flows (Task 1)

Each bond promises 0.15 per quarter and 10.15 in quarter 20. The BIS function turns the default
quarters into cash flows for every case, bond and quarter, and the pool is the sum over the 10
bonds. The simulated average pool cash flow is compared with the exact expected value,
promised x [(1 - LGD) + LGD x 0.96^(k/4)].
"""),
        code(source_of(expected_pool_cash_flows), """
bond_promised = promised_cash_flows(base["face"], base["coupon"], N_PERIODS, FREQ)
pool_promised = N_BONDS * bond_promised

bond_cf = bis_cash_flows(bond_promised, Q, base["lgd"])      # (1000, 10, 20)
pool_cf = bond_cf.sum(axis=1)                                # (1000, 20)

pool_expected = expected_pool_cash_flows(base)
check = pd.DataFrame({
    "promised": pool_promised,
    "expected (exact)": pool_expected,
    "simulated mean": pool_cf.mean(axis=0),
    "simulated min": pool_cf.min(axis=0),
    "simulated max": pool_cf.max(axis=0),
}, index=pd.Index(quarters, name="quarter"))
check["sim / exact - 1 (%)"] = (check["simulated mean"] / check["expected (exact)"] - 1) * 100
check.round(4)
"""),
        md("""
## 6. Waterfall (Task 2)

Both classes are treated as bullet bonds that pay quarterly, like the collateral: Class A is owed
20 x 2% / 4 = 0.10 per quarter and 20.10 in quarter 20, Class B is owed 10 x 4% / 4 = 0.10 per
quarter and 10.10 in quarter 20.

Each quarter, on its own:

- Class A gets min(pool cash, amount due to A)
- Class B gets min(what is left, amount due to B)
- the bank's equity gets the rest

There are no carry-forwards, so a shortfall in one quarter is not made up later and excess cash
is not held back.
"""),
        code(source_of(waterfall), """
a_due = promised_cash_flows(base["a_notional"], base["a_coupon"], N_PERIODS, FREQ)
b_due = promised_cash_flows(base["b_notional"], base["b_coupon"], N_PERIODS, FREQ)

paid = waterfall(pool_cf, a_due, b_due)
a_cf, b_cf, eq_cf = paid["A"], paid["B"], paid["Equity"]

assert np.allclose(a_cf + b_cf + eq_cf, pool_cf), "waterfall does not add up to the pool"
assert (eq_cf >= -1e-12).all()

a_short = (a_due - a_cf).sum(axis=1)
b_short = (b_due - b_cf).sum(axis=1)
print(f"Due to Class A: {a_due[0]:.2f} per quarter, {a_due[-1]:.2f} at maturity, {a_due.sum():.2f} in total")
print(f"Due to Class B: {b_due[0]:.2f} per quarter, {b_due[-1]:.2f} at maturity, {b_due.sum():.2f} in total")
print(f"Cases with any shortfall on Class A: {(a_short > 1e-9).sum()} of {N_CASES}")
print(f"Cases with any shortfall on Class B: {(b_short > 1e-9).sum()} of {N_CASES}")
print()
floor = pool_promised * (1 - base["lgd"])
print("Worst possible pool cash (all 10 bonds default in quarter 1): "
      f"{floor[0]:.2f} per quarter and {floor[-1]:.2f} at maturity")
print("Needed for A + B:                                             "
      f"{a_due[0] + b_due[0]:.2f} per quarter and {a_due[-1] + b_due[-1]:.2f} at maturity")
"""),
        md("""
With LGD at 60% the pool can never pay less than 40% of what it promised, which is 0.60 per
quarter and 40.60 at maturity. Classes A and B together need 0.20 per quarter and 30.20 at
maturity, so under the base inputs they are paid in full in every case, not only in the 1000
simulated ones. All of the default risk lands on the bank's equity. Section 9 shows which inputs
have to move before the classes are touched.

The steps above are also wrapped in one function, `simulate`, so the whole model can be rerun
with other inputs in section 9. It checks the inputs first and gives back the same arrays.
"""),
        code(source_of(validate_deal, simulate), """
r = simulate(Z, base)
assert np.array_equal(r["pool_cf"], pool_cf) and np.array_equal(r["eq_cf"], eq_cf)
print("simulate() reproduces the step-by-step cash flows")
"""),
        md("""
## 7. Look at one case

Set `CASE` to any number from 1 to 1000.
"""),
        code(f"""
CASE = {EXAMPLE_CASE}
""" + """
assert 1 <= CASE <= N_CASES, "CASE must be between 1 and 1000"
i = CASE - 1
case_bonds = pd.DataFrame({
    "random normal (as drawn)": Z_raw[i],
    "moment matched": Z[i],
    "correlated normal": X[i],
    "uniform u": norm.cdf(X[i]),
    "default time (years)": T[i],
    "default quarter": np.where(defaulted[i], Q[i], 0),
}, index=bond_names)
case_bonds["status"] = np.where(defaulted[i], "defaults", "survives")
case_bonds["total cash received"] = bond_cf[i].sum(axis=1)
print(f"Case {CASE}: {n_defaults[i]} of {N_BONDS} bonds default within 5 years")
case_bonds.round(4)
"""),
        code("""
case_cf = pd.DataFrame(bond_cf[i].T, columns=bond_names, index=pd.Index(quarters, name="quarter"))
case_cf["Pool"] = pool_cf[i]
case_cf["Class A"] = a_cf[i]
case_cf["Class B"] = b_cf[i]
case_cf["Equity"] = eq_cf[i]
case_cf.loc["Total"] = case_cf.sum()
case_cf.round(3)
"""),
        code("# One colour per entity in every chart.\nCOLORS = " + pprint.pformat(COLORS, sort_dicts=False),
             source_of(apply_chart_style, plot_case), """
apply_chart_style()
plot_case(CASE, a_cf[i], b_cf[i], eq_cf[i], pool_promised)
plt.show()
"""),
        md("""
## 8. Statistical analysis (Task 3)

Totals are the sum of the 20 quarterly cash flows, not discounted (discounting is Part 2).

The "no-default amount" is what the pool or the class would receive if no bond defaulted. The
standard error is the standard deviation divided by the square root of the number of cases. It
treats the 1000 cases as independent, which is only approximately true after moment matching.
"""),
        code(source_of(describe), """
pool_total = pool_cf.sum(axis=1)
a_total = a_cf.sum(axis=1)
b_total = b_cf.sum(axis=1)
eq_total = eq_cf.sum(axis=1)
eq_promised = pool_promised.sum() - a_due.sum() - b_due.sum()

stats = pd.DataFrame({
    "Collateral pool": describe(pool_total, pool_promised.sum()),
    "Class A": describe(a_total, a_due.sum()),
    "Class B": describe(b_total, b_due.sum()),
    "Equity (bank)": describe(eq_total, eq_promised),
}).T
stats.round(3)
"""),
        md("""
The number of defaults per case is compared with two benchmarks. The first is the exact
distribution under the model itself, worked out without random numbers: equal pairwise correlation
is the same as one common factor shared by all bonds, and given that factor the bonds default
independently. It shows how close 1000 cases come to the true distribution. The second is the
binomial that would apply if the bonds defaulted independently, which shows what the correlation does.
"""),
        code(source_of(exact_default_count_distribution, default_count_table), """
default_dist = default_count_table(n_defaults, pool_total, eq_total, N_BONDS, p_default, base["rho"])
default_dist.round(4)
"""),
        code(source_of(plot_default_count), """
plot_default_count(default_dist, base["rho"])
plt.show()
"""),
        code(source_of(reference_line, plot_total_cash), """
plot_total_cash(pool_total, eq_total, pool_promised.sum(), eq_promised)
plt.show()
"""),
        code(source_of(quarterly_table), """
quarterly = quarterly_table(r, pool_expected)
quarterly.round(3)
"""),
        code(source_of(plot_quarterly), """
plot_quarterly(quarterly, floor[0], a_due[0] + b_due[0])
plt.show()
"""),
        md("""
## 9. Sensitivities

Each input is changed one at a time and the whole model is rerun on the same 1000 x 10 fixed
random numbers, so the differences between rows come from the input and not from new draws.
A "shortfall" means the class received less than it was due in at least one quarter.
"""),
        code(source_of(summary_row, sensitivity), f"""
sens_pd = sensitivity(Z, base, "pd", {PD_GRID!r})
sens_pd.round(3)
"""),
        code(f"""
sens_lgd = sensitivity(Z, base, "lgd", {LGD_GRID!r})
sens_lgd.round(3)
"""),
        code(f"""
sens_rho = sensitivity(Z, base, "rho", {RHO_GRID!r})
sens_rho.round(3)
"""),
        code(source_of(plot_sensitivities), """
plot_sensitivities(sens_pd, sens_lgd, sens_rho)
plt.show()
"""),
        md(f"""
The classes only become risky when the LGD goes above the point where a fully defaulted pool can
no longer cover them. At maturity the worst case pool pays 101.5 x (1 - LGD) and A + B need 30.2,
so Class B is safe up to LGD = 1 - 30.2 / 101.5 = 70.2% and Class A up to 1 - 20.1 / 101.5 = 80.2%.
Above that it takes several defaults at once to cause a shortfall, which is where the default
probability and the correlation start to matter for the classes. The two tables below repeat the
PD and correlation runs with LGD set to {STRESS_LGD:.0%} (nothing recovered).
"""),
        code(f"""
stress = {{**base, "lgd": {STRESS_LGD}}}
stress_columns = ["avg defaults", "equity mean", "equity 5th pct", "P(A shortfall)", "P(B shortfall)", "B paid / due"]

# LGD = 100%, varying the annual default probability (correlation 0.20)
stress_pd = sensitivity(Z, stress, "pd", {STRESS_PD_GRID!r})
stress_pd[stress_columns].round(4)
"""),
        code(f"""
# LGD = 100%, varying the correlation (annual default probability 4%)
stress_rho = sensitivity(Z, stress, "rho", {STRESS_RHO_GRID!r})
stress_rho[stress_columns].round(4)
"""),
        md("""
The last run changes the structure instead of the collateral: the Class B notional is increased
with everything else at base. In the worst case the pool pays 40.6 at maturity, Class A takes
20.1, and Class B is due 1.01 times its notional, so Class B stays fully covered up to
(40.6 - 20.1) / 1.01 = $20.3 MM. Beyond that its payment starts to depend on how many bonds default.
"""),
        code(f"""
sens_b = sensitivity(Z, base, "b_notional", {B_NOTIONAL_GRID!r})
sens_b[["equity mean", "equity 5th pct", "equity min", "P(A shortfall)", "P(B shortfall)", "B paid / due"]].round(4)
"""),
        md("""
## 10. Summary table for the write-up
"""),
        code("""
summary = pd.DataFrame({
    "No-default amount ($MM)": stats["no-default amount"],
    "Mean ($MM)": stats["mean"],
    "Std error ($MM)": stats["std error"],
    "Std dev ($MM)": stats["std dev"],
    "5th pct ($MM)": stats["5th pct"],
    "Worst case ($MM)": stats["min"],
    "Mean / no-default amount": stats["mean / no-default amount"],
    "P(below no-default amount)": [(pool_total < pool_promised.sum() - 1e-9).mean(), (a_short > 1e-9).mean(),
                          (b_short > 1e-9).mean(), (eq_total < eq_promised - 1e-9).mean()],
})
summary.round(3)
"""),
    ]


def build_notebook() -> nbformat.NotebookNode:
    cells = [new_markdown_cell(text) if kind == "md" else new_code_cell(text) for kind, text in build_cells()]
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return nb


def execute(nb: nbformat.NotebookNode, cwd: Path) -> None:
    """Run every cell with cwd as the working directory, keeping the outputs in the notebook."""
    from nbclient import NotebookClient

    NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(cwd)}}).execute()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=DEFAULT_OUT, help="notebook file to write")
    parser.add_argument("--no-execute", action="store_true", help="write the notebook without running it")
    args = parser.parse_args(argv)

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    nb = build_notebook()
    if not args.no_execute:
        execute(nb, out.parent)
    nbformat.write(nb, out)
    print(f"Notebook written to {out} ({len(nb.cells)} cells, {'executed' if not args.no_execute else 'not executed'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
