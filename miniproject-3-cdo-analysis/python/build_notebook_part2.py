#!/usr/bin/env python3
"""Build the Part 2 notebook (CDO valuation with the Lintner model) from the package source.

The notebook is self-contained: every function it uses is copied in from the cdo package, so it
runs next to fixed_random_numbers.csv with nothing else installed beyond numpy, scipy, pandas
and matplotlib.

    python build_notebook_part2.py                  # ../notebooks/Miniproject3_Part2_CDO_Valuation.ipynb
    python build_notebook_part2.py --out x.ipynb    # somewhere else
"""
from __future__ import annotations

import argparse
import pprint
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_notebook import code, execute, md, source_of  # noqa: E402
from cdo import MARKET_YTM, N_CASES, PART2_DEAL, PART2_RISK_FREE, PROJECT_PART2, SEED  # noqa: E402
from cdo.bonds import bis_cash_flows, promised_cash_flows  # noqa: E402
from cdo.defaults import correlate, correlation_matrix, default_period, default_times  # noqa: E402
from cdo.leverage import GOAL_SEEK, bank_position, irr, leverage_analysis, leverage_point, max_class_b_for_risk  # noqa: E402
from cdo.model import simulate, validate_deal  # noqa: E402
from cdo.random_numbers import load_fixed_normals, moment_match  # noqa: E402
from cdo.report import (  # noqa: E402
    COLORS, apply_chart_style, plot_class_b_choice, plot_leverage, plot_pv_distributions, reference_line,
)
from cdo.valuation import (  # noqa: E402
    bond_market_value, break_even_size, cost_of_risk, default_indicators, discount_factors, fair_coupon,
    gain_turns_positive, largest_class_b_at_par, lintner_values, optimise_class_b, structure_values,
    tranche_cash_flows, tranche_pvs, valuation_precision, value_cdo, value_sensitivity,
)
from cdo.waterfall import waterfall  # noqa: E402

DEFAULT_OUT = "../notebooks/Miniproject3_Part2_CDO_Valuation.ipynb"


def inputs_cell() -> str:
    d = PART2_DEAL
    return f"""
N_CASES = {N_CASES}
SEED = {SEED}                   # for the fixed random numbers (same as Part 1)

base = {{
    "n_bonds": {d["n_bonds"]},
    "years": {d["years"]},
    "freq": {d["freq"]},                # payments per year
    "face": {d["face"]},             # $MM per bond
    "coupon": {d["coupon"]},           # annual, paid quarterly
    "pd": {d["pd"]},               # annual default probability (pi)
    "lgd": {d["lgd"]},               # loss given default (lambda)
    "rho": {d["rho"]},               # "correlation" between any two bonds' default dates
    "a_notional": {d["a_notional"]},       # Class A principal ($MM)
    "a_coupon": {d["a_coupon"]},        # Class A target rate, used as its coupon
    "b_notional": {d["b_notional"]},       # Class B principal ($MM)
    "b_coupon": {d["b_coupon"]},        # Class B target rate, used as its coupon
}}

RISK_FREE = {PART2_RISK_FREE}              # p.a., compounded quarterly
MARKET_YTM = {MARKET_YTM}             # market yield on the bonds

N_BONDS = base["n_bonds"]
FREQ = base["freq"]
N_PERIODS = base["years"] * base["freq"]      # 20 quarters
quarters = np.arange(1, N_PERIODS + 1)
"""


def build_cells() -> list:
    return [
        code("""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from scipy.optimize import brentq
from scipy.stats import norm
"""),
        md(f"""
# Mini-Project 3 (Part 2): CDO valuation with the Lintner model

{PROJECT_PART2["course"]} | {PROJECT_PART2["authors"]}

We value Class A, Class B and the bank's residual with the Lintner model from class (Lecture 6), on the
same 1000 cases of correlated defaults as Part 1. The steps follow the assignment:

1. scheduled cash flows and the waterfall for the new classes,
2. 1000 simulations from the static (moment-matched) random numbers,
3. the PV of each tranche in each case at the risk-free rate, the covariance matrix and a default indicator,
4. the implicit cost of risk from the bonds' market value, and the cost per unit risk,
5. the Lintner values,

and then the bonus: the structure that gives the residual the most value with Class A held at $15 MM.
"""),
        md("""
## 1. Inputs

The classes are the ones given for Part 2. We use the target rates as the coupons, so a class sold
at par pays its target rate. All amounts are in $ millions.
"""),
        code(inputs_cell()),
        md("""
## 2. The Part 1 model

These are the Part 1 functions, unchanged: the BIS bond cash flows, the fixed random numbers and
their moment matching, the correlated default times and the waterfall.
"""),
        code(source_of(promised_cash_flows, bis_cash_flows, load_fixed_normals, moment_match, correlation_matrix,
                       correlate, default_times, default_period, waterfall, validate_deal, simulate)),
        md("""
## 3. Step 1: scheduled cash flows of the classes

Each class gets its coupon every quarter and its principal back in quarter 20. If no bond defaults the residual
gets the rest. Each quarter stands alone: a shortfall is not made up later.
"""),
        code("""
a_due = promised_cash_flows(base["a_notional"], base["a_coupon"], N_PERIODS, FREQ)
b_due = promised_cash_flows(base["b_notional"], base["b_coupon"], N_PERIODS, FREQ)
pool_due = N_BONDS * promised_cash_flows(base["face"], base["coupon"], N_PERIODS, FREQ)
schedule = pd.DataFrame({"pool promised": pool_due, "due to Class A": a_due, "due to Class B": b_due,
                         "residual if no default": pool_due - a_due - b_due}, index=pd.Index(quarters, name="quarter"))
schedule.loc["Total"] = schedule.sum()
schedule.round(5)
"""),
        md("""
## 4. Step 2: 1000 simulations from the static random numbers

The same file of normals as Part 1, moment matched as shown in class, so the default dates and the
pool's cash flows are the Part 1 ones. Only the split between the classes changes.
"""),
        code("""
candidates = [Path("fixed_random_numbers.csv"), Path("../data/fixed_random_numbers.csv")]
random_file = next((f for f in candidates if f.exists()), candidates[0])
z_table, source = load_fixed_normals(random_file, N_CASES, N_BONDS, SEED)
Z = moment_match(z_table.values)
sim = simulate(Z, base)
pool_cf = sim["pool_cf"]
print(f"{Z.shape[0]} cases x {Z.shape[1]} bonds ({source}), moment matched")
print(f"Average number of defaults per case: {sim['defaulted'].sum(axis=1).mean():.3f}")
"""),
        md("""
## 5. Step 3: PVs at the risk-free rate, covariance matrix and default indicators

In each case every tranche's 20 quarterly cash flows are discounted at the risk-free rate. The
default indicator is 1 when a class is paid less than it is due in some quarter.
"""),
        code(source_of(discount_factors, tranche_cash_flows, tranche_pvs, default_indicators), """
flows = tranche_cash_flows(pool_cf, base)
pv = tranche_pvs(flows, RISK_FREE, FREQ)
defaults = default_indicators(flows)
names = ["Class A", "Class B", "Residual"]
pv_table = pd.DataFrame(pv, columns=names, index=pd.Index(np.arange(1, N_CASES + 1), name="case"))
pv_table["Pool"] = pv_table.sum(axis=1)
pv_table["Class A default"] = defaults[:, 0]
pv_table["Class B default"] = defaults[:, 1]
print("Cases in which each class is short:", dict(zip(["Class A", "Class B"], defaults.sum(axis=0))))
pv_table.head(10).round(4)
"""),
        code("""
covariance = pd.DataFrame(np.cov(pv, rowvar=False), index=names, columns=names)
covariance.round(4)
"""),
        md("""
## 6. Step 4: the implicit cost of risk

The bonds' market value is the pool's promised cash flows at the 9% market yield. The pool's expected
PV at the risk-free rate is higher; the gap is what the market charges for the risk. Dividing by the
variance of the pool's PV gives the cost per unit risk (Lecture 6, slide 44).
"""),
        code(source_of(bond_market_value, cost_of_risk), """
risk = cost_of_risk(pv.sum(axis=1), bond_market_value(base, MARKET_YTM))
for key, value in risk.items():
    print(f"{key:20} {value:12.6f}")
"""),
        md("""
## 7. Step 5: Lintner values

Each tranche's value is its expected PV less the cost per unit risk times the covariance of its PV
with the pool's PV. The three covariances add up to the pool's variance, so the three values add up
to the bonds' market value.
"""),
        code(source_of(lintner_values, value_cdo), """
v = value_cdo(pool_cf, base, RISK_FREE, MARKET_YTM)
values = pd.DataFrame({
    "principal": [base["a_notional"], base["b_notional"], np.nan],
    "expected PV": v["expected_pv"],
    "covariance with pool": v["cov_with_pool"],
    "risk charge": v["risk_charge"],
    "Lintner value": v["value"],
}, index=names)
values.loc["Total"] = values.sum()
print(f"Total of the values {v['value'].sum():.6f} vs bonds' market value {v['market_value']:.6f}")
values.round(4)
"""),
        md("""
Neither class is ever short, so their PVs are the same in every case: no covariance with the pool, no
risk charge, and each is worth its promised cash flows at 4%. That is more than par, because their
coupons are above 4%. The coupon at which each class is worth exactly its principal:
"""),
        code(source_of(structure_values, fair_coupon), """
lam = v["cost_per_unit_risk"]
fair = {cls: fair_coupon(pool_cf, base, RISK_FREE, lam, cls) for cls in ("a", "b")}
at_fair = value_cdo(pool_cf, {**base, "a_coupon": fair["a"], "b_coupon": fair["b"]}, RISK_FREE, MARKET_YTM)
print(f"Fair coupons: Class A {fair['a']:.4%}, Class B {fair['b']:.4%}")
print(f"Values at those coupons: {np.round(at_fair['value'], 4)}")
print(f"Bank's gain at the target rates: {base['a_notional'] + base['b_notional'] + v['value'][2] - v['market_value']:.4f}")
"""),
        code("# One colour per entity in every chart.\nCOLORS = " + pprint.pformat(COLORS, sort_dicts=False),
             source_of(apply_chart_style, reference_line, plot_pv_distributions), """
apply_chart_style()
plot_pv_distributions(pv.sum(axis=1), pv[:, 2], v["market_value"], v["value"][2], RISK_FREE)
plt.show()
"""),
        md("""
## 8. What moves the values

One input of the default model at a time, with the bonds' market value held at its 9% price. Then the
same valuation on 500 fresh sets of 1000 cases, as drawn and moment matched, to see how much the numbers
move from one set of random numbers to another.
"""),
        code(source_of(value_sensitivity), """
for name, grid in {"rho": (0.0, 0.2, 0.4, 0.6, 0.8), "pd": (0.02, 0.04, 0.06, 0.08, 0.12),
                   "lgd": (0.4, 0.6, 0.8, 1.0)}.items():
    display(pd.DataFrame(value_sensitivity(Z, base, RISK_FREE, MARKET_YTM, name, grid)).set_index(name).round(4))
"""),
        code(source_of(valuation_precision), """
precision = valuation_precision(base, RISK_FREE, MARKET_YTM, N_CASES, n_sets=500)
pd.DataFrame({"sd as drawn": precision["sd_as_drawn"], "sd moment matched": precision["sd_matched"]},
             index=precision["columns"]).round(4)
"""),
        md("""
## 9. Bonus: choosing Class B with Class A held at $15 MM

We search Class B's principal ($0 to $100 MM) and coupon (0% to 10%) and re-value each structure on the
same cases. The cost per unit risk comes from the bonds alone, so it stays fixed. We report the literal
answer (the residual's highest value), the bank's gain (cash from selling both classes at par plus the
residual, less the bonds) at Class B's target rate, Class B's fair coupon at each size, and the best size
at the target rate if Class B's default probability is capped.
"""),
        code(source_of(break_even_size, gain_turns_positive, largest_class_b_at_par, optimise_class_b), """
sizes = np.arange(0.0, 100.5, 1.0)
coupons = np.round(np.arange(0.0, 0.10001, 0.0025), 4)
bonus = optimise_class_b(pool_cf, base, RISK_FREE, MARKET_YTM, sizes, coupons)
print("Literal answer:", {k: round(x, 4) for k, x in bonus["literal"].items()})
print(f"5.5% is a fair rate for Class B at {bonus['break_even_size']:.2f} MM "
      f"(short in {bonus['p_default_b_at_break_even']:.1%} of cases)")
print(f"At 5.5% the bank's gain turns positive at about {bonus['gain_turns_positive']:.2f} MM")
print(f"Largest Class B that any coupon can make worth par: {bonus['largest_class_b_at_par']:.2f} MM "
      f"(the bonds less Class A: {bonus['bonds_less_class_a']:.2f} MM)")
print(f"Lowest residual value in the search: {bonus['min_residual_value_in_grid']:.4f}")
pd.DataFrame(bonus["default_limits"]).round(4)
"""),
        code("""
at_target = pd.DataFrame(bonus["at_target_rate"]).set_index("b_notional")
fair_curve = pd.DataFrame(bonus["fair_coupon_curve"]).set_index("b_notional")
table = at_target[["value_b", "value_residual", "bank_gain", "p_default_b"]].join(
    fair_curve[["fair_coupon", "value_residual"]], rsuffix=" at fair coupon")
table.loc[[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 65.0, 70.0, 80.0]].round(4)
"""),
        code(source_of(plot_class_b_choice), """
plot_class_b_choice(bonus["at_target_rate"], bonus["fair_coupon_curve"], base["b_coupon"], bonus["break_even_size"],
                    base["b_notional"])
plt.show()
"""),
        md("""
Taken literally, the residual is worth the most with no Class B, because the three values always add up
to the bonds' market value. Measured by the bank's gain, Class B at its fair coupon gives the same result
at every size (the bank loses only what Class A's 4.5% costs). At the 5.5% target rate, 5.5% is a fair rate
for Class B at about $65.5 MM and the bank's gain only turns positive at about $66.5 MM, where Class B investors
would be paying more than the class is worth. So once Class B is priced fairly, its size is a question of risk,
not value. That is the next section.
"""),
        md("""
## 10. Bonus: leverage and the bank's risk (the Leverage sheet of the workbook)

Each class is sold for what it is worth: Class A for its Lintner value, Class B at par at its fair coupon.
The bank pays the rest of the bonds' price, which is always exactly the residual's Lintner value. For each
size of Class B we find its fair coupon, then the bank's expected IRR (its investment against the residual's
average cash flow in each quarter, compounded to a year) and its relative equity risk (the standard
deviation of the residual's PV across the cases divided by the investment). For each cap on that risk we
find the largest Class B by root finding. The workbook does the same with Goal Seek; its answers are in
`GOAL_SEEK` for comparison.
"""),
        code(f"GOAL_SEEK = {GOAL_SEEK!r}\nRISK_LIMITS = tuple(limit for limit, _ in GOAL_SEEK)",
             source_of(irr, bank_position, leverage_point, max_class_b_for_risk, leverage_analysis), """
lev = leverage_analysis(pool_cf, base, RISK_FREE, MARKET_YTM, np.arange(0.0, 72.01, 0.5))
for name, label in (("no_class_b", "No Class B"), ("as_given", "Class B $30 MM at 5.5%, sold for its value")):
    x = lev[name]
    print(f"{label}: bank's investment {x['investment']:.4f}, expected IRR {x['expected_irr']:.4%}, "
          f"relative risk {x['relative_risk']:.4f}")
limits = pd.DataFrame([row for row in lev["risk_limits"] if row is not None]).set_index("max_relative_risk")
limits[["b_notional", "b_coupon", "p_default_b", "investment", "expected_irr", "excess_irr", "relative_risk",
        "goal_seek_b_notional"]].round(4)
"""),
        code(source_of(plot_leverage), """
plot_leverage(lev["profile"], lev["risk_limits"], lev["no_class_b"], lev["risk_free_annual"])
plt.show()
"""),
        md("""
Under the Lintner model a bigger Class B adds no value: the bank's investment always equals the residual's
value. It only raises the bank's expected IRR and its risk together, so the choice depends on how much risk
the bank will carry. With a cap of 100% on the relative equity risk the bank can sell about $65.25 MM of
Class B at a fair 5.47%, close to the 5.5% target rate, and expect 26.15% a year on $7.44 MM; with a cap of
50%, about $54.19 MM at 4.46% for 16.16%. Goal Seek's sizes are within $0.17 MM of the exact ones.
"""),
    ]


def build_notebook() -> nbformat.NotebookNode:
    cells = [new_markdown_cell(text) if kind == "md" else new_code_cell(text) for kind, text in build_cells()]
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return nb


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
