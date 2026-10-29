#!/usr/bin/env python3
"""Write the Part 2 report (PDF) from output/part2/results.json and the figures in output/part2/figures.

    python write_report_part2.py            # writeup/Miniproject3_Part2_CDO_Valuation_Report.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cdo import PROJECT_PART2  # noqa: E402
from write_report import (  # noqa: E402
    AVAILABLE, BODY, BOTTOM_MARGIN, CAPTION, CELL, H2, H2_SPLIT, ITEM, LEFT, MUTED, NAVY, SIDE_MARGIN, SUBTITLE, TITLE,
    TOP_MARGIN, figure, keyed, mm, pct, table,
)

REPORT_NAME = "Miniproject3_Part2_CDO_Valuation_Report.pdf"
BONUS_SIZES = (10, 20, 30, 40, 50, 60, 65, 70, 80)


def assumptions(r: dict) -> list:
    """(heading, text) pairs for the Key Assumptions block."""
    d, v = r["deal"], r["valuation"]
    a, b = r["promised"]["class_a"], r["promised"]["class_b"]
    return [
        ("Collateral and Defaults",
         f"As in Part 1: {d['n_bonds']} bonds of ${d['face']:.0f} MM, {pct(d['coupon'], 0)} coupon paid quarterly, "
         f"{d['years']} years; BIS debt model with a {pct(d['pd'], 0)} annual default probability, {pct(d['lgd'], 0)} LGD "
         f"and {d['rho']:.2f} correlation between default dates; the same {r['n_cases']} moment-matched cases "
         f"(seed {r['seed']})."),
        ("Classes",
         f"Class A: ${d['a_notional']:.0f} MM at {pct(d['a_coupon'])}. Class B: ${d['b_notional']:.0f} MM at "
         f"{pct(d['b_coupon'])}. We use the target rates in the assignment as the coupons, so a class sold at par pays "
         f"its target rate. Both are bullet bonds: Class A is owed ${a[0]:.5f} MM a quarter and ${a[-1]:.5f} MM at "
         f"maturity, Class B ${b[0]:.4f} MM and ${b[-1]:.4f} MM. If no bond defaults the residual receives "
         f"${r['promised']['pool'][0] - a[0] - b[0]:.5f} MM a quarter and ${r['promised']['pool'][-1] - a[-1] - b[-1]:.5f} MM "
         "at maturity. The waterfall is the same as in Part 1: each quarter stands alone, so a shortfall is not "
         "made up later."),
        ("Risk-Free Rate",
         f"{pct(r['risk_free'], 0)} a year, compounded quarterly: a payment in quarter k is discounted by "
         f"(1 + {r['risk_free']:.2f}/4)^-k."),
        ("Bonds' Market Value",
         f"The pool's promised cash flows discounted at the {pct(r['market_ytm'], 0)} market yield, compounded "
         f"quarterly: ${mm(v['market_value'])} MM for the ten bonds."),
        ("Lintner Method",
         "From Lecture 6. Cost of risk = expected PV of the pool at the risk-free rate less the bonds' market value. "
         "Cost per unit risk = cost of risk / variance of the pool's PV. Value of a tranche = its expected PV less the "
         "cost per unit risk times the covariance of its PV with the pool's PV. Covariances use the sample (n - 1) "
         "formula, like COVARIANCE.S in Excel; the choice cancels out of the values."),
        ("Default Indicator",
         "For each case and class, 1 if the class is paid less than it is due in some quarter, otherwise 0."),
        ("Units", "$ millions. All PVs are as of today, at the risk-free rate."),
    ]


def steps(r: dict) -> list:
    """(heading, text) pairs for the Implementation Steps block, following the steps in the assignment."""
    v = r["valuation"]
    return [
        ("Scheduled Cash Flows",
         "Promised quarterly payments to Class A and Class B from their principal and coupon, then the Part 1 "
         "waterfall on the pool's cash flows in every case: Class A first, then Class B, the rest to the residual."),
        ("1000 Simulations",
         f"The {r['n_cases']} cases come from the fixed random numbers of Part 1 after moment matching, so the "
         "default dates, and therefore the pool's cash flows, are exactly the ones in Part 1."),
        ("PVs, Covariances and Default Indicators",
         f"In every case we discount each tranche's 20 quarterly cash flows at {pct(r['risk_free'], 0)}. That gives a "
         f"{r['n_cases']} x 3 table of PVs (Class A, Class B, residual), its 3 x 3 covariance matrix and a default "
         "indicator per case for each class."),
        ("Cost of Risk",
         f"The pool's expected PV at {pct(r['risk_free'], 0)} is ${mm(v['expected_pool_pv'])} MM, but the bonds trade "
         f"for ${mm(v['market_value'])} MM, so the market charges ${mm(v['cost_of_risk'])} MM for their risk. The pool's "
         f"PV has a variance of {v['pool_variance']:.2f} across the cases, so the cost per unit risk is "
         f"{v['cost_of_risk']:.4f} / {v['pool_variance']:.2f} = {v['cost_per_unit_risk']:.5f}."),
        ("Lintner Values",
         "Each tranche is charged the cost per unit risk times its covariance with the pool. The three covariances "
         "add up to the pool's variance, so the three values add up to the bonds' market value."),
        ("Cross-check",
         "The Valuation sheet of the Excel workbook does the same steps with formulas. It gives the same PVs in all "
         "1000 cases and the same values as the Python code. The Leverage sheet does the bank's side of the bonus "
         "with Goal Seek, and the Python code solves the same problem exactly."),
    ]


def valuation_table(r: dict) -> Table:
    v, d = r["valuation"], r["deal"]
    principal = [d["a_notional"], d["b_notional"], None]
    rows = [["", "Principal", "Expected PV\nat 4%", "Max error\n(2 s.e.)", "Covariance\nwith pool", "Risk\ncharge",
             "Lintner\nvalue", "Value per\n$100", "Cases\nshort"]]
    for i, name in enumerate(v["tranches"]):
        short = f"{v['default_cases'][i]}" if i < 2 else "-"
        per_100 = f"{100 * v['value'][i] / principal[i]:.2f}" if principal[i] else "-"
        rows.append([name, mm(principal[i]) if principal[i] else "-", mm(v["expected_pv"][i]), mm(v["max_error"][i]),
                     mm(v["cov_with_pool"][i]), mm(v["risk_charge"][i]), mm(v["value"][i]), per_100, short])
    rows.append(["Pool (bonds)", "100.00", mm(v["expected_pool_pv"]), "", mm(v["pool_variance"]), mm(v["cost_of_risk"]),
                 mm(sum(v["value"])), f"{v['market_value']:.2f}", ""])
    flowable = table(rows, font=8.5)
    flowable.setStyle(TableStyle([("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold")]))
    return flowable


def covariance_table(r: dict) -> Table:
    v = r["valuation"]
    rows = [["Covariance of PVs"] + list(v["tranches"]) + ["Sum (with pool)"]]
    for name, row, total in zip(v["tranches"], v["covariance"], v["cov_with_pool"]):
        rows.append([name] + [f"{0.0 if abs(x) < 5e-5 else x:.4f}" for x in row] + [f"{total:.4f}"])
    return table(rows, font=8.5)


def results_text(r: dict) -> list:
    v, d, fair, fl = r["valuation"], r["deal"], r["fair_coupons"], r["floor"]
    a, b, res = v["value"]
    return [
        Paragraph(
            f"Neither class misses a payment in any of the {r['n_cases']} cases: the default indicators are 0 "
            f"everywhere. Each quarter the pool pays at least ${fl['pool_per_quarter']:.2f} MM, more than the "
            f"${fl['classes_per_quarter']:.2f} MM the classes need. At maturity they need "
            f"${fl['classes_at_maturity']:.2f} MM, and the pool pays less than that only if all ten bonds default (it "
            f"would then pay ${fl['pool_at_maturity_all_default']:.2f} MM). That has a "
            f"{pct(fl['p_all_default_exact'], 2)} chance under the model and happens in none of our cases: the worst case "
            f"has {fl['worst_case_defaults']} defaults and still pays ${fl['min_pool_at_maturity']:.2f} MM at maturity. So "
            "the PVs of Class A and Class B are the same in every case, their covariances with the pool are 0, and they "
            "carry no risk charge: each is worth its promised cash flows at the risk-free rate. All of the "
            f"${mm(v['cost_of_risk'])} MM cost of risk falls on the residual, which is worth ${mm(res)} MM against an "
            f"expected PV of ${mm(v['expected_pv'][2])} MM.", BODY),
        Paragraph(
            f"Because the classes are never short in our cases but pay more than {pct(r['risk_free'], 0)}, they are "
            "worth more than "
            f"their principal: Class A ${mm(a)} MM for ${d['a_notional']:.0f} MM, Class B ${mm(b)} MM for "
            f"${d['b_notional']:.0f} MM. If the bank sells them at par, the investors receive "
            f"${mm(a + b - d['a_notional'] - d['b_notional'])} MM more than they pay. The bank keeps a residual worth "
            f"${mm(res)} MM after putting in ${mm(v['bank_investment'])} MM of its own (the bonds less the "
            f"${d['a_notional'] + d['b_notional']:.0f} MM raised), a loss of ${mm(-v['bank_gain'])} MM. The fair coupon "
            f"for both classes is {pct(fair['b'])}, the risk-free rate; at that coupon each class is worth exactly par "
            "and the residual is worth "
            f"${mm(r['at_fair_coupons']['value'][2])} MM.", BODY),
    ]


def risky_table(r: dict) -> Table:
    x = r["risky_example"]
    rows = [["", "Principal", "Expected PV\nat 4%", "Covariance\nwith pool", "Risk\ncharge", "Lintner\nvalue",
             "Value per\n$100", "Cases\nshort"]]
    principal = [r["deal"]["a_notional"], x["b_notional"], None]
    for i, name in enumerate(x["tranches"]):
        rows.append([name, mm(principal[i]) if principal[i] else "-", mm(x["expected_pv"][i]), mm(x["cov_with_pool"][i]),
                     mm(x["risk_charge"][i]), mm(x["value"][i]),
                     f"{100 * x['value'][i] / principal[i]:.2f}" if principal[i] else "-",
                     f"{x['default_cases'][i]}" if i < 2 else "-"])
    return table(rows, font=8.5)


def risky_text(r: dict) -> Paragraph:
    x, v = r["risky_example"], r["valuation"]
    return Paragraph(
        f"We kept the assignment's structure, and with it neither class carries any risk, so the covariance matrix "
        "has only one entry that is not 0. To see the method charge a class for risk, Table 3 values the same cases "
        f"with Class B at ${x['b_notional']:.0f} MM (still at {pct(x['b_coupon'])}). Class B is then short in "
        f"{x['default_cases'][1]} of the {r['n_cases']} cases, its PV moves with the pool's, and it is charged "
        f"${mm(x['risk_charge'][1])} MM for risk: it is worth ${mm(x['value'][1])} MM against an expected PV of "
        f"${mm(x['expected_pv'][1])} MM. The residual's charge falls to ${mm(x['risk_charge'][2])} MM, and the three "
        f"charges still add up to the same ${mm(v['cost_of_risk'])} MM.", BODY)


def sensitivity_table(r: dict) -> Table:
    labels = {"rho": ("Correlation", lambda x: f"{x:.1f}"), "pd": ("Default prob.", lambda x: pct(x, 0)),
              "lgd": ("LGD", lambda x: pct(x, 0))}
    base = r["deal"]
    rows = [["Input", "Value", "Expected\npool PV", "Cost of\nrisk", "Cost per\nunit risk", "Class A", "Class B",
             "Residual", "B short"]]
    for name, (label, fmt) in labels.items():
        for row in r["sensitivities"][name]:
            is_base = abs(row[name] - base[name]) < 1e-12
            rows.append([label, fmt(row[name]) + (" (base)" if is_base else ""), mm(row["expected_pool_pv"]),
                         mm(row["cost_of_risk"]), f"{row['cost_per_unit_risk']:.4f}", mm(row["value_a"]),
                         mm(row["value_b"]), mm(row["value_residual"]), pct(row["p_default_b"])])
    return table(rows, font=8, pad=1.6)


def sensitivity_text(r: dict) -> list:
    s = {name: keyed(rows, name) for name, rows in r["sensitivities"].items()}
    prec = r["precision"]
    col = {name: i for i, name in enumerate(prec["columns"])}
    worst = s["pd"][0.12]
    return [
        Paragraph(
            "Changing an input of the default model moves the pool's expected PV, but the bonds' market value stays at "
            f"${mm(r['valuation']['market_value'])} MM because it comes from their price, not from our model. So the cost "
            "of risk takes up the change and the values hardly move: the residual stays near "
            f"${mm(r['valuation']['value'][2], 1)} MM in every row, and the classes keep their risk-free values until they "
            "start to miss payments (Class B at a high correlation, default probability or LGD; Class A only at 100% LGD, "
            "in 0.1% of the cases). Correlation leaves the expected PV alone but "
            "spreads the pool's PV out, so the cost per unit risk falls from "
            f"{s['rho'][0.0]['cost_per_unit_risk']:.3f} to {s['rho'][0.8]['cost_per_unit_risk']:.3f}. At a "
            f"{pct(0.12, 0)} default probability the pool's expected PV (${mm(worst['expected_pool_pv'])} MM) is below the "
            f"market value, the cost of risk turns negative (minus ${mm(-worst['cost_of_risk'])} MM) and the method stops making "
            "sense: a 9% yield would be too low for bonds that risky.", BODY),
        Paragraph(
            f"<b>Precision.</b> We valued the structure again on {prec['n_sets']} fresh sets of {r['n_cases']} cases, "
            "once with the numbers as drawn and once after moment matching. Moment matching cut the spread of the "
            f"pool's expected PV from ${prec['sd_as_drawn'][col['expected_pool_pv']]:.2f} MM to "
            f"${prec['sd_matched'][col['expected_pool_pv']]:.2f} MM and the spread of the cost per unit risk from "
            f"{prec['sd_as_drawn'][col['cost_per_unit_risk']]:.4f} to {prec['sd_matched'][col['cost_per_unit_risk']]:.4f}. "
            f"The Lintner values moved even less (about ${prec['sd_matched'][col['value_residual']]:.2f} MM for the "
            "residual either way), because the bonds' market value pins the total and the classes are nearly always "
            "paid in full.", BODY),
    ]


def bonus_table(r: dict) -> Table:
    bonus = r["bonus"]
    target = keyed(bonus["at_target_rate"], "b_notional")
    fair = keyed(bonus["fair_coupon_curve"], "b_notional")
    coupon = pct(bonus["target_b_coupon"])
    rows = [["Class B\nprincipal", f"Class B value\nat {coupon}", f"Residual\nat {coupon}", f"Bank's gain\nat {coupon}",
             f"Cases short\nat {coupon}", "Fair\ncoupon", "Residual at\nfair coupon"]]
    for size in BONUS_SIZES:
        t, f = target[float(size)], fair.get(float(size))
        rows.append([f"${size} MM", mm(t["value_b"]), mm(t["value_residual"]), mm(t["bank_gain"]), pct(t["p_default_b"]),
                     pct(f["fair_coupon"], 2) if f else "none", mm(f["value_residual"]) if f else "-"])
    return table(rows, font=8.5, first_left=False)


def bonus_text(r: dict) -> list:
    bonus, d = r["bonus"], r["deal"]
    n_structures = len({row["b_notional"] for row in bonus["at_target_rate"]})
    gain_at_fair = bonus["fair_coupon_curve"][0]["bank_gain"]
    target = keyed(bonus["at_target_rate"], "b_notional")
    fair = keyed(bonus["fair_coupon_curve"], "b_notional")
    proposed = target[float(d["b_notional"])]
    return [
        Paragraph(
            f"We held Class A at ${bonus['a_notional']:.0f} MM and {pct(bonus['a_coupon'])} and searched over Class B's "
            f"principal ($0 to $100 MM in $1 MM steps, {n_structures} sizes) and coupon (0% to 10% in 0.25% steps), "
            "re-running the waterfall and the Lintner valuation on the same 1000 cases for each structure. The cost per "
            "unit risk does not change from one structure to another, because it comes from the bonds alone.", BODY),
        Paragraph(
            f"<b>Taken literally,</b> the residual is worth the most with no Class B at all: "
            f"${mm(bonus['residual_with_no_class_b'])} MM. That follows from the method. The values of Class A, Class B "
            f"and the residual always add up to the bonds' ${mm(bonus['market_value'])} MM, so whatever Class B is worth "
            "comes out of the residual, and the residual is worth the most when Class B is worth nothing. We held Class "
            f"A's {pct(bonus['a_coupon'])} coupon in the search because the assignment fixes Class A; a lower Class A "
            "coupon would move value to the residual in the same way. It is not a useful answer, because it ignores "
            "the cash the bank receives for Class B.", BODY),
        Paragraph(
            "<b>The bank's gain.</b> So we measured each structure by the bank's gain: the cash from selling both "
            "classes at par plus the value of the residual it keeps, less the bonds it puts in (Figure 2 and Table 5). "
            f"At Class B's {pct(bonus['target_b_coupon'])} target rate the gain is negative for every Class B up to about "
            f"${bonus['gain_turns_positive']:.1f} MM. Up to ${bonus['break_even_size']:.1f} MM, Class B's fair rate is "
            f"under {pct(bonus['target_b_coupon'])}, so the coupon pays its investors more than their risk is worth; at "
            f"the proposed ${d['b_notional']:.0f} MM the bank gives away ${mm(-proposed['bank_gain'])} MM. At "
            f"${bonus['break_even_size']:.1f} MM, {pct(bonus['target_b_coupon'])} is exactly a fair rate (Class B is "
            f"short in {pct(bonus['p_default_b_at_break_even'])} of the cases), and the bank still loses what Class A's "
            f"{pct(bonus['a_coupon'])} costs. Above about ${bonus['gain_turns_positive']:.1f} MM the gain turns positive, "
            "but only because Class B investors would then pay more than the class is worth, which a buyer using the "
            "same model would refuse.", BODY),
        Paragraph(
            "<b>Fair coupons.</b> If Class B's coupon is set to its fair rate instead (the coupon at which it is worth "
            f"exactly par), the bank loses the same ${mm(-gain_at_fair)} MM at every size, which is what Class A's "
            f"{pct(bonus['a_coupon'])} coupon costs. That holds up to about ${mm(bonus['largest_class_b_at_par'], 1)} MM, "
            "the largest Class B that any coupon can make worth par. That is a little more than the bonds less Class A "
            f"(${mm(bonus['bonds_less_class_a'])} MM), because the Lintner model gives the cases with no default a negative "
            "weight: at very high coupons the residual is worth slightly less than zero (as low as minus "
            f"${mm(-bonus['min_residual_value_in_grid'])} MM in our search), the negative values the lecture warns about. "
            "The fair rate stays at "
            f"{pct(fair[30.0]['fair_coupon'])} while Class B is never short (up to about $31 MM) and rises to "
            f"{pct(fair[50.0]['fair_coupon'])} at $50 MM and {pct(fair[70.0]['fair_coupon'])} at $70 MM.", BODY),
    ]


def leverage_rows(r: dict) -> dict:
    """The rows of the leverage analysis keyed by the cap on the relative equity risk."""
    return {row["max_relative_risk"]: row for row in r["leverage"]["risk_limits"] if row is not None}


def leverage_text(r: dict) -> Paragraph:
    lev, d = r["leverage"], r["deal"]
    given, none = lev["as_given"], lev["no_class_b"]
    rows = leverage_rows(r)
    gap = max(abs(row["b_notional"] - row["goal_seek_b_notional"]) for row in rows.values())
    loss_at_par = -r["bonus"]["fair_coupon_curve"][0]["bank_gain"]
    return Paragraph(
        "<b>Leverage.</b> So the real choice is how much of the bonds the bank funds with Class B, and with it how "
        "much risk it keeps. The Leverage sheet of our workbook looks at this from the bank's side. Unlike Table 5, "
        f"Class A is now sold for its value, ${mm(given['value_a'])} MM, not at par (with no risk and a "
        f"{pct(d['a_coupon'])} coupon, investors will pay more than its ${d['a_notional']:.0f} MM principal), which "
        f"removes the ${mm(loss_at_par)} MM loss. Class B is sold at par at its fair coupon, and the bank "
        f"pays the rest of the bonds' ${mm(lev['market_value'])} MM. That is always exactly the Lintner value of the "
        "residual it keeps, so the bank neither gains nor loses value at any size. What changes is its return and its "
        "risk. The expected IRR is the rate at which the residual's average cash flow in each quarter pays back the "
        "bank's investment, compounded to a year. The relative equity risk is the standard deviation of the "
        "residual's PV across the cases divided by the bank's investment. With no Class B the bank puts in "
        f"${mm(none['investment'])} MM for an expected IRR of {pct(none['expected_irr'])} at a relative risk of "
        f"{pct(none['relative_risk'])}; with the assignment's Class B (${d['b_notional']:.0f} MM at "
        f"{pct(d['b_coupon'])}, sold for its value of ${mm(given['value_b'])} MM rather than at par as in section 3, "
        f"where the bank puts in ${mm(r['valuation']['bank_investment'])} MM) it puts in "
        f"${mm(given['investment'])} MM for {pct(given['expected_irr'])} at {pct(given['relative_risk'])}. Both rise "
        "as Class B grows and the bank's stake shrinks (Figure 3). For a cap on the relative risk, the Leverage sheet "
        "uses Goal Seek to find the largest Class B, which also gives the highest expected IRR under that cap. In "
        "Python we solve the same problem exactly, finding Class B's fair coupon at each size by root finding and then "
        f"the size at which the relative risk equals the cap (Table 6). The Goal Seek answers are within "
        f"${gap:.2f} MM of the exact sizes; where they differ, Goal Seek stopped a little short of the cap.", BODY)


def leverage_table(r: dict) -> Table:
    rows = [["Cap on\nrelative risk", "Largest\nClass B", "Fair\ncoupon", "Class B\nshort", "Bank's\ninvestment",
             "Expected\nIRR", "IRR over\nrisk-free", "Goal Seek\nClass B"]]
    for limit, row in leverage_rows(r).items():
        rows.append([pct(limit, 0), mm(row["b_notional"]), pct(row["b_coupon"], 3), pct(row["p_default_b"]),
                     mm(row["investment"]), pct(row["expected_irr"], 2), pct(row["excess_irr"], 2),
                     mm(row["goal_seek_b_notional"])])
    return table(rows, font=8.5, first_left=False)


def conclusion_text(r: dict) -> list:
    d, bonus = r["deal"], r["bonus"]
    rows = leverage_rows(r)
    top, mid = rows[1.0], rows[0.5]
    proposed = keyed(bonus["at_target_rate"], "b_notional")[float(d["b_notional"])]
    return [
        Paragraph(
            "<b>What this means.</b> Under the Lintner model the bank cannot add value by choosing Class B's size. Once "
            "each class is sold for what it is worth, a larger Class B only swaps the bank's own money for a riskier "
            "stake, and the higher expected IRR is only the reward for that extra risk. So there is no single best structure: "
            "it depends on how much risk the bank will carry. The residual can only gain at the investors' expense, by "
            "paying a class less than its fair coupon or selling it for more than it is worth, which buyers using the "
            "same model would refuse.", BODY),
        Paragraph(
            f"<b>Our structure.</b> Keep Class A at ${d['a_notional']:.0f} MM and {pct(d['a_coupon'])}, sold for its "
            f"value of ${mm(r['leverage']['as_given']['value_a'])} MM, and size Class B to the bank's risk limit at its "
            "fair coupon. If the bank will carry a standard deviation as large as its own stake (a cap of 100%), it "
            f"should sell ${mm(top['b_notional'])} MM of Class B at {pct(top['b_coupon'], 2)}, close to the "
            f"{pct(bonus['target_b_coupon'])} target rate ({pct(bonus['target_b_coupon'])} is exactly fair at "
            f"${bonus['break_even_size']:.1f} MM). It then puts in ${mm(top['investment'])} MM and expects "
            f"{pct(top['expected_irr'], 2)} a year. With a cap of 50% it would sell ${mm(mid['b_notional'])} MM at "
            f"{pct(mid['b_coupon'], 2)} and expect {pct(mid['expected_irr'], 2)}. Either way the bank gets full value "
            f"for the classes, instead of giving investors ${mm(-proposed['bank_gain'])} MM by selling the proposed "
            "classes at par.", BODY),
    ]


def assumption_table(r: dict) -> Table:
    rows = [[Paragraph(f"<b>{heading}</b>", CELL), Paragraph(text, CELL)] for heading, text in assumptions(r)]
    flowable = Table(rows, colWidths=[1.35 * inch, AVAILABLE - 1.35 * inch])
    flowable.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, MUTED),
        ("LINEABOVE", (0, 0), (-1, 0), 0.4, MUTED),
        ("LEFTPADDING", (0, 0), (0, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return flowable


def front_matter(r: dict) -> list:
    site = PROJECT_PART2["site_url"]
    story = [
        Paragraph(PROJECT_PART2["title"], TITLE),
        Paragraph("Mini-Project 3, Part 2  |  FRE 6103 Valuation for Financial Engineering, NYU Tandon", SUBTITLE),
        Paragraph(f"Group members: {PROJECT_PART2['members_line']}", SUBTITLE),
        HRFlowable(width="100%", thickness=0.9, color=NAVY, spaceBefore=9, spaceAfter=11),
        Paragraph(f"<b>Interactive page:</b> <link href=\"{site}\"><font face=\"Courier\" size=\"8.5\">{site}</font>"
                  "</link> (change Class B's principal and coupon and the values update)", LEFT),
        Paragraph("<b>Aim:</b> To value Class A, Class B and the bank's residual with the Lintner model from class, "
                  "using the cash flows of Part 1 in 1000 cases, to find the cost of risk implied by the bonds' market "
                  "price, and (bonus) to find the structure that gives the residual the most value with Class A held at "
                  "$15 MM, and how much Class B the bank should sell for a given limit on its risk.", BODY),
        Paragraph("1. Key Assumptions", H2_SPLIT),
        assumption_table(r),
        Paragraph("2. Implementation Steps", H2),
    ]
    story += [Paragraph(f"{n}. <b>{heading}:</b> {text}", ITEM) for n, (heading, text) in enumerate(steps(r), 1)]
    return story


def body(r: dict, figures: Path) -> list:
    v = r["valuation"]
    story = [Paragraph("3. Values", H2), valuation_table(r),
             Paragraph(f"Table 1. Lintner values of the Part 2 structure, $ MM, from {r['n_cases']} cases. The max error is "
                       "two standard errors of the expected PV, as in class, from the formula for independent cases; with "
                       "moment matching that formula overstates the error (see Precision). Covariance "
                       f"with the pool is the sum of the row of Table 2. Risk charge = {v['cost_per_unit_risk']:.5f} x "
                       "covariance. For the pool row, the value per $100 column shows the bonds' market value.", CAPTION)]
    story += results_text(r)
    story += [covariance_table(r),
              Paragraph("Table 2. Covariance matrix of the three PVs across the cases. Class A and Class B have the same PV in "
                        "every case, so their rows and columns are 0.", CAPTION),
              risky_text(r), risky_table(r),
              Paragraph(f"Table 3. The same valuation with Class B at ${r['risky_example']['b_notional']:.0f} MM and "
                        f"{pct(r['risky_example']['b_coupon'])}, Class A unchanged, $ MM.", CAPTION),
              figure(figures / "pv_distributions.png", 6.4 * inch),
              Paragraph("Figure 1. PV at the risk-free rate across the 1000 cases. Left: the pool, with its expected PV and the "
                        "bonds' market value; the gap is the cost of risk. Right: the residual, with its expected PV and its "
                        "Lintner value; here the gap is the whole risk charge.", CAPTION)]
    story += [Paragraph("4. What Moves the Values", H2), sensitivity_table(r),
              Paragraph("Table 4. Values with one input of the default model changed, the bonds' market value held at its 9% "
                        "price. B short = share of cases in which Class B is paid less than it is due.", CAPTION)]
    story += sensitivity_text(r)
    story += [Paragraph("5. Bonus: Choosing Class B", H2)]
    story += bonus_text(r)[:3]
    story += [figure(figures / "class_b_choice.png", 5.6 * inch),
              Paragraph("Figure 2. Left: the bank's gain against Class B's principal, with Class B's coupon fixed at its target "
                        "rate and with it set to its fair rate. Right: Class B's fair coupon and the share of cases in which "
                        "it is short at that coupon.", CAPTION),
              bonus_table(r),
              Paragraph(f"Table 5. Class B sizes with Class A held at ${r['deal']['a_notional']:.0f} MM and "
                        f"{pct(r['deal']['a_coupon'])}. Bank's gain = cash from selling both classes at par + residual value "
                        "- bonds' market value. 'none': no coupon makes a Class B that large worth par.", CAPTION)]
    story += bonus_text(r)[3:]
    story += [leverage_text(r), figure(figures / "leverage.png", 6.4 * inch),
              Paragraph("Figure 3. The bank's side, with Class A sold for its value and Class B at its fair coupon at every "
                        "size. Left: relative equity risk (standard deviation of the residual's PV across the cases / the "
                        "bank's investment) against Class B's principal; the dots are the Goal Seek answers on the "
                        "Leverage sheet. Right: the expected IRR on the bank's investment against that risk, from no "
                        "Class B up to a cap of 120%.", CAPTION),
              leverage_table(r),
              Paragraph("Table 6. Largest Class B (at its fair coupon) for each cap on the relative equity risk, and the "
                        "bank's position there, $ MM. Expected IRR: the bank's investment against the residual's expected "
                        f"quarterly cash flows, compounded to a year; the risk-free rate is "
                        f"{pct(r['leverage']['risk_free_annual'], 2)} on the same basis. Goal Seek: the answer on the "
                        "Leverage sheet of the workbook.", CAPTION)]
    story += conclusion_text(r)
    return story


def footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(document.leftMargin, 0.55 * inch, f"FRE 6103 Mini-Project 3, Part 2: {PROJECT_PART2['title']}")
    canvas.drawRightString(letter[0] - document.rightMargin, 0.55 * inch, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


def check_report_assumptions(r: dict) -> None:
    """The wording assumes the classes are paid in full in every case at the base inputs. Stop if not."""
    if any(r["valuation"]["default_cases"]):
        raise SystemExit("The report text assumes Class A and Class B are never short at the base inputs, which is not "
                         "true for these results. Revise write_report_part2.py before using it.")


def write_pdf(r: dict, figures: Path, target: Path) -> None:
    check_report_assumptions(r)
    story = front_matter(r) + [Spacer(1, 4)] + body(r, figures)
    document = SimpleDocTemplate(str(target), pagesize=letter, leftMargin=SIDE_MARGIN, rightMargin=SIDE_MARGIN,
                                 topMargin=TOP_MARGIN, bottomMargin=BOTTOM_MARGIN,
                                 title=f"Mini Project 3 Part 2: {PROJECT_PART2['title']}", author=PROJECT_PART2["authors"])
    document.build(story, onFirstPage=footer, onLaterPages=footer)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="..", help="repository root")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    results_file = root / "output" / "part2" / "results.json"
    if not results_file.exists():
        raise SystemExit(f"missing {results_file}: run run_valuation.py first")
    r = json.loads(results_file.read_text())
    target = root / "writeup" / REPORT_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    write_pdf(r, root / "output" / "part2" / "figures", target)
    print(f"Report written to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
