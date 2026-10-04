#!/usr/bin/env python3
"""Write the client report (PDF) from output/results.json and the figures in output/figures.

    python write_report.py            # writeup/Miniproject3_CDO_Analysis_Report.pdf
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import HRFlowable, Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cdo import PROJECT  # noqa: E402

REPORT_NAME = "Miniproject3_CDO_Analysis_Report.pdf"
NAVY = colors.HexColor("#1F3864")
RULE = colors.HexColor("#999999")

SIDE_MARGIN, TOP_MARGIN, BOTTOM_MARGIN = 0.95 * inch, 0.9 * inch, 0.95 * inch
AVAILABLE = letter[0] - 2 * SIDE_MARGIN - 12         # width of the text block (the frame pads 6 pt each side)
INK = colors.HexColor("#1A1A1A")
MUTED = colors.HexColor("#5A5A5A")
STRIPE = colors.HexColor("#F3F5F8")

BODY = ParagraphStyle("body", fontName="Helvetica", fontSize=10, leading=14.5, alignment=TA_JUSTIFY, spaceAfter=7,
                      textColor=INK)
LEFT = ParagraphStyle("left", parent=BODY, alignment=0)
TITLE = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=21, leading=25, spaceAfter=5, textColor=NAVY)
SUBTITLE = ParagraphStyle("subtitle", fontName="Helvetica", fontSize=10.5, leading=15, spaceAfter=1, textColor=MUTED)
H2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=13, leading=16, spaceBefore=16, spaceAfter=7,
                    textColor=NAVY, keepWithNext=1)
CAPTION = ParagraphStyle("caption", fontName="Helvetica-Oblique", fontSize=8.8, leading=11.5, alignment=TA_CENTER,
                         spaceBefore=6, spaceAfter=14, textColor=MUTED)
ITEM = ParagraphStyle("item", parent=LEFT, leftIndent=18, firstLineIndent=-18, spaceAfter=5)
H2_SPLIT = ParagraphStyle("h2split", parent=H2, keepWithNext=0)      # before a table that may break over a page
BULLET = ParagraphStyle("bullet", parent=LEFT, leftIndent=16, bulletIndent=4, spaceAfter=5)
CELL = ParagraphStyle("cell", parent=LEFT, fontSize=9.5, leading=13, spaceAfter=0)


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def mm(x: float, digits: int = 2) -> str:
    return f"{x:,.{digits}f}"


def keyed(rows: list, key: str) -> dict:
    """Sensitivity rows keyed by the value of the input that was varied."""
    return {row[key]: row for row in rows}


def table(rows: list, widths: list, font: float = 8.5, first_left: bool = True, pad: float = 3.5) -> Table:
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 1), (-1, -1), 0.4, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), pad),
        ("BOTTOMPADDING", (0, 0), (-1, -1), pad),
    ]
    if first_left:
        style.append(("ALIGN", (0, 1), (0, -1), "LEFT"))
    total = sum(widths)
    if total > AVAILABLE:                                # shrink to the text block
        widths = [w * AVAILABLE / total for w in widths]
    style.append(("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, STRIPE]))
    flowable = Table(rows, colWidths=widths, repeatRows=1)
    flowable.setStyle(TableStyle(style))
    flowable.keepWithNext = 1                            # a table stays with its caption
    return flowable


def figure(path: Path, width: float) -> Image:
    width = min(width, AVAILABLE)
    pixel_width, pixel_height = ImageReader(str(path)).getSize()
    image = Image(str(path), width=width, height=width * pixel_height / pixel_width)
    image.keepWithNext = 1                               # a figure stays with its caption
    return image


def assumptions(results: dict) -> list:
    """(heading, text) pairs for the Key Assumptions block."""
    d, promised = results["deal"], results["promised"]
    n_periods = results["n_periods"]
    recovered = 1.0 - d["lgd"]
    return [
        ("Collateral",
         f"{d['n_bonds']} bonds, ${d['face']:.0f} MM face each, {d['years']} years, {pct(d['coupon'], 0)} annual coupon "
         f"paid quarterly. Each bond promises ${promised['bond'][0]:.2f} MM per quarter and ${promised['bond'][-1]:.2f} MM "
         f"in quarter {n_periods}, so the pool promises ${promised['pool'][0]:.1f} MM per quarter, "
         f"${promised['pool'][-1]:.1f} MM at maturity and ${promised['totals']['pool']:.0f} MM in total."),
        ("Default Model",
         f"We use the BIS debt model from class. (1 - LGD) = {pct(recovered, 0)} of every promised payment is "
         f"non-defaultable and is always paid. The other {pct(d['lgd'], 0)} stops in the quarter of default. So a "
         f"defaulted bond still pays {pct(recovered, 0)} of its coupons and {pct(recovered, 0)} of its principal, on "
         "the original dates."),
        ("Default Timing",
         f"Each bond has a {pct(d['pd'], 0)} annual default probability. Its default time in years comes from the "
         f"formula in class, t = ln(1 - u) / ln(1 - {d['pd']:.2f}), and the bond defaults in quarter ceil(4t). Past "
         f"quarter {n_periods} it survives."),
        ("Correlation",
         f"We apply the {d['rho']:.2f} correlation to the normal random numbers behind the default times (Cholesky "
         f"factor of a {d['n_bonds']} x {d['n_bonds']} matrix with {d['rho']:.2f} off the diagonal) and then take "
         "u = N(x)."),
        ("Random Numbers",
         f"{results['n_cases']} cases x {d['n_bonds']} bonds of standard normals, generated once (seed {results['seed']}) "
         f"and stored in fixed_random_numbers.csv. Cases are numbered 1 to {results['n_cases']} and every run uses the "
         "same numbers. Before use they are de-meaned and moment matched as shown in class, so that across the "
         f"{results['n_cases']} cases each bond's numbers have mean 0 and variance 1 and are uncorrelated with the "
         "other bonds' numbers."),
        ("Classes",
         f"Class A: ${d['a_notional']:.0f} MM, {pct(d['a_coupon'], 0)} coupon. Class B: ${d['b_notional']:.0f} MM, "
         f"{pct(d['b_coupon'], 0)} coupon. We treat both as bullet bonds with quarterly coupons and principal at year "
         f"{d['years']}, since the collateral only returns principal at maturity. Class A is owed "
         f"${promised['class_a'][0]:.2f} MM per quarter and ${promised['class_a'][-1]:.2f} MM at maturity, Class B "
         f"${promised['class_b'][0]:.2f} MM and ${promised['class_b'][-1]:.2f} MM."),
        ("Waterfall",
         "Each quarter Class A is paid first, then Class B, and the residual goes to the bank. No carry-forwards: a "
         "shortfall is not made up later and no cash is held back."),
        ("Units",
         f"$ millions. Totals are undiscounted sums over the {n_periods} quarters. The {pct(results['market_ytm'], 0)} "
         f"YTM and {pct(results['risk_free'], 0)} risk-free rate are not used until Part 2."),
    ]


def steps(results: dict) -> list:
    """(heading, text) pairs for the Implementation Steps block."""
    c, d = results["checks"], results["deal"]
    recovered = 1.0 - d["lgd"]
    drawn = results["moment_matching"]["raw"]
    return [
        ("BIS Bond Function",
         "Given the period a bond defaults in, the function returns the promised cash flows before it and "
         "(1 - LGD) x promised from it onward. We checked it against the Lecture 5 slide (face 1000, 5.5%, LGD 60%, "
         f"pi 3%, r 4%) and it gives {c['slide_value']:.2f}, the same as the slide."),
        ("Fixed Random Numbers",
         f"Read the {results['n_cases']} x {d['n_bonds']} table of independent normals from the csv file (it is "
         "created from the seed if it is missing)."),
        ("Moment Matching",
         "Subtract each column's mean, compute the covariance matrix of the draws (dividing by the number of "
         "cases, bonds in the order 1 to 10) and its Cholesky factor, and multiply each case's de-meaned draws by the "
         "inverse of that factor. As drawn, the column means were up to "
         f"{drawn['max_abs_mean']:.3f} away from 0, the variances ran from {drawn['min_variance']:.2f} to "
         f"{drawn['max_variance']:.2f} and two bonds' numbers were correlated by up to "
         f"{drawn['max_abs_correlation']:.2f}. After matching the means are 0, the variances are 1 and the "
         "correlations are 0. Only these moments of the normal draws are fixed, so default rates and cash flows "
         "are still estimates."),
        ("Correlated Defaults",
         "Multiply the matched numbers by the Cholesky factor of the correlation matrix, convert to uniforms, then "
         "to default times and default quarters. Because of the moment matching the correlated normals have a "
         f"pairwise correlation of exactly {c['corr_normals']:.2f} across the cases. For the default times it comes "
         f"out at {c['corr_default_times']:.3f}."),
        ("Collateral Cash Flows",
         f"Apply the BIS function to every case and bond ({results['n_cases']} x {d['n_bonds']} x {results['n_periods']} "
         "array) and add up the bonds to get the pool."),
        ("Check",
         "We compare the simulated mean pool cash flow in each quarter with the exact expected value, promised x "
         f"[{recovered:.1f} + {d['lgd']:.1f} x {1 - d['pd']:.2f}^(k/4)]. The largest gap over the "
         f"{results['n_periods']} quarters is {c['max_quarterly_gap_pct']:.2f}%."),
        ("Waterfall",
         "A = min(pool, due to A), B = min(pool - A, due to B), equity = pool - A - B, quarter by quarter. The three "
         "always add back to the pool."),
        ("Case Viewer",
         f"Setting CASE in the notebook, the case cell in the Excel workbook or the case box on the dashboard "
         f"(1 to {results['n_cases']}) shows that case's random numbers, default quarters and quarterly cash flows."),
        ("Statistics and Sensitivities",
         "Distributions of total and quarterly cash flows. Then we rerun the model on the same random numbers with "
         "the default probability, LGD, correlation and Class B size changed one at a time."),
        ("Cross-checks",
         "We also built the model with live formulas in Excel and in JavaScript for the dashboard, and wrote a plain "
         f"loop version for the tests. All three give the same cash flows as the Python model in all "
         f"{results['n_cases']} cases."),
    ]


def results_summary(results: dict) -> str:
    c = results["checks"]
    s = {row["series"]: row for row in results["summary"]}
    pool, equity = s["pool"], s["equity"]
    zero_default_share = results["default_distribution"][0]["simulated probability"]
    return (
        f"Across the {results['n_cases']} cases an average of {c['avg_defaults_simulated']:.2f} of the "
        f"{results['n_bonds']} bonds default within 5 years (theory {c['avg_defaults_theory']:.2f}), and "
        f"{pct(zero_default_share)} of cases have no default at all. The pool collects ${mm(pool['mean'], 1)} MM on "
        f"average out of the ${pool['no-default amount']:.0f} MM promised ({pct(pool['mean / no-default amount'])}; "
        f"the exact expected value is ${mm(c['pool_total_expected'], 1)} MM), with a standard "
        f"deviation of ${mm(pool['std dev'], 1)} MM, a 5th percentile of ${mm(pool['5th pct'], 1)} MM and a worst case "
        f"of ${mm(pool['min'], 1)} MM (case {c['worst_case']}, {c['worst_case_defaults']} defaults). "
        "Class A and Class B are paid in full in every case. This is not luck in the sample: with a "
        f"{pct(results['deal']['lgd'], 0)} LGD the pool can never pay less than {pct(1 - results['deal']['lgd'], 0)} of "
        f"what it promised, which is ${c['floor_per_quarter']:.2f} MM per quarter and ${c['floor_at_maturity']:.2f} MM "
        f"at maturity, and the two classes together need only ${c['classes_need_per_quarter']:.2f} MM and "
        f"${c['classes_need_at_maturity']:.2f} MM. Under the BIS model with these inputs both classes are effectively "
        "free of default risk, so all of the variability in the collateral passes to the bank's equity. Equity "
        f"receives ${mm(equity['mean'], 1)} MM on average against ${equity['no-default amount']:.0f} MM if nothing defaults, "
        f"with the same standard deviation as the pool (${mm(equity['std dev'], 1)} MM), a 5th percentile of "
        f"${mm(equity['5th pct'], 1)} MM and a minimum of ${mm(equity['min'], 1)} MM. Correlation matters for the "
        f"tails: {pct(c['p_four_or_more_simulated'])} of cases have 4 or more defaults "
        f"({pct(c['p_four_or_more_exact'])} exactly under the model), against "
        f"{pct(c['p_four_or_more_independent'])} if the bonds defaulted independently.")


def sensitivity_notes(results: dict) -> list:
    """(heading, text) pairs explaining what each input does to the equity and the classes."""
    sens, c = results["sensitivities"], results["checks"]
    by_pd, by_lgd, by_rho = keyed(sens["pd"], "pd"), keyed(sens["lgd"], "lgd"), keyed(sens["rho"], "rho")
    low, high = by_pd[0.01], by_pd[0.12]
    per_point = (by_pd[0.02]["equity mean"] - by_pd[0.06]["equity mean"]) / 4
    return [
        ("Default probability",
         "the largest effect on the average. Going from 1% to 12% per year takes mean equity cash from "
         f"${mm(low['equity mean'], 1)} MM to ${mm(high['equity mean'], 1)} MM and the 5th percentile from "
         f"${mm(low['equity 5th pct'], 1)} MM to ${mm(high['equity 5th pct'], 1)} MM. Around the base case each extra "
         f"1% of annual default probability costs equity roughly ${per_point:.1f} MM."),
        ("LGD",
         "the loss scales in a straight line. Every 10 points of LGD lowers mean equity cash by "
         f"${by_lgd[0.5]['equity mean'] - by_lgd[0.6]['equity mean']:.1f} MM and the 5th percentile by "
         f"${by_lgd[0.5]['equity 5th pct'] - by_lgd[0.6]['equity 5th pct']:.1f} MM. LGD is also the input that decides "
         "whether the classes are at risk. Class B is fully covered as long as the worst case pool at maturity, "
         f"{results['promised']['pool'][-1]:.1f} x (1 - LGD), is at least {c['classes_need_at_maturity']:.1f}, that is "
         f"up to an LGD of {pct(c['safe_lgd']['class_b'])}, and Class A up to {pct(c['safe_lgd']['class_a'])}. In the "
         f"simulation Class B shortfalls first appear at LGD 80% ({pct(by_lgd[0.8]['P(B shortfall)'])} of cases) "
         f"and stand at {pct(by_lgd[0.9]['P(B shortfall)'])} at LGD 90% and {pct(by_lgd[1.0]['P(B shortfall)'])} at "
         "LGD 100%."),
        ("Correlation",
         f"it does not change the average (mean equity stays near ${mm(by_rho[0.2]['equity mean'], 0)} MM, since each "
         "bond's own default probability is unchanged) but it widens the distribution. From a correlation of 0 to "
         f"0.8 the standard deviation of equity cash rises from ${mm(by_rho[0.0]['equity std'], 1)} MM to "
         f"${mm(by_rho[0.8]['equity std'], 1)} MM and the 5th percentile falls from "
         f"${mm(by_rho[0.0]['equity 5th pct'], 1)} MM to ${mm(by_rho[0.8]['equity 5th pct'], 1)} MM, while the chance "
         f"of no defaults at all rises from {pct(by_rho[0.0]['P(no default)'])} to {pct(by_rho[0.8]['P(no default)'])}."),
    ]


def build_sections(results: dict) -> list:
    """The methodology as plain paragraphs, for the project page."""
    blocks = assumptions(results) + steps(results)
    return [{"heading": heading, "html": text} for heading, text in blocks]


def assumption_table(results: dict) -> Table:
    """The key assumptions as a two-column table: what it is about, then the assumption."""
    rows = [[Paragraph(f"<b>{heading}</b>", CELL), Paragraph(text, CELL)] for heading, text in assumptions(results)]
    flowable = Table(rows, colWidths=[1.25 * inch, AVAILABLE - 1.25 * inch])
    flowable.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#D5D9E0")),
        ("LINEABOVE", (0, 0), (-1, 0), 0.4, colors.HexColor("#D5D9E0")),
        ("LEFTPADDING", (0, 0), (0, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return flowable


def footer(canvas, document) -> None:
    """Short title on the left and the page number on the right of every page."""
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(document.leftMargin, 0.55 * inch, f"FRE 6103 Mini-Project 3: {PROJECT['title']}")
    canvas.drawRightString(letter[0] - document.rightMargin, 0.55 * inch, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


def front_matter(results: dict) -> list:
    site = PROJECT["site_url"]
    story = [
        Paragraph(PROJECT["title"], TITLE),
        Paragraph("Mini-Project 3, Part 1  |  FRE 6103 Valuation for Financial Engineering, NYU Tandon", SUBTITLE),
        Paragraph(f"Group members: {PROJECT['members_line']}", SUBTITLE),
        HRFlowable(width="100%", thickness=0.9, color=NAVY, spaceBefore=9, spaceAfter=11),
        Paragraph(f"<b>Interactive dashboard:</b> <link href=\"{site}\"><font face=\"Courier\" size=\"8.5\">{site}</font>"
                  "</link> (the same model in the browser: pick any of the 1000 cases or change the inputs, and the "
                  "cash flows, distributions and sensitivities update)", LEFT),
        Paragraph("<b>Aim:</b> To build a simulation model of a CDO backed by 10 speculative grade corporate bonds, "
                  "with correlated default dates drawn from fixed random numbers, to distribute the pool's quarterly "
                  "cash flows to Class A, Class B and the bank's equity through the waterfall, and to describe the "
                  "resulting cash flows statistically, including their sensitivity to the main inputs. Valuation of "
                  "the classes is Part 2.", BODY),
    ]
    story += [Paragraph("1. Key Assumptions", H2_SPLIT), assumption_table(results)]
    story.append(Paragraph("2. Implementation Steps", H2))
    story += [Paragraph(f"{n}. <b>{heading}:</b> {text}", ITEM) for n, (heading, text) in enumerate(steps(results), 1)]
    story += [Paragraph("3. Results Summary", H2), Paragraph(results_summary(results), BODY), Spacer(1, 6)]
    return story


def results_pages(results: dict, figures: Path) -> list:
    names = {"pool": "Collateral pool", "class_a": "Class A", "class_b": "Class B", "equity": "Equity (bank)"}
    check = results["sampling_check"]
    rows = [["", "No-default\namount", "Mean", "Std error\nof the mean", "Std dev", "5th pct", "Median", "Worst case",
             "Mean / no-\ndefault amount"]]
    for r in results["summary"]:
        rows.append([names[r["series"]], mm(r["no-default amount"]), mm(r["mean"]), mm(r["std error"]), mm(r["std dev"]),
                     mm(r["5th pct"]), mm(r["median"]), mm(r["min"]), pct(r["mean / no-default amount"])])
    story = [
        table(rows, [1.2 * inch, 0.8 * inch, 0.65 * inch, 0.8 * inch, 0.65 * inch, 0.65 * inch, 0.65 * inch,
                     0.72 * inch, 0.88 * inch]),
        Paragraph("Table 1. Total cash received over the 5 years by the collateral pool and by each class, $ MM, "
                  f"undiscounted, across the {results['n_cases']} cases. The no-default amount is what each would "
                  "receive if no bond defaulted. The standard error is the standard deviation divided by the square "
                  "root of the number of cases, the usual formula for independent cases. Moment matching ties the "
                  f"cases together, so the formula overstates the error of the pool and equity means: across {check['n_tables']:,} fresh sets of "
                  f"random numbers the mean pool cash had a standard deviation of ${check['sd_of_mean_matched']:.2f} MM "
                  f"with moment matching and ${check['sd_of_mean_as_drawn']:.2f} MM without.", CAPTION),
        figure(figures / "total_cash_distributions.png", 5.6 * inch),
        Paragraph("Figure 1. Distribution of total 5-year cash from the collateral pool (left) and to the bank's "
                  "equity (right). The equity distribution is the pool distribution shifted down by the "
                  f"${results['promised']['totals']['class_a'] + results['promised']['totals']['class_b']:.0f} MM paid "
                  "to Classes A and B.", CAPTION),
    ]
    rows = [["Defaults in 5 years", "Cases", "Simulated", "Exact (model)", "If independent", "Avg pool cash",
             "Avg equity cash"]]
    for r in results["default_distribution"]:
        has_cases = r["cases"] > 0
        rows.append([str(r["defaults in 5 years"]), str(r["cases"]), pct(r["simulated probability"]),
                     pct(r["exact with correlation"], 2), pct(r["if independent (binomial)"], 2),
                     mm(r["avg pool cash ($MM)"]) if has_cases else "-",
                     mm(r["avg equity cash ($MM)"]) if has_cases else "-"])
    per_default = results["default_distribution"][0]["avg pool cash ($MM)"] - results["default_distribution"][1]["avg pool cash ($MM)"]
    story += [
        table(rows, [1.25 * inch, 0.65 * inch, 0.9 * inch, 1.0 * inch, 1.05 * inch, 1.05 * inch, 1.1 * inch],
              first_left=False, font=8.2, pad=1.2),
        Paragraph("Table 2. Number of bonds defaulting within 5 years: share of the simulated cases, the exact "
                  "probability under the model (common factor, no random numbers) and the binomial if defaults were "
                  "independent, with the average cash ($ MM) in cases with that many defaults. Each default costs the "
                  f"pool about ${per_default:.0f} MM.", CAPTION),
        figure(figures / "default_count.png", 4.1 * inch),
        Paragraph(f"Figure 2. Distribution of the number of defaults. With correlation {results['deal']['rho']:.2f} "
                  "both ends are more likely than under independence: more cases with no defaults and more cases with "
                  "5 or more. The dots are the exact probabilities under the model, which the 1000 cases track closely.",
                  CAPTION),
        figure(figures / "quarterly_cash_flows.png", 4.5 * inch),
    ]
    q = results["quarterly"]
    story.append(Paragraph(
        f"Figure 3. Quarterly coupon cash flows in quarters 1 to {len(q) - 1}. The mean pool cash flow falls from "
        f"${q[0]['pool mean']:.2f} MM to ${q[-2]['pool mean']:.2f} MM as defaults build up, and equity receives the "
        f"pool cash less the ${results['checks']['classes_need_per_quarter']:.2f} MM due to the classes. Even if every "
        "bond defaulted in quarter 1 the pool would pay "
        f"{results['checks']['floor_per_quarter'] / results['checks']['classes_need_per_quarter']:.0f} times what the "
        "classes need. At maturity (not shown) "
        f"the pool pays ${q[-1]['pool mean']:.1f} MM on average out of ${q[-1]['pool promised']:.1f} MM, of which "
        f"equity receives ${q[-1]['equity mean']:.1f} MM.", CAPTION))
    return story


def sensitivity_page(results: dict, figures: Path) -> list:
    story = [
        Paragraph("4. Sensitivities", H2),
        Paragraph("We changed each input on its own, keeping the others at their base values and using the same "
                  "fixed random numbers. Because the classes are fully covered in the base case, the sensitivities "
                  "show up in the equity. Pool cash is equity cash plus what the two classes are paid, so while both classes "
                  "are paid in full the pool figures are the equity figures plus "
                  f"${results['promised']['totals']['class_a'] + results['promised']['totals']['class_b']:.0f} MM, "
                  "with the same standard deviation.", BODY),
    ]
    story += [Paragraph(f"<b>{heading}:</b> {text}", LEFT) for heading, text in sensitivity_notes(results)]
    story.append(Spacer(1, 5))

    deal = results["deal"]
    blocks = [("Default prob.", "pd", lambda v: pct(v, 0)), ("LGD", "lgd", lambda v: pct(v, 0)),
              ("Correlation", "rho", lambda v: f"{v:.1f}")]
    rows = [["Input", "Value", "Avg defaults", "P(no default)", "Equity mean", "Equity std", "Equity 5th pct",
             "Equity min", "P(B short)"]]
    for label, key, fmt in blocks:
        for r in results["sensitivities"][key]:
            is_base = abs(r[key] - deal[key]) < 1e-12
            rows.append([label, fmt(r[key]) + (" (base)" if is_base else ""), f"{r['avg defaults']:.2f}",
                         pct(r["P(no default)"]), mm(r["equity mean"]), mm(r["equity std"]), mm(r["equity 5th pct"]),
                         mm(r["equity min"]), pct(r["P(B shortfall)"])])
    story += [
        table(rows, [0.95 * inch, 0.9 * inch] + [0.78 * inch] * 7, font=8, pad=1.3),
        Paragraph(f"Table 3. One-at-a-time sensitivities on the same {results['n_cases']} cases. Equity figures are "
                  "total 5-year cash in $ MM. P(B short) is the share of cases in which Class B is paid less than it "
                  "is due in any quarter.", CAPTION),
        figure(figures / "sensitivities.png", 5.0 * inch),
        Paragraph("Figure 4. Mean, 5th percentile and worst case of total equity cash as each input is varied.", CAPTION),
    ]
    return story


def class_risk_page(results: dict) -> list:
    stress, c, deal = results["stress"], results["checks"], results["deal"]
    by_pd, by_rho = keyed(stress["pd"], "pd"), keyed(stress["rho"], "rho")
    promised = results["promised"]
    # with nothing recovered, how many bonds must survive to pay each class at maturity
    survivors_a = math.ceil(promised["class_a"][-1] / promised["bond"][-1] - 1e-12)
    survivors_b = math.ceil((promised["class_a"][-1] + promised["class_b"][-1]) / promised["bond"][-1] - 1e-12)
    story = [
        Paragraph("5. When do the classes become risky?", H2),
        Paragraph(
            "With the base LGD no combination of default probability and correlation can touch Class A or Class B, "
            f"so for the runs below we set the LGD to {pct(stress['lgd'], 0)} (nothing recovered on a defaulted bond). Each "
            f"surviving bond pays ${promised['bond'][-1]:.2f} MM at maturity, so Class B needs at least {survivors_b} "
            f"of the {deal['n_bonds']} bonds to survive to be paid in full and Class A needs at least {survivors_a}. "
            "At the base default "
            f"probability and correlation this fails for Class B in only {pct(by_pd[0.04]['P(B shortfall)'])} of "
            f"cases, but it rises quickly with either input: to {pct(by_pd[0.12]['P(B shortfall)'])} at a 12% default "
            f"probability and to {pct(by_rho[0.8]['P(B shortfall)'])} at a correlation of 0.8. At the base default "
            f"probability with independent defaults there are no shortfalls in any of the {results['n_cases']} "
            "cases, so at that default rate the risk to the classes comes from the correlation. These "
            f"are tail events, so {results['n_cases']} cases measure them roughly: the base row rests on "
            f"{round(by_pd[0.04]['P(B shortfall)'] * results['n_cases'])} cases for Class B and "
            f"{round(by_pd[0.04]['P(A shortfall)'] * results['n_cases'])} for Class A, and the exact probabilities "
            f"under the model are {pct(c['stress_exact_p_b_shortfall'], 2)} and "
            f"{pct(c['stress_exact_p_a_shortfall'], 2)}.", BODY),
    ]
    rows = [["Input", "Value", "Avg defaults", "Equity mean", "Equity 5th pct", "P(A shortfall)", "P(B shortfall)",
             "B paid / due"]]
    for label, key, fmt in (("Default prob.", "pd", lambda v: pct(v, 0)), ("Correlation", "rho", lambda v: f"{v:.1f}")):
        for r in stress[key]:
            rows.append([label, fmt(r[key]), f"{r['avg defaults']:.2f}", mm(r["equity mean"]), mm(r["equity 5th pct"]),
                         pct(r["P(A shortfall)"]), pct(r["P(B shortfall)"]), pct(r["B paid / due"])])
    story += [
        table(rows, [1.0 * inch, 0.7 * inch] + [0.95 * inch] * 6),
        Paragraph(f"Table 4. Stress runs with LGD = {pct(stress['lgd'], 0)}. The default probability rows use "
                  f"correlation {deal['rho']:.2f} and the correlation rows use a {pct(deal['pd'], 0)} default "
                  "probability.", CAPTION),
    ]

    safe_b = (c["floor_at_maturity"] - results["promised"]["class_a"][-1]) / (1 + deal["b_coupon"] / deal["freq"])
    by_size = keyed(results["sensitivities"]["b_notional"], "b_notional")
    story.append(Paragraph(
        "The size of Class B is the other lever. In the worst case the pool pays "
        f"${c['floor_at_maturity']:.1f} MM at maturity and Class A takes ${results['promised']['class_a'][-1]:.1f} MM, "
        f"so Class B stays fully covered by recoveries alone up to a notional of about ${safe_b:.1f} MM, "
        f"{safe_b / deal['b_notional']:.1f} times its proposed size. At ${mm(30, 0)} MM it is short in {pct(by_size[30.0]['P(B shortfall)'])} of cases and at "
        f"${mm(60, 0)} MM in {pct(by_size[60.0]['P(B shortfall)'])}. Each extra $1 MM of Class B takes about "
        f"${(by_size[10.0]['equity mean'] - by_size[20.0]['equity mean']) / 10:.2f} MM out of the equity cash (its "
        "principal plus five years of coupons).", BODY))
    rows = [["Class B notional", "Equity mean", "Equity 5th pct", "Equity min", "P(A shortfall)", "P(B shortfall)",
             "B paid / due"]]
    for r in results["sensitivities"]["b_notional"]:
        is_base = abs(r["b_notional"] - deal["b_notional"]) < 1e-12
        rows.append([f"${r['b_notional']:.0f} MM" + (" (base)" if is_base else ""), mm(r["equity mean"]),
                     mm(r["equity 5th pct"]), mm(r["equity min"]), pct(r["P(A shortfall)"]), pct(r["P(B shortfall)"]),
                     pct(r["B paid / due"], 2)])
    story += [
        table(rows, [1.35 * inch] + [0.95 * inch] * 6),
        Paragraph(f"Table 5. Class B notional varied with all other inputs at base (LGD {pct(deal['lgd'], 0)}). Equity "
                  "figures are total 5-year cash in $ MM.", CAPTION),
    ]
    return story + client_points(results, safe_b)


def client_points(results: dict, safe_b: float) -> list:
    c = results["checks"]
    equity = next(row for row in results["summary"] if row["series"] == "equity")
    by_rho = keyed(results["sensitivities"]["rho"], "rho")
    points = [
        "Class A and Class B are covered by the recovery value of the collateral alone. As long as recoveries are at "
        f"least {pct(1 - c['safe_lgd']['class_b'], 0)} of promised payments (LGD of {pct(c['safe_lgd']['class_b'], 0)} "
        "or less), they are paid in full whatever the default experience.",
        "The bank's retained equity carries all of the default risk. Its expected cash is about "
        f"{pct(equity['mean / no-default amount'], 0)} of the no-default amount, and in 1 case out of 20 it receives "
        f"${mm(equity['5th pct'], 0)} MM or less out of ${equity['no-default amount']:.0f} MM.",
        "Over the ranges we tested, the default probability moves the equity result the most. Correlation leaves the "
        f"average alone but lowers the 5th percentile from ${mm(by_rho[0.0]['equity 5th pct'], 0)} MM to "
        f"${mm(by_rho[0.8]['equity 5th pct'], 0)} MM between 0 and 0.8. The LGD assumption is the one to watch for "
        "the classes.",
        f"The structure is conservative: Class B could be about ${safe_b:.0f} MM instead of "
        f"${results['deal']['b_notional']:.0f} MM and still be fully covered if every bond defaulted.",
        "These are cash flow results only. What the classes and the equity are worth, given the "
        f"{pct(results['market_ytm'], 0)} market yield on the collateral and the {pct(results['risk_free'], 0)} "
        "risk-free rate, is the subject of Part 2.",
    ]
    return [Paragraph("6. Points for the client", H2)] + [Paragraph("<bullet>&bull;</bullet>" + text, BULLET) for text in points]


def appendix(results: dict, figures: Path) -> list:
    story = [PageBreak(), Paragraph(f"Appendix A. Quarterly cash flows across the {results['n_cases']} cases ($ MM)", H2)]
    rows = [["Quarter", "Pool promised", "Pool exp. (exact)", "Pool mean", "Pool 5th pct", "Pool 95th pct", "Class A",
             "Class B", "Equity mean", "Equity 5th pct"]]
    for r in results["quarterly"]:
        rows.append([str(r["quarter"]), mm(r["pool promised"]), mm(r["pool expected (exact)"], 3), mm(r["pool mean"], 3),
                     mm(r["pool 5th pct"]), mm(r["pool 95th pct"]), mm(r["Class A mean"]), mm(r["Class B mean"]),
                     mm(r["equity mean"], 3), mm(r["equity 5th pct"])])
    widths = [0.55 * inch, 0.85 * inch, 0.95 * inch] + [0.76 * inch] * 2 + [0.8 * inch] + [0.6 * inch] * 2 + [0.78 * inch, 0.82 * inch]
    story += [
        table(rows, widths, font=7.6, first_left=False, pad=1.6),
        Paragraph("Class A and Class B receive the same amount in every case, so only one column is shown for each. "
                  "The pipeline also writes this table to output/quarterly_cash_flow_summary.csv.", CAPTION),
    ]

    example = results["example"]
    defaults = ", ".join(f"bond {b} in quarter {q}" for b, q in enumerate(example["default_quarters"], 1) if q)
    story += [
        PageBreak(),
        Paragraph(f"Appendix B. Example of a single case (case {example['case']})", H2),
        Paragraph(f"In case {example['case']}, {example['n_defaults']} bonds default: {defaults}. The pool, class and "
                  "equity cash flows are below. Any other case can be shown by changing CASE in the notebook, the case "
                  "cell in the Excel workbook or the case box on the dashboard.", LEFT),
        Spacer(1, 4),
        figure(figures / "case_waterfall.png", 6.6 * inch),
        Paragraph(f"Figure 5. Case {example['case']}: who receives the pool's cash each quarter. The classes take the "
                  "same amount every quarter and the equity absorbs each default as it happens.", CAPTION),
    ]
    rows = [["Quarter", "Bonds in default", "Pool", "Class A", "Class B", "Equity"]]
    for k in range(results["n_periods"]):
        in_default = sum(1 for q in example["default_quarters"] if q and q <= k + 1)
        rows.append([str(k + 1), str(in_default), mm(example["pool"][k]), mm(example["class_a"][k]),
                     mm(example["class_b"][k]), mm(example["equity"][k])])
    rows.append(["Total", "", mm(sum(example["pool"])), mm(sum(example["class_a"])), mm(sum(example["class_b"])),
                 mm(sum(example["equity"]))])
    case_table = table(rows, [0.8 * inch, 1.3 * inch] + [0.95 * inch] * 4, font=7.6, first_left=False, pad=1.6)
    case_table.setStyle(TableStyle([("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold")]))
    story.append(case_table)
    return story


def check_report_assumptions(results: dict) -> None:
    """The wording of the report assumes both classes are paid in full at the base inputs. Stop if not."""
    c = results["checks"]
    if c["class_a_shortfall_cases"] or c["class_b_shortfall_cases"]:
        raise SystemExit("The report text assumes Class A and Class B are paid in full in every case at the base "
                         "inputs, which is not true for these results. Revise write_report.py before using it.")


def write_pdf(results: dict, figures: Path, target: Path) -> None:
    check_report_assumptions(results)
    story = front_matter(results) + results_pages(results, figures) + sensitivity_page(results, figures)
    story += class_risk_page(results) + appendix(results, figures)
    document = SimpleDocTemplate(str(target), pagesize=letter, leftMargin=SIDE_MARGIN, rightMargin=SIDE_MARGIN,
                                 topMargin=TOP_MARGIN, bottomMargin=BOTTOM_MARGIN,
                                 title=f"Mini Project 3: {PROJECT['title']}", author=PROJECT["authors"])
    document.build(story, onFirstPage=footer, onLaterPages=footer)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="..", help="repository root")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    results_file = root / "output" / "results.json"
    if not results_file.exists():
        raise SystemExit(f"missing {results_file}: run run_analysis.py first")
    results = json.loads(results_file.read_text())
    target = root / "writeup" / REPORT_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    write_pdf(results, root / "output" / "figures", target)
    print(f"Report written to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
