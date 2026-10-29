#!/usr/bin/env python3
"""Assemble the GitHub Pages site: docs/data.js (the fixed random numbers, the Python results the
page checks itself against and the methodology), docs/valuation-data.js for the Part 2 page, and
copies of the downloads under docs/files/."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cdo import BASE_DEAL, N_CASES, PROJECT, PROJECT_PART2, SEED, load_fixed_normals  # noqa: E402
from cdo.analysis import B_NOTIONAL_GRID, LGD_GRID, PD_GRID, RHO_GRID  # noqa: E402
from write_report import REPORT_NAME, build_sections  # noqa: E402
from write_report_part2 import REPORT_NAME as REPORT_NAME_PART2  # noqa: E402
from write_report_part2 import assumptions as part2_assumptions  # noqa: E402
from write_report_part2 import steps as part2_steps  # noqa: E402

DOWNLOADS = [
    (f"writeup/{REPORT_NAME}", REPORT_NAME),
    ("excel/Miniproject3_CDO_Analysis.xlsx", "Miniproject3_CDO_Analysis.xlsx"),
    ("notebooks/Miniproject3_CDO_Analysis.ipynb", "Miniproject3_CDO_Analysis.ipynb"),
    ("data/fixed_random_numbers.csv", "fixed_random_numbers.csv"),
    ("output/case_totals.csv", "case_totals.csv"),
    ("output/pool_cash_flows.csv", "pool_cash_flows.csv"),
    (f"writeup/{REPORT_NAME_PART2}", REPORT_NAME_PART2),
    ("excel/Miniproject3_Part2_CDO_Valuation.xlsx", "Miniproject3_Part2_CDO_Valuation.xlsx"),
    ("notebooks/Miniproject3_Part2_CDO_Valuation.ipynb", "Miniproject3_Part2_CDO_Valuation.ipynb"),
    ("output/part2/case_values.csv", "case_values.csv"),
]


def site_payload(results: dict, normals: list) -> dict:
    """What the page needs: inputs, random numbers, and Python totals to check its own model against."""
    cases = results["cases"]
    return {
        "project": PROJECT,
        "deal": results["deal"],
        "seed": results["seed"],
        "market_ytm": results["market_ytm"],
        "risk_free": results["risk_free"],
        "example_case": results["example"]["case"],
        "grids": {"pd": PD_GRID, "lgd": LGD_GRID, "rho": RHO_GRID, "b_notional": B_NOTIONAL_GRID},
        "python": {
            "pool_total": [case["pool"] for case in cases],
            "equity_total": [case["equity"] for case in cases],
            "slide_value": results["checks"]["slide_value"],
            "max_quarterly_gap_pct": results["checks"]["max_quarterly_gap_pct"],
        },
        "methodology": build_sections(results),
        "z": normals,
    }


def part2_payload(r: dict) -> dict:
    """What the Part 2 page needs on top of data.js: the structure, the rates and the Python values to compare."""
    v = r["valuation"]
    return {
        "project": PROJECT_PART2,
        "seed": r["seed"],
        "deal": r["deal"],
        "risk_free": r["risk_free"],
        "market_ytm": r["market_ytm"],
        "python": {"value": v["value"], "cost_per_unit_risk": v["cost_per_unit_risk"], "market_value": v["market_value"]},
        "methodology": [{"heading": heading, "html": text} for heading, text in part2_assumptions(r) + part2_steps(r)],
        "leverage": leverage_payload(r["leverage"]),
    }


def leverage_payload(lev: dict) -> dict:
    """The bank's side of the bonus for the page: the largest Class B for each cap on the relative equity risk."""
    fields = ("b_notional", "b_coupon", "p_default_b", "investment", "expected_irr", "relative_risk")

    def pick(row, extra=()):
        return {k: row[k] for k in fields + extra}

    return {
        "risk_free_annual": lev["risk_free_annual"],
        "no_class_b": pick(lev["no_class_b"]),
        "as_given": pick(lev["as_given"]),
        "risk_limits": [pick(row, ("max_relative_risk", "goal_seek_b_notional")) for row in lev["risk_limits"] if row],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="..", help="repository root")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    docs = root / "docs"
    files = docs / "files"
    files.mkdir(parents=True, exist_ok=True)

    results = json.loads((root / "output" / "results.json").read_text())
    table, _ = load_fixed_normals(root / "data" / "fixed_random_numbers.csv", N_CASES, BASE_DEAL["n_bonds"], SEED)
    payload = site_payload(results, table.values.tolist())
    (docs / "data.js").write_text("window.CDO_DATA = " + json.dumps(payload, separators=(",", ":"), allow_nan=False) + ";\n")
    part2 = part2_payload(json.loads((root / "output" / "part2" / "results.json").read_text()))
    (docs / "valuation-data.js").write_text("window.CDO_PART2 = " + json.dumps(part2, separators=(",", ":"), allow_nan=False) + ";\n")

    for source, target in DOWNLOADS:
        src = root / source
        if not src.exists():
            raise SystemExit(f"missing deliverable: {src}")
        shutil.copy2(src, files / target)
    (docs / ".nojekyll").touch()
    print(f"Site data written to {docs} ({len(DOWNLOADS)} downloads copied)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
