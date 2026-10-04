#!/usr/bin/env python3
"""Assemble the GitHub Pages site: docs/data.js (the fixed random numbers, the Python results the
page checks itself against and the methodology) and copies of the downloads under docs/files/."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cdo import BASE_DEAL, N_CASES, PROJECT, SEED, load_fixed_normals  # noqa: E402
from cdo.analysis import B_NOTIONAL_GRID, LGD_GRID, PD_GRID, RHO_GRID  # noqa: E402
from write_report import REPORT_NAME, build_sections  # noqa: E402

DOWNLOADS = [
    (f"writeup/{REPORT_NAME}", REPORT_NAME),
    ("excel/Miniproject3_CDO_Analysis.xlsx", "Miniproject3_CDO_Analysis.xlsx"),
    ("notebooks/Miniproject3_CDO_Analysis.ipynb", "Miniproject3_CDO_Analysis.ipynb"),
    ("data/fixed_random_numbers.csv", "fixed_random_numbers.csv"),
    ("output/case_totals.csv", "case_totals.csv"),
    ("output/pool_cash_flows.csv", "pool_cash_flows.csv"),
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
