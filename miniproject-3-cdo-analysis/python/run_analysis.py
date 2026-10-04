#!/usr/bin/env python3
"""Run the full Miniproject 3 (Part 1) pipeline: load the fixed random numbers, simulate
the collateral bonds and the waterfall, and write tables, JSON and figures.

    python run_analysis.py --data ../data/fixed_random_numbers.csv --out ../output
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cdo import (  # noqa: E402
    BASE_DEAL,
    N_CASES,
    SEED,
    build_results,
    load_fixed_normals,
    moment_match,
    sampling_error_check,
    simulate,
)
from cdo.report import (  # noqa: E402
    apply_chart_style,
    plot_case,
    plot_default_count,
    plot_quarterly,
    plot_sensitivities,
    plot_total_cash,
    save_figure,
    write_json,
)

DEFAULT_EXAMPLE_CASE = 5


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="../data/fixed_random_numbers.csv", help="fixed random numbers (created if missing)")
    parser.add_argument("--out", default="../output", help="directory for CSV/JSON/PNG outputs")
    parser.add_argument("--case", type=int, default=DEFAULT_EXAMPLE_CASE, help="case shown as the worked example (1-1000)")
    return parser.parse_args(argv)


def sensitivity_frame(results: dict) -> pd.DataFrame:
    """All sensitivity runs in one long table: which run, which input, its value, then the results."""
    frames = []
    for run, tables in (("base", results["sensitivities"]), ("stress LGD 100%", results["stress"])):
        for name, rows in tables.items():
            if not isinstance(rows, list):
                continue
            frame = pd.DataFrame(rows).rename(columns={name: "value"})
            frame.insert(0, "input", name)
            frame.insert(0, "run", run)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def write_tables(out: Path, results: dict, r: dict, bond_names: list) -> None:
    """CSV versions of the main tables, for readers who do not want to parse the JSON."""
    quarters = [f"Q{k}" for k in range(1, results["n_periods"] + 1)]
    case_index = pd.RangeIndex(1, results["n_cases"] + 1, name="case")
    pd.DataFrame(results["cases"]).to_csv(out / "case_totals.csv", index=False)
    pd.DataFrame(r["pool_cf"], index=case_index, columns=quarters).to_csv(out / "pool_cash_flows.csv", float_format="%.6f")
    default_quarters = np.where(r["defaulted"], r["Q"], 0)
    pd.DataFrame(default_quarters, index=case_index, columns=bond_names).to_csv(out / "default_quarters.csv")
    pd.DataFrame(results["quarterly"]).to_csv(out / "quarterly_cash_flow_summary.csv", index=False)
    pd.DataFrame(results["default_distribution"]).to_csv(out / "default_distribution.csv", index=False)
    sensitivity_frame(results).to_csv(out / "sensitivities.csv", index=False)


def write_figures(figures: Path, results: dict, r: dict) -> None:
    """The figures used in the write-up and the README."""
    apply_chart_style()
    cases = pd.DataFrame(results["cases"])
    totals = results["promised"]["totals"]
    quarterly = pd.DataFrame(results["quarterly"]).set_index("quarter")
    distribution = pd.DataFrame(results["default_distribution"]).set_index("defaults in 5 years")
    sens = {name: pd.DataFrame(rows).set_index(name) for name, rows in results["sensitivities"].items()}
    checks = results["checks"]
    case = results["example"]["case"]
    i = case - 1

    save_figure(plot_total_cash(cases["pool"].values, cases["equity"].values, totals["pool"], totals["equity"]),
                figures / "total_cash_distributions.png")
    save_figure(plot_default_count(distribution, results["deal"]["rho"]), figures / "default_count.png")
    save_figure(plot_quarterly(quarterly, checks["floor_per_quarter"], checks["classes_need_per_quarter"]),
                figures / "quarterly_cash_flows.png")
    save_figure(plot_sensitivities(sens["pd"], sens["lgd"], sens["rho"]), figures / "sensitivities.png")
    save_figure(plot_case(case, r["a_cf"][i], r["b_cf"][i], r["eq_cf"][i], r["pool_promised"]),
                figures / "case_waterfall.png")


def main(argv=None) -> int:
    args = parse_args(argv)
    if not 1 <= args.case <= N_CASES:
        raise SystemExit(f"--case must be between 1 and {N_CASES}, got {args.case}")
    out = Path(args.out)
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    table, source = load_fixed_normals(Path(args.data), N_CASES, BASE_DEAL["n_bonds"], SEED)
    raw = table.values
    Z = moment_match(raw)      # de-meaned and moment matched: the numbers the model uses
    results = build_results(Z, BASE_DEAL, SEED, example_case=args.case, raw=raw)
    results["sampling_check"] = sampling_error_check(BASE_DEAL, N_CASES)
    r = simulate(Z, BASE_DEAL)

    write_json(out / "results.json", results)
    write_tables(out, results, r, list(table.columns))
    pd.DataFrame(Z, index=table.index, columns=table.columns).to_csv(out / "moment_matched_random_numbers.csv", float_format="%.12f")
    write_figures(figures, results, r)

    checks = results["checks"]
    summary = {row["series"]: row for row in results["summary"]}
    print(f"Random numbers: {Z.shape[0]} cases x {Z.shape[1]} bonds ({source})")
    print(f"BIS slide check: {checks['slide_value']:.2f} (slide {checks['slide_target']:.2f})")
    print(f"Average defaults per case: {checks['avg_defaults_simulated']:.3f} (theory {checks['avg_defaults_theory']:.3f})")
    print(f"Pool total:   mean {summary['pool']['mean']:.3f} of {summary['pool']['no-default amount']:.0f}, "
          f"5th pct {summary['pool']['5th pct']:.3f}, min {summary['pool']['min']:.2f}")
    print(f"Equity total: mean {summary['equity']['mean']:.3f} of {summary['equity']['no-default amount']:.0f}, "
          f"5th pct {summary['equity']['5th pct']:.3f}, min {summary['equity']['min']:.2f}")
    print(f"Cases with a shortfall: Class A {checks['class_a_shortfall_cases']}, Class B {checks['class_b_shortfall_cases']}")
    print(f"Outputs written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
