#!/usr/bin/env python3
"""Run the Miniproject 3 (Part 2) valuation: value Class A, Class B and the residual with the
Lintner method on the same 1000 moment-matched cases as Part 1, search over Class B for the
bonus, and write tables, JSON and figures.

    python run_valuation.py --data ../data/fixed_random_numbers.csv --out ../output/part2
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
    MARKET_YTM,
    N_CASES,
    PART2_DEAL,
    PART2_RISK_FREE,
    SEED,
    TRANCHES,
    fair_coupon,
    load_fixed_normals,
    moment_match,
    optimise_class_b,
    simulate,
    valuation_precision,
    value_cdo,
    value_sensitivity,
)
from cdo.report import apply_chart_style, plot_class_b_choice, plot_pv_distributions, save_figure, write_json  # noqa: E402

B_SIZES = np.arange(0.0, 100.5, 1.0)                    # Class B principals searched in the bonus ($MM)
B_COUPONS = np.round(np.arange(0.0, 0.10001, 0.0025), 4)  # Class B coupons searched in the bonus
DEFAULT_LIMITS = (0.0, 0.01, 0.05, 0.10, 0.20)          # caps on the share of cases Class B is short
SENSITIVITIES = {"rho": (0.0, 0.2, 0.4, 0.6, 0.8), "pd": (0.02, 0.04, 0.06, 0.08, 0.12), "lgd": (0.4, 0.6, 0.8, 1.0)}
PRECISION_SETS = 500
MAX_ERROR_SE = 3                                         # max error of a mean, in standard errors


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="../data/fixed_random_numbers.csv", help="fixed random numbers (created if missing)")
    parser.add_argument("--out", default="../output/part2", help="directory for CSV/JSON/PNG outputs")
    return parser.parse_args(argv)


def valuation_block(v: dict, deal: dict) -> dict:
    """The numbers asked for in steps 3 to 5, ready for JSON."""
    n = len(v["pv"])
    sd = v["pv"].std(axis=0, ddof=1)
    principal = deal["a_notional"] + deal["b_notional"]
    return {
        "tranches": list(TRANCHES),
        "market_value": v["market_value"],
        "expected_pool_pv": v["expected_pool_pv"],
        "cost_of_risk": v["cost_of_risk"],
        "pool_variance": v["pool_variance"],
        "cost_per_unit_risk": v["cost_per_unit_risk"],
        "expected_pv": v["expected_pv"].tolist(),
        "sd_pv": sd.tolist(),
        "max_error": (MAX_ERROR_SE * sd / np.sqrt(n)).tolist(),
        "covariance": v["covariance"].tolist(),
        "cov_with_pool": v["cov_with_pool"].tolist(),
        "risk_charge": v["risk_charge"].tolist(),
        "value": v["value"].tolist(),
        "p_default": v["defaults"].mean(axis=0).tolist(),
        "default_cases": v["defaults"].sum(axis=0).tolist(),
        "bank_investment": v["market_value"] - principal,
        "bank_gain": principal + v["value"][2] - v["market_value"],
    }


def build_valuation_results(Z: np.ndarray, deal: dict, rate: float, ytm: float, seed: int) -> tuple:
    """Everything the write-up, notebook checks and project page need, plus the per-case table and the grid."""
    r = simulate(Z, deal)
    v = value_cdo(r["pool_cf"], deal, rate, ytm)
    lam = v["cost_per_unit_risk"]
    fair = {"a": fair_coupon(r["pool_cf"], deal, rate, lam, "a"), "b": fair_coupon(r["pool_cf"], deal, rate, lam, "b")}
    at_fair = value_cdo(r["pool_cf"], {**deal, "a_coupon": fair["a"], "b_coupon": fair["b"]}, rate, ytm)
    bonus = optimise_class_b(r["pool_cf"], deal, rate, ytm, B_SIZES, B_COUPONS, DEFAULT_LIMITS)
    grid = pd.DataFrame(bonus.pop("grid"))
    results = {
        "deal": deal,
        "risk_free": rate,
        "market_ytm": ytm,
        "n_cases": len(Z),
        "seed": seed,
        "promised": {"class_a": r["a_due"].tolist(), "class_b": r["b_due"].tolist(), "pool": r["pool_promised"].tolist()},
        "valuation": valuation_block(v, deal),
        "fair_coupons": fair,
        "at_fair_coupons": valuation_block(at_fair, deal),
        "bonus": bonus,
        "sensitivities": {name: value_sensitivity(Z, deal, rate, ytm, name, values) for name, values in SENSITIVITIES.items()},
        "precision": valuation_precision(deal, rate, ytm, len(Z), n_sets=PRECISION_SETS),
    }
    cases = pd.DataFrame({
        "case": np.arange(1, len(Z) + 1),
        "pv_class_a": v["pv"][:, 0], "pv_class_b": v["pv"][:, 1], "pv_residual": v["pv"][:, 2],
        "pv_pool": v["pv"].sum(axis=1),
        "default_class_a": v["defaults"][:, 0], "default_class_b": v["defaults"][:, 1],
    })
    return results, cases, grid


def write_tables(results: dict, cases: pd.DataFrame, grid: pd.DataFrame, out: Path) -> None:
    cases.to_csv(out / "case_values.csv", index=False, float_format="%.10f")
    grid.to_csv(out / "structure_grid.csv", index=False, float_format="%.10f")
    bonus = results["bonus"]
    pd.DataFrame(bonus["at_target_rate"]).to_csv(out / "class_b_at_target_rate.csv", index=False, float_format="%.10f")
    pd.DataFrame(bonus["fair_coupon_curve"]).to_csv(out / "fair_coupon_curve.csv", index=False, float_format="%.10f")
    frames = [pd.DataFrame(rows).rename(columns={name: "input value"}).assign(input=name)
              for name, rows in results["sensitivities"].items()]
    pd.concat(frames, ignore_index=True).to_csv(out / "value_sensitivities.csv", index=False, float_format="%.10f")


def write_figures(results: dict, cases: pd.DataFrame, figures: Path) -> None:
    apply_chart_style()
    v, bonus = results["valuation"], results["bonus"]
    save_figure(plot_pv_distributions(cases["pv_pool"].to_numpy(), cases["pv_residual"].to_numpy(), v["market_value"],
                                      v["value"][2], results["risk_free"]), figures / "pv_distributions.png")
    save_figure(plot_class_b_choice(bonus["at_target_rate"], bonus["fair_coupon_curve"], bonus["target_b_coupon"],
                                    bonus["break_even_size"], results["deal"]["b_notional"]), figures / "class_b_choice.png")


def main(argv=None) -> int:
    args = parse_args(argv)
    out = Path(args.out)
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    raw, source = load_fixed_normals(Path(args.data), N_CASES, PART2_DEAL["n_bonds"], SEED)
    print(f"Random numbers: {source}")
    results, cases, grid = build_valuation_results(moment_match(raw), PART2_DEAL, PART2_RISK_FREE, MARKET_YTM, SEED)
    write_json(out / "results.json", results)
    write_tables(results, cases, grid, out)
    write_figures(results, cases, figures)

    v, bonus = results["valuation"], results["bonus"]
    print(f"Bonds at {MARKET_YTM:.0%}: {v['market_value']:.4f}  expected PV at {PART2_RISK_FREE:.0%}: {v['expected_pool_pv']:.4f}  "
          f"cost of risk: {v['cost_of_risk']:.4f}  per unit variance: {v['cost_per_unit_risk']:.6f}")
    for name, expected, value in zip(TRANCHES, v["expected_pv"], v["value"]):
        print(f"  {name:9} expected PV {expected:9.4f}   Lintner value {value:9.4f}")
    print(f"Bonus: break-even Class B at {bonus['target_b_coupon']:.1%} = {bonus['break_even_size']:.2f}, "
          f"largest Class B worth par = {bonus['largest_class_b_at_par']:.2f}")
    print(f"Outputs written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
