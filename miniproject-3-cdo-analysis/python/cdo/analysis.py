"""Statistics of the simulated cash flows and one-at-a-time sensitivities."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import binom, norm

from .bonds import bis_expected_value, promised_cash_flows
from .defaults import correlate
from .model import MARKET_YTM, RISK_FREE, simulate
from .random_numbers import moment_match

# Values each input takes in the sensitivity runs (the base value is always included).
PD_GRID = (0.01, 0.02, 0.04, 0.06, 0.08, 0.12)
LGD_GRID = (0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00)
RHO_GRID = (0.0, 0.10, 0.20, 0.40, 0.60, 0.80)
B_NOTIONAL_GRID = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0)
STRESS_LGD = 1.00
STRESS_PD_GRID = (0.04, 0.08, 0.12, 0.20)
STRESS_RHO_GRID = (0.0, 0.20, 0.40, 0.60, 0.80)

SAMPLING_CHECK_SEED = 6104      # for the fresh sets of random numbers in sampling_error_check

# The example on the BIS slide in Lecture 5, used as a check on the bond function.
SLIDE_EXAMPLE = {"face": 1000.0, "coupon": 0.055, "years": 5, "pd": 0.03, "lgd": 0.60, "rate": 0.04, "value": 984.73}


def describe(x: np.ndarray, no_default: float) -> dict:
    """Summary statistics of one total per case, next to what it would be with no defaults."""
    return {
        "no-default amount": no_default,
        "mean": x.mean(),
        "std dev": x.std(ddof=1),
        "std error": x.std(ddof=1) / np.sqrt(len(x)),      # of the mean if the cases were independent (an upper bound here)
        "min": x.min(),
        "5th pct": np.percentile(x, 5),
        "median": np.median(x),
        "95th pct": np.percentile(x, 95),
        "max": x.max(),
        "mean / no-default amount": x.mean() / no_default if no_default else np.nan,
    }


def summary_row(Z: np.ndarray, p: dict, tol: float = 1e-9) -> dict:
    """Key results for one parameter set (one row of a sensitivity table).

    A shortfall means the class received less than it was due in at least one period.
    """
    r = simulate(Z, p)
    n_defaults = r["defaulted"].sum(axis=1)
    pool = r["pool_cf"].sum(axis=1)
    equity = r["eq_cf"].sum(axis=1)
    a_short = (r["a_due"] - r["a_cf"]).sum(axis=1)
    b_short = (r["b_due"] - r["b_cf"]).sum(axis=1)
    b_due_total = r["b_due"].sum()
    return {
        "avg defaults": n_defaults.mean(),
        "P(no default)": (n_defaults == 0).mean(),
        "P(4+ defaults)": (n_defaults >= 4).mean(),
        "pool mean": pool.mean(),
        "pool 5th pct": np.percentile(pool, 5),
        "equity mean": equity.mean(),
        "equity std": equity.std(ddof=1),
        "equity 5th pct": np.percentile(equity, 5),
        "equity min": equity.min(),
        "P(A shortfall)": (a_short > tol).mean(),
        "P(B shortfall)": (b_short > tol).mean(),
        "B paid / due": r["b_cf"].sum(axis=1).mean() / b_due_total if b_due_total else np.nan,
    }


def sensitivity(Z: np.ndarray, start: dict, name: str, values: tuple) -> pd.DataFrame:
    """Rerun the model on the same random numbers with one input changed at a time."""
    rows = {v: summary_row(Z, {**start, name: v}) for v in values}
    return pd.DataFrame(rows).T.rename_axis(name)


def expected_pool_cash_flows(p: dict) -> np.ndarray:
    """Exact expected pool cash flow per period: promised x [(1 - LGD) + LGD x (1 - pi)^(k / freq)]."""
    n_periods = p["years"] * p["freq"]
    promised = p["n_bonds"] * promised_cash_flows(p["face"], p["coupon"], n_periods, p["freq"])
    survival = (1.0 - p["pd"]) ** (np.arange(1, n_periods + 1) / p["freq"])
    return promised * ((1.0 - p["lgd"]) + p["lgd"] * survival)


def exact_default_count_distribution(n_bonds: int, p_default: float, rho: float, steps: int = 2000) -> np.ndarray:
    """Exact distribution of the number of defaults under the model, with no random numbers.

    Equal pairwise correlation rho is the same as one common factor M shared by all bonds. Given M
    the bonds default independently with probability N((N^-1(p) - sqrt(rho) M) / sqrt(1 - rho)),
    so the answer is a binomial averaged over M. The average uses Simpson's rule on -9 to 9, which
    stays accurate at high correlation where the conditional probability is close to a step.
    """
    k = np.arange(n_bonds + 1)
    if rho == 0 or p_default in (0.0, 1.0):
        return binom.pmf(k, n_bonds, p_default)
    m = np.linspace(-9.0, 9.0, steps + 1)
    simpson = np.ones(steps + 1)
    simpson[1:-1:2] = 4.0
    simpson[2:-1:2] = 2.0
    weights = simpson * (m[1] - m[0]) / 3.0 * norm.pdf(m)
    conditional = norm.cdf((norm.ppf(p_default) - np.sqrt(rho) * m) / np.sqrt(1.0 - rho))
    return (binom.pmf(k[:, None], n_bonds, conditional[None, :]) * weights).sum(axis=1)


def exact_shortfall_probability(p: dict, due: np.ndarray) -> float | None:
    """Exact probability that the pool cannot pay `due` in some period (no random numbers).

    The pool falls short once enough bonds have defaulted. When that number is smallest at maturity,
    which holds whenever principal dominates, the probability is the exact chance of at least that
    many defaults over the life of the deal. Returns None if a coupon period would bind first.
    """
    n_periods = p["years"] * p["freq"]
    bond = promised_cash_flows(p["face"], p["coupon"], n_periods, p["freq"])
    slack = p["n_bonds"] * bond - due
    if (slack < -1e-12).any():
        return 1.0  # the class is owed more than the pool even promises
    if p["lgd"] == 0:
        return 0.0
    needed = np.floor(slack / (p["lgd"] * bond) + 1e-12).astype(int) + 1
    if needed.min() > p["n_bonds"]:
        return 0.0
    if needed[-1] > needed.min():
        return None
    p_default = 1.0 - (1.0 - p["pd"]) ** p["years"]
    exact = exact_default_count_distribution(p["n_bonds"], p_default, p["rho"])
    return float(exact[max(needed[-1], 0):].sum())


def default_count_table(n_defaults: np.ndarray, pool_total: np.ndarray, eq_total: np.ndarray, n_bonds: int,
                        p_default: float, rho: float) -> pd.DataFrame:
    """Distribution of the number of defaults: simulated, exact under the model, and if independent."""
    k = np.arange(n_bonds + 1)
    counts = np.bincount(n_defaults, minlength=n_bonds + 1)
    return pd.DataFrame({
        "cases": counts,
        "simulated probability": counts / len(n_defaults),
        "exact with correlation": exact_default_count_distribution(n_bonds, p_default, rho),
        "if independent (binomial)": binom.pmf(k, n_bonds, p_default),
        "avg pool cash ($MM)": [pool_total[n_defaults == j].mean() if counts[j] else np.nan for j in k],
        "avg equity cash ($MM)": [eq_total[n_defaults == j].mean() if counts[j] else np.nan for j in k],
    }, index=pd.Index(k, name="defaults in 5 years"))


def quarterly_table(r: dict, pool_expected: np.ndarray) -> pd.DataFrame:
    """Per-period cash flow statistics across the cases for the pool and each class."""
    quarters = np.arange(1, r["pool_cf"].shape[1] + 1)
    return pd.DataFrame({
        "pool promised": r["pool_promised"],
        "pool expected (exact)": pool_expected,
        "pool mean": r["pool_cf"].mean(axis=0),
        "pool 5th pct": np.percentile(r["pool_cf"], 5, axis=0),
        "pool 95th pct": np.percentile(r["pool_cf"], 95, axis=0),
        "Class A mean": r["a_cf"].mean(axis=0),
        "Class B mean": r["b_cf"].mean(axis=0),
        "equity mean": r["eq_cf"].mean(axis=0),
        "equity 5th pct": np.percentile(r["eq_cf"], 5, axis=0),
        "equity 95th pct": np.percentile(r["eq_cf"], 95, axis=0),
    }, index=pd.Index(quarters, name="quarter"))


def average_pairwise_correlation(A: np.ndarray) -> float:
    """Mean of the off-diagonal sample correlations between the columns of A."""
    corr = np.corrcoef(np.asarray(A, dtype=float).T)
    return float(corr[~np.eye(corr.shape[0], dtype=bool)].mean())


def safe_lgd_limits(r: dict) -> dict:
    """Largest LGD at which each class is still paid in full even if every bond defaults at once."""
    with np.errstate(divide="ignore", invalid="ignore"):
        a_limit = np.min(1.0 - r["a_due"] / r["pool_promised"])
        b_limit = np.min(1.0 - (r["a_due"] + r["b_due"]) / r["pool_promised"])
    return {"class_a": float(a_limit), "class_b": float(b_limit)}


def example_case_record(r: dict, case: int) -> dict:
    """Default quarters and cash flows of one case (cases are numbered from 1)."""
    i = case - 1
    return {
        "case": case,
        "n_defaults": int(r["defaulted"][i].sum()),
        "default_quarters": [int(q) if hit else None for q, hit in zip(r["Q"][i], r["defaulted"][i])],
        "pool": r["pool_cf"][i].tolist(),
        "class_a": r["a_cf"][i].tolist(),
        "class_b": r["b_cf"][i].tolist(),
        "equity": r["eq_cf"][i].tolist(),
    }


def _records(table: pd.DataFrame) -> list:
    """DataFrame (with its index) -> list of plain dicts that json can write; NaN becomes null."""
    return json.loads(table.reset_index().to_json(orient="records"))


def sampling_error_check(base: dict, n_cases: int, n_tables: int = 2000, seed: int = SAMPLING_CHECK_SEED) -> dict:
    """How far the mean total pool cash moves from one set of random numbers to the next.

    Draws n_tables fresh sets, runs the model on each as drawn and after moment matching, and reports
    the standard deviation of the mean across sets. That is the real sampling error of the mean. The
    usual std dev / sqrt(n) formula only gives it when the cases are independent.
    """
    rng = np.random.default_rng(seed)
    as_drawn, matched, formula = [], [], []
    for _ in range(n_tables):
        Z = rng.standard_normal((n_cases, base["n_bonds"]))
        plain_total = simulate(Z, base)["pool_cf"].sum(axis=1)
        matched_total = simulate(moment_match(Z), base)["pool_cf"].sum(axis=1)
        as_drawn.append(plain_total.mean())
        matched.append(matched_total.mean())
        formula.append(matched_total.std(ddof=1) / np.sqrt(n_cases))
    return {
        "n_tables": n_tables,
        "seed": seed,
        "exact_mean": float(expected_pool_cash_flows(base).sum()),
        "mean_as_drawn": float(np.mean(as_drawn)),
        "mean_matched": float(np.mean(matched)),
        "sd_of_mean_as_drawn": float(np.std(as_drawn, ddof=1)),
        "sd_of_mean_matched": float(np.std(matched, ddof=1)),
        "average_formula_std_error": float(np.mean(formula)),
    }


def moment_matching_summary(raw: np.ndarray, matched: np.ndarray) -> dict:
    """How far the stored draws are from the moments of independent standard normals, before and after matching."""
    def moments(A: np.ndarray) -> dict:
        cov = np.cov(A, rowvar=False, bias=True)
        corr = np.corrcoef(A, rowvar=False)
        off = ~np.eye(A.shape[1], dtype=bool)
        return {
            "max_abs_mean": float(np.abs(A.mean(axis=0)).max()),
            "min_variance": float(cov.diagonal().min()),
            "max_variance": float(cov.diagonal().max()),
            "max_abs_correlation": float(np.abs(corr[off]).max()),
        }
    return {"raw": moments(raw), "matched": moments(matched)}


def build_results(Z: np.ndarray, base: dict, seed: int, example_case: int = 5, raw: np.ndarray | None = None) -> dict:
    """Everything the report, the Excel check and the site need, as plain JSON-ready values."""
    r = simulate(Z, base)
    n_cases, n_bonds = Z.shape
    tol = 1e-9

    n_defaults = r["defaulted"].sum(axis=1)
    totals = {name: r[key].sum(axis=1) for name, key in
              (("pool", "pool_cf"), ("class_a", "a_cf"), ("class_b", "b_cf"), ("equity", "eq_cf"))}
    a_short = (r["a_due"] - r["a_cf"]).sum(axis=1)
    b_short = (r["b_due"] - r["b_cf"]).sum(axis=1)
    promised = {
        "pool": float(r["pool_promised"].sum()),
        "class_a": float(r["a_due"].sum()),
        "class_b": float(r["b_due"].sum()),
    }
    promised["equity"] = promised["pool"] - promised["class_a"] - promised["class_b"]

    summary = pd.DataFrame({name: describe(totals[name], promised[name]) for name in totals}).T.rename_axis("series")
    summary["P(below no-default amount)"] = [
        (totals["pool"] < promised["pool"] - tol).mean(),
        (a_short > tol).mean(),
        (b_short > tol).mean(),
        (totals["equity"] < promised["equity"] - tol).mean(),
    ]

    p_default = 1.0 - (1.0 - base["pd"]) ** base["years"]
    pool_expected = expected_pool_cash_flows(base)
    quarterly = quarterly_table(r, pool_expected)
    gap_pct = (quarterly["pool mean"] / quarterly["pool expected (exact)"] - 1.0) * 100.0

    slide = SLIDE_EXAMPLE
    slide_promised = promised_cash_flows(slide["face"], slide["coupon"], slide["years"], 1)
    slide_rows, slide_value = bis_expected_value(slide_promised, slide["pd"], slide["lgd"], slide["rate"])

    stress = {**base, "lgd": STRESS_LGD}
    floor = r["pool_promised"] * (1.0 - base["lgd"])
    worst = int(np.argmin(totals["equity"]))
    cases = pd.DataFrame({"n_defaults": n_defaults, **totals}, index=pd.RangeIndex(1, n_cases + 1, name="case"))

    return {
        "deal": dict(base),
        "market_ytm": MARKET_YTM,
        "risk_free": RISK_FREE,
        "seed": seed,
        "moment_matching": moment_matching_summary(raw, Z) if raw is not None else None,
        "n_cases": n_cases,
        "n_bonds": n_bonds,
        "n_periods": int(r["pool_cf"].shape[1]),
        "example": example_case_record(r, example_case),
        "promised": {
            "bond": r["bond_promised"].tolist(),
            "pool": r["pool_promised"].tolist(),
            "class_a": r["a_due"].tolist(),
            "class_b": r["b_due"].tolist(),
            "totals": promised,
        },
        "checks": {
            "slide_value": slide_value,
            "slide_target": slide["value"],
            "slide_rows": [{"default_year": k, "probability": prob, "pv": pv} for k, prob, pv in slide_rows],
            "corr_normals": average_pairwise_correlation(correlate(Z, base["rho"])),
            "corr_default_times": average_pairwise_correlation(r["T"]),
            "corr_default_flags": average_pairwise_correlation(r["defaulted"]),
            "p_default_5y_theory": p_default,
            "p_default_5y_simulated": float(r["defaulted"].mean()),
            "avg_defaults_theory": n_bonds * p_default,
            "avg_defaults_simulated": float(n_defaults.mean()),
            "max_quarterly_gap_pct": float(gap_pct.abs().max()),
            "pool_total_expected": float(pool_expected.sum()),
            "floor_per_quarter": float(floor[0]),
            "floor_at_maturity": float(floor[-1]),
            "classes_need_per_quarter": float(r["a_due"][0] + r["b_due"][0]),
            "classes_need_at_maturity": float(r["a_due"][-1] + r["b_due"][-1]),
            "safe_lgd": safe_lgd_limits(r),
            "class_a_shortfall_cases": int((a_short > tol).sum()),
            "class_b_shortfall_cases": int((b_short > tol).sum()),
            "p_four_or_more_simulated": float((n_defaults >= 4).mean()),
            "p_four_or_more_exact": float(exact_default_count_distribution(n_bonds, p_default, base["rho"])[4:].sum()),
            "p_four_or_more_independent": float(binom.sf(3, n_bonds, p_default)),
            "stress_exact_p_a_shortfall": exact_shortfall_probability(stress, r["a_due"]),
            "stress_exact_p_b_shortfall": exact_shortfall_probability(stress, r["a_due"] + r["b_due"]),
            "worst_case": worst + 1,
            "worst_case_defaults": int(n_defaults[worst]),
        },
        "summary": _records(summary),
        "default_distribution": _records(
            default_count_table(n_defaults, totals["pool"], totals["equity"], n_bonds, p_default, base["rho"])),
        "quarterly": _records(quarterly),
        "sensitivities": {
            "pd": _records(sensitivity(Z, base, "pd", PD_GRID)),
            "lgd": _records(sensitivity(Z, base, "lgd", LGD_GRID)),
            "rho": _records(sensitivity(Z, base, "rho", RHO_GRID)),
            "b_notional": _records(sensitivity(Z, base, "b_notional", B_NOTIONAL_GRID)),
        },
        "stress": {
            "lgd": STRESS_LGD,
            "pd": _records(sensitivity(Z, stress, "pd", STRESS_PD_GRID)),
            "rho": _records(sensitivity(Z, stress, "rho", STRESS_RHO_GRID)),
        },
        "cases": _records(cases),
    }
