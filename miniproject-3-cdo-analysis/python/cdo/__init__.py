"""Mini-Project 3 - Simplified CDO analysis.

Simulate correlated defaults of 10 collateral bonds from fixed random numbers,
turn them into quarterly cash flows with the BIS debt model and distribute the
pool through the Class A / Class B / equity waterfall
(FRE 6103 Valuation for Financial Engineering, NYU).
"""
from .analysis import (
    build_results,
    default_count_table,
    describe,
    exact_default_count_distribution,
    expected_pool_cash_flows,
    quarterly_table,
    sampling_error_check,
    sensitivity,
    summary_row,
)
from .bonds import bis_cash_flows, bis_expected_value, promised_cash_flows
from .defaults import correlate, correlation_matrix, default_period, default_times
from .model import BASE_DEAL, MARKET_YTM, RISK_FREE, simulate, validate_deal
from .random_numbers import N_CASES, SEED, load_fixed_normals, moment_match
from .valuation import (
    PART2_DEAL,
    PART2_RISK_FREE,
    TRANCHES,
    bond_market_value,
    break_even_size,
    cost_of_risk,
    default_indicators,
    discount_factors,
    fair_coupon,
    lintner_values,
    optimise_class_b,
    structure_values,
    tranche_cash_flows,
    tranche_pvs,
    valuation_precision,
    value_cdo,
    value_sensitivity,
)
from .waterfall import waterfall

# Shown on the notebook, the write-up and the project page.
PROJECT = {
    "title": "Simplified CDO Analysis",
    "assignment": "Miniproject 3 (Part 1)",
    "course": "FRE 6103 Valuation for Financial Engineering (NYU Tandon)",
    "authors": "Raj Pawar (rsp9234) and Michael Brick (mb11311)",
    "members_line": "Raj Pawar rsp9234, Michael Brick mb11311",
    "repo_url": "https://github.com/pawarraj8888/valuation-assignments/tree/main/miniproject-3-cdo-analysis",
    "site_url": "https://pawarraj8888.github.io/valuation-assignments/cdo-analysis/",
}

PROJECT_PART2 = {
    **PROJECT,
    "title": "CDO Valuation with the Lintner Model",
    "assignment": "Miniproject 3 (Part 2)",
    "site_url": "https://pawarraj8888.github.io/valuation-assignments/cdo-analysis/valuation.html",
}

__all__ = [
    "PROJECT",
    "PROJECT_PART2",
    "BASE_DEAL",
    "MARKET_YTM",
    "RISK_FREE",
    "N_CASES",
    "SEED",
    "promised_cash_flows",
    "bis_cash_flows",
    "bis_expected_value",
    "correlation_matrix",
    "correlate",
    "default_times",
    "default_period",
    "waterfall",
    "validate_deal",
    "simulate",
    "describe",
    "summary_row",
    "sensitivity",
    "expected_pool_cash_flows",
    "exact_default_count_distribution",
    "default_count_table",
    "quarterly_table",
    "sampling_error_check",
    "build_results",
    "load_fixed_normals",
    "moment_match",
    "PART2_DEAL",
    "PART2_RISK_FREE",
    "TRANCHES",
    "discount_factors",
    "bond_market_value",
    "tranche_cash_flows",
    "tranche_pvs",
    "default_indicators",
    "cost_of_risk",
    "lintner_values",
    "value_cdo",
    "structure_values",
    "fair_coupon",
    "break_even_size",
    "optimise_class_b",
    "value_sensitivity",
    "valuation_precision",
]
