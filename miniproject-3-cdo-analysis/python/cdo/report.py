"""Figures and JSON output shared by the pipeline, the notebook and the write-up."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

# One colour per entity, the same in every figure and on the project page.
COLORS = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "pool": "#52514e",
    "benchmark": "#898781",
    "equity": "#2a78d6",
    "class_a": "#eb6834",
    "class_b": "#1baf7a",
}


def apply_chart_style() -> None:
    """Quiet chart defaults: thin marks, a hairline grid and no box around the plot."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9.5,
        "figure.facecolor": "white",
        "axes.facecolor": COLORS["surface"],
        "axes.edgecolor": COLORS["axis"],
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "axes.labelcolor": COLORS["ink_secondary"],
        "axes.titlecolor": COLORS["ink"],
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "xtick.color": COLORS["ink_secondary"],
        "ytick.color": COLORS["ink_secondary"],
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    })


def reference_line(ax, x: float, text: str, height: float) -> None:
    """Vertical hairline with its label written beside it, so it needs no legend entry."""
    ax.axvline(x, color=COLORS["ink"], linewidth=1.0)
    ax.annotate(text, xy=(x, height), xycoords=("data", "axes fraction"), xytext=(-5, 0),
                textcoords="offset points", ha="right", va="top", fontsize=8.5, color=COLORS["ink_secondary"])


def plot_total_cash(pool_total: np.ndarray, eq_total: np.ndarray, pool_promised: float,
                    eq_promised: float) -> Figure:
    """Distribution of total 5-year cash: the collateral pool (left) and the bank's equity (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    panels = [
        (axes[0], pool_total, pool_promised, COLORS["pool"], "Collateral pool", "promised"),
        (axes[1], eq_total, eq_promised, COLORS["equity"], "Equity (bank residual)", "no defaults"),
    ]
    for ax, x, top, color, title, top_label in panels:
        ax.hist(x, bins=40, color=color, edgecolor=COLORS["surface"], linewidth=0.8)
        reference_line(ax, top, f"{top_label} {top:.0f}", 0.99)
        reference_line(ax, x.mean(), f"mean {x.mean():.1f}", 0.87)
        reference_line(ax, np.percentile(x, 5), f"5th percentile {np.percentile(x, 5):.1f}", 0.75)
        ax.set_title(title)
        ax.set_xlabel("Total 5-year cash ($MM)")
    axes[0].set_ylabel("Number of cases")
    fig.tight_layout()
    return fig


def plot_default_count(table: pd.DataFrame, rho: float) -> Figure:
    """Number of defaults per case: simulated, exact under the model, and if defaults were independent."""
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    k = table.index.values
    ax.bar(k - 0.19, table["simulated probability"], width=0.34, color=COLORS["pool"],
           label=f"Simulated, correlation {rho:.2f}")
    ax.bar(k + 0.19, table["if independent (binomial)"], width=0.34, color=COLORS["benchmark"],
           label="Independent defaults (binomial)")
    ax.plot(k - 0.19, table["exact with correlation"], linestyle="none", marker="o", markersize=6.5,
            color=COLORS["equity"], markeredgecolor=COLORS["surface"], markeredgewidth=1.2,
            label=f"Exact under the model, correlation {rho:.2f}")
    ax.set_xticks(k)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Number of bonds defaulting within 5 years")
    ax.set_ylabel("Share of cases")
    ax.set_title("Distribution of the number of defaults")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_quarterly(quarterly: pd.DataFrame, floor: float, classes_need: float) -> Figure:
    """Coupon quarters: what the pool pays (mean and 5th to 95th band) against what the classes need."""
    part = quarterly.iloc[:-1]
    q = part.index.values
    promised = part["pool promised"].iloc[0]
    marker = {"marker": "o", "markersize": 5, "markeredgecolor": COLORS["surface"], "markeredgewidth": 1.0}
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.fill_between(q, part["pool 5th pct"], part["pool 95th pct"], color=COLORS["pool"], alpha=0.12, linewidth=0,
                    label="Pool, 5th to 95th percentile")
    ax.plot(q, part["pool mean"], color=COLORS["pool"], label="Pool, mean", **marker)
    ax.plot(q, part["equity mean"], color=COLORS["equity"], label="Equity, mean", **marker)
    levels = [
        (promised, f"promised by the collateral: {promised:.2f}"),
        (floor, f"pool if all bonds default in quarter 1: {floor:.2f}"),
        (classes_need, f"due to Class A + Class B: {classes_need:.2f}"),
    ]
    for level, text in levels:
        ax.axhline(level, color=COLORS["ink"], linewidth=1.0)
        ax.annotate(text, xy=(q[-1], level), xytext=(0, 4), textcoords="offset points", ha="right", va="bottom",
                    fontsize=8.5, color=COLORS["ink_secondary"])
    ax.set_xticks(q)
    ax.set_ylim(0, promised * 1.12)
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Cash flow ($MM)")
    ax.set_title(f"Quarterly coupon cash flows, quarters 1 to {q[-1]}")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.52))
    fig.tight_layout()
    return fig


def plot_sensitivities(sens_pd: pd.DataFrame, sens_lgd: pd.DataFrame, sens_rho: pd.DataFrame) -> Figure:
    """Mean, 5th percentile and worst case of total equity cash as each input is varied on its own."""
    panels = [
        (sens_pd, "Annual default probability (%)", 100),
        (sens_lgd, "Loss given default (%)", 100),
        (sens_rho, "Default correlation", 1),
    ]
    statistics = [("equity mean", "-", "Mean"), ("equity 5th pct", "--", "5th percentile"), ("equity min", ":", "Worst case")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0), sharey=True)
    for ax, (table, label, scale) in zip(axes, panels):
        x = table.index.values * scale
        for column, line_style, name in statistics:
            ax.plot(x, table[column], linestyle=line_style, color=COLORS["equity"], marker="o", markersize=5,
                    markeredgecolor=COLORS["surface"], markeredgewidth=1.0, label=name)
        ax.set_xlabel(label)
    axes[0].set_ylabel("Total 5-year cash to equity ($MM)")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(title="Equity", loc="lower left")
    fig.suptitle("Sensitivity of the equity cash flows (other inputs at base values)", fontsize=10.5,
                 fontweight="bold", color=COLORS["ink"], x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_case(case: int, a_cf: np.ndarray, b_cf: np.ndarray, eq_cf: np.ndarray, pool_promised: np.ndarray) -> Figure:
    """Waterfall of one case: who receives the pool's cash in the coupon quarters and at maturity."""
    n = len(a_cf)
    series = [("Class A", a_cf, COLORS["class_a"]), ("Class B", b_cf, COLORS["class_b"]),
              ("Equity", eq_cf, COLORS["equity"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [6, 1]})
    for ax, start, stop in ((axes[0], 0, n - 1), (axes[1], n - 1, n)):
        x = np.arange(start + 1, stop + 1)
        bottom = np.zeros(stop - start)
        for name, cf, color in series:
            ax.bar(x, cf[start:stop], bottom=bottom, width=0.62, color=color, edgecolor=COLORS["surface"],
                   linewidth=1.2, label=name)
            bottom = bottom + cf[start:stop]
        ax.hlines(pool_promised[start:stop], x - 0.42, x + 0.42, color=COLORS["ink"], linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xlim(x[0] - 0.7, x[-1] + 0.7)
    axes[0].annotate("black line = promised by the collateral", xy=(n - 1, pool_promised[0]), xytext=(0, 5),
                     textcoords="offset points", ha="right", va="bottom", fontsize=8.5,
                     color=COLORS["ink_secondary"])
    axes[0].set_ylim(0, pool_promised[0] * 1.25)
    axes[0].set_xlabel("Quarter")
    axes[0].set_ylabel("Cash flow ($MM)")
    axes[0].set_title(f"Case {case}: coupon quarters")
    axes[0].legend(loc="upper left", ncol=3)
    axes[1].set_xlabel("Quarter")
    axes[1].set_title("Maturity")
    fig.tight_layout()
    return fig


def plot_pv_distributions(pool_pv: np.ndarray, residual_pv: np.ndarray, market_value: float,
                          residual_value: float, rate: float) -> Figure:
    """PV at the risk-free rate across the cases: the collateral pool (left) and the residual (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    panels = [
        (axes[0], pool_pv, COLORS["pool"], "Collateral pool", (market_value, "market value at 9%")),
        (axes[1], residual_pv, COLORS["equity"], "Residual (bank)", (residual_value, "Lintner value")),
    ]
    label_box = {"boxstyle": "square,pad=0.15", "fc": COLORS["surface"], "ec": "none"}
    for ax, x, color, title, (mark, mark_label) in panels:
        ax.hist(x, bins=40, color=color, edgecolor=COLORS["surface"], linewidth=0.8)
        for value, text, height in ((x.mean(), f"expected PV {x.mean():.1f}", 0.99), (mark, f"{mark_label} {mark:.1f}", 0.87)):
            ax.axvline(value, color=COLORS["ink"], linewidth=1.0)
            ax.annotate(text, xy=(value, height), xycoords=("data", "axes fraction"), xytext=(-5, 0), textcoords="offset points",
                        ha="right", va="top", fontsize=8.5, color=COLORS["ink_secondary"], bbox=label_box, zorder=5)
        ax.set_title(title)
        ax.set_xlabel(f"PV of the 5-year cash flows at {rate:.0%} ($MM)")
    axes[0].set_ylabel("Number of cases")
    fig.tight_layout()
    return fig


def plot_class_b_choice(at_target: list, fair_curve: list, target_coupon: float, break_even: float,
                        proposed: float) -> Figure:
    """Bonus: the bank's gain against Class B's size (left); Class B's fair coupon and default probability (right)."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.0))
    shown = [row for row in at_target if row["b_notional"] <= 80]
    left.plot([row["b_notional"] for row in shown], [row["bank_gain"] for row in shown], color=COLORS["class_b"],
              label=f"Class B coupon fixed at {target_coupon:.1%}")
    left.plot([row["b_notional"] for row in fair_curve], [row["bank_gain"] for row in fair_curve],
              color=COLORS["benchmark"], linestyle="--", label="Class B coupon set to its fair rate")
    left.axhline(0, color=COLORS["axis"], linewidth=0.8)
    reference_line(left, proposed, f"proposed {proposed:.0f}", 0.99)
    reference_line(left, break_even, f"{target_coupon:.1%} is fair at {break_even:.1f}", 0.87)
    left.set_title("Bank's gain: cash from the classes + residual value - bonds")
    left.set_xlabel("Class B principal ($MM)")
    left.set_ylabel("$MM")
    left.legend(loc="upper left", bbox_to_anchor=(0.0, 0.8))

    fair_size = [row["b_notional"] for row in fair_curve]
    fair_rate = [row["fair_coupon"] for row in fair_curve]
    right.plot(fair_size, fair_rate, color=COLORS["class_b"], label="Fair coupon (left axis)")
    right.axhline(target_coupon, color=COLORS["axis"], linewidth=0.8)
    right.annotate(f"target rate {target_coupon:.1%}", xy=(0.01, target_coupon), xycoords=("axes fraction", "data"),
                   xytext=(0, 3), textcoords="offset points", fontsize=8.5, color=COLORS["ink_secondary"])
    right.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    right.set_ylim(0.035, max(fair_rate) * 1.08)
    right.set_ylabel("Fair coupon")
    short = right.twinx()
    short.plot(fair_size, [row["p_default_b"] for row in fair_curve], color=COLORS["ink_secondary"], linestyle=":",
               label="Share of cases Class B is short (right axis)")
    short.yaxis.set_major_formatter(PercentFormatter(1.0))
    short.set_ylim(0, 1)
    short.grid(False)
    short.spines["right"].set_visible(True)
    short.set_ylabel("Share of cases short")
    lines = right.get_legend_handles_labels(), short.get_legend_handles_labels()
    right.legend(lines[0][0] + lines[1][0], lines[0][1] + lines[1][1], loc="upper left", bbox_to_anchor=(0.0, 0.92))
    right.set_title("Class B at its fair coupon")
    right.set_xlabel("Class B principal ($MM)")
    fig.tight_layout()
    return fig


def plot_leverage(profile: list, limits: list, no_class_b: dict, risk_free: float, labelled: tuple = (0.25, 0.5, 1.0),
                  max_risk: float = 1.2) -> Figure:
    """Bonus, the bank's side, with Class B at its fair coupon: the relative equity risk against Class B's
    size with the Goal Seek answers on top (left), and the expected IRR against that risk (right)."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.0))
    shown = [no_class_b] + [row for row in profile if row["relative_risk"] <= max_risk]
    sizes = [row["b_notional"] for row in shown]
    risks = [row["relative_risk"] for row in shown]
    irrs = [row["expected_irr"] for row in shown]
    solved = [row for row in limits if row is not None]

    left.plot(sizes, risks, color=COLORS["equity"], label="Python (exact)")
    left.scatter([row["goal_seek_b_notional"] for row in solved], [row["max_relative_risk"] for row in solved], s=24,
                 color=COLORS["class_b"], zorder=3, label="Goal Seek on the Leverage sheet")
    left.yaxis.set_major_formatter(PercentFormatter(1.0))
    left.set_ylim(0, max_risk)
    left.set_title("Relative equity risk: SD of the residual's PV / bank's stake")
    left.set_xlabel("Class B principal ($MM)")
    left.set_ylabel("Relative equity risk")
    left.legend(loc="upper left")

    right.plot(risks, irrs, color=COLORS["equity"])
    right.axhline(risk_free, color=COLORS["axis"], linewidth=0.8)
    right.annotate(f"risk-free {risk_free:.2%}", xy=(0.99, risk_free), xycoords=("axes fraction", "data"), xytext=(0, 3),
                   textcoords="offset points", ha="right", fontsize=8.5, color=COLORS["ink_secondary"])
    right.scatter([no_class_b["relative_risk"]], [no_class_b["expected_irr"]], s=24, color=COLORS["pool"], zorder=3)
    right.annotate("no Class B", xy=(no_class_b["relative_risk"], no_class_b["expected_irr"]), xytext=(6, -12),
                   textcoords="offset points", fontsize=8.5, color=COLORS["ink_secondary"])
    for row in solved:
        if any(abs(row["max_relative_risk"] - x) < 1e-12 for x in labelled):
            right.scatter([row["relative_risk"]], [row["expected_irr"]], s=24, color=COLORS["class_b"], zorder=3)
            right.annotate(f"Class B ${row['b_notional']:.1f} MM", xy=(row["relative_risk"], row["expected_irr"]),
                           xytext=(-6, 6), textcoords="offset points", ha="right", fontsize=8.5, color=COLORS["ink"])
    right.xaxis.set_major_formatter(PercentFormatter(1.0))
    right.yaxis.set_major_formatter(PercentFormatter(1.0))
    right.set_xlim(0, max_risk)
    right.set_ylim(0, max(irrs) * 1.1)
    right.set_title("Expected IRR against relative equity risk")
    right.set_xlabel("Relative equity risk")
    right.set_ylabel("Expected IRR (a year)")
    fig.tight_layout()
    return fig


def save_figure(fig: Figure, path: Path) -> None:
    """Write a figure to disk and release it."""
    fig.savefig(path)
    plt.close(fig)


def write_json(path: Path, payload: dict) -> None:
    """Write results as strict JSON (no NaN) so the browser can read the same file."""
    Path(path).write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n")
