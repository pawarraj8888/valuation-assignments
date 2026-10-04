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


def save_figure(fig: Figure, path: Path) -> None:
    """Write a figure to disk and release it."""
    fig.savefig(path)
    plt.close(fig)


def write_json(path: Path, payload: dict) -> None:
    """Write results as strict JSON (no NaN) so the browser can read the same file."""
    Path(path).write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n")
