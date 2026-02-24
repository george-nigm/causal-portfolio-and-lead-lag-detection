"""
Sector-based causal analysis pipeline with regime features.

Produces comprehensive results suitable for academic publication:

Use Case 1: Sector Rotation Detection
    Identify which sectors lead the business cycle via causal discovery.

Use Case 2: Crisis Contagion Mapping
    Track how causal links change when regimes shift — measure how
    financial shocks propagate to other sectors.

Use Case 3: Regime-Adaptive Portfolio
    Portfolio that adapts allocation based on detected regime and
    causal structure. Compare to static strategies.

Use Case 4: Method Comparison (Granger vs TE vs CCF vs PCMCI)
    Head-to-head detection accuracy across methods, broken down by
    regime, lag, and signal-to-noise ratio.

Use Case 5: Parameter Sensitivity Analysis
    How detection power varies with lead-lag strength, noise level,
    regime duration, and sample size.

Usage:
    python -m causal_trading.run_sectors [--fast] [--output-dir DIR]
"""
from __future__ import annotations

import os
import sys
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import seaborn as sns
from typing import Dict, List, Tuple, Optional

from .sectors import (
    SectorDataGenerator, SectorGenConfig, RegimeConfig, MacroConfig,
    SECTOR_INFO, SECTOR_TICKERS, MACRO_NAMES, REGIME_CAUSAL_LINKS,
)
from .config import CausalConfig, LeadLagConfig, BacktestConfig, PortfolioConfig, PairsConfig
from .causal import run_causal_analysis, CausalResult
from .lead_lag import rolling_lead_lag_detection
from .portfolio import optimize_portfolio
from .backtest import (
    backtest_portfolio, backtest_pairs,
    backtest_equal_weight_benchmark, BacktestResult,
)
from .pairs import select_pairs, CausalPair, generate_pair_signals

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ===========================================================================
# VISUALIZATION HELPERS
# ===========================================================================

REGIME_COLORS = {"expansion": "#2ecc71", "contraction": "#f39c12", "crisis": "#e74c3c"}
REGIME_CMAP = ListedColormap(["#2ecc71", "#f39c12", "#e74c3c"])


def _add_regime_shading(ax, dates, regime_sequence, regime_names, alpha=0.12):
    """Add regime background shading to a matplotlib axes."""
    for r_idx, r_name in enumerate(regime_names):
        mask = regime_sequence == r_idx
        for i in range(1, len(mask)):
            if mask[i]:
                ax.axvspan(dates[i - 1], dates[i],
                           color=REGIME_COLORS[r_name], alpha=alpha, linewidth=0)


def plot_regime_timeline(gen: SectorDataGenerator, save_path: str):
    """Plot regime sequence with macro variables dashboard."""
    fig, axes = plt.subplots(7, 1, figsize=(18, 16), sharex=True,
                              gridspec_kw={"height_ratios": [1, 2, 2, 2, 2, 2, 2]})

    dates = gen.macro_data.index
    regimes = gen.regime_sequence
    regime_names = gen.cfg.regime.regime_names

    # Panel 0: Regime timeline
    ax = axes[0]
    regime_arr = regimes.reshape(1, -1)
    ax.imshow(regime_arr, aspect="auto", cmap=REGIME_CMAP,
              extent=[0, len(dates), 0, 1], interpolation="nearest")
    ax.set_yticks([])
    ax.set_ylabel("Regime", fontsize=9)
    patches = [mpatches.Patch(color=REGIME_COLORS[r], label=r.capitalize())
               for r in regime_names]
    ax.legend(handles=patches, loc="upper right", ncol=3, fontsize=8)
    ax.set_title("Macro Regime Timeline & Variables", fontsize=13, fontweight="bold")

    # Panels 1-6: Macro variables
    for idx, macro_name in enumerate(MACRO_NAMES):
        ax = axes[idx + 1]
        vals = gen.macro_data[macro_name].values
        ax.plot(range(len(dates)), vals, "k-", linewidth=0.8)
        ax.fill_between(range(len(dates)), vals, 0,
                        where=vals > 0, color="#2ecc71", alpha=0.3)
        ax.fill_between(range(len(dates)), vals, 0,
                        where=vals <= 0, color="#e74c3c", alpha=0.3)
        _add_regime_shading(ax, range(len(dates)), regimes, regime_names, alpha=0.08)
        ax.set_ylabel(macro_name.replace("_", " ").title(), fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.axhline(0, color="gray", linewidth=0.5)

    # Set x-axis labels at sensible intervals
    n_ticks = 12
    tick_positions = np.linspace(0, len(dates) - 1, n_ticks, dtype=int)
    axes[-1].set_xticks(tick_positions)
    axes[-1].set_xticklabels([dates[i].strftime("%Y-%m") for i in tick_positions],
                              rotation=45, fontsize=8)
    axes[-1].set_xlabel("Date")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sector_prices(gen: SectorDataGenerator, save_path: str):
    """Plot normalized sector prices with regime shading."""
    fig, ax = plt.subplots(figsize=(18, 8))

    prices = gen.sector_prices
    norm_prices = prices / prices.iloc[0] * 100
    dates = prices.index

    colors = plt.cm.tab20(np.linspace(0, 1, len(SECTOR_TICKERS)))
    for i, ticker in enumerate(SECTOR_TICKERS):
        label = f"{ticker} ({SECTOR_INFO[ticker]['name']})"
        ax.plot(dates, norm_prices[ticker], linewidth=1.0, color=colors[i], label=label)

    _add_regime_shading(ax, dates, gen.regime_sequence,
                        gen.cfg.regime.regime_names, alpha=0.15)

    ax.set_ylabel("Normalized Price (base=100)")
    ax.set_title("SPDR Sector ETF Prices (Synthetic) with Regime Shading", fontweight="bold")
    ax.legend(loc="upper left", fontsize=7, ncol=3)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_causal_heatmaps_by_regime(
    results_by_regime: Dict[str, Dict[str, CausalResult]],
    method: str,
    alpha: float,
    save_path: str,
):
    """Plot causal p-value heatmaps side-by-side for each regime."""
    regimes = list(results_by_regime.keys())
    n_regimes = len(regimes)

    fig, axes = plt.subplots(1, n_regimes, figsize=(7 * n_regimes, 6))
    if n_regimes == 1:
        axes = [axes]

    for idx, regime in enumerate(regimes):
        result = results_by_regime[regime].get(method)
        if result is None:
            continue

        p_df = pd.DataFrame(result.p_matrix, index=result.assets, columns=result.assets)
        mask = np.eye(len(result.assets), dtype=bool)

        sns.heatmap(
            p_df, mask=mask, ax=axes[idx],
            cmap="RdYlGn_r", vmin=0, vmax=0.10,
            annot=True, fmt=".2f", linewidths=0.5,
            annot_kws={"fontsize": 7},
        )
        n_links = (p_df.values[~mask] < alpha).sum()
        axes[idx].set_title(
            f"{regime.upper()}\n({method}, {n_links} significant links)",
            fontsize=11, fontweight="bold",
            color=REGIME_COLORS[regime],
        )

    plt.suptitle(f"Causal P-Value Matrices by Regime — {method}",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_causal_networks_by_regime(
    results_by_regime: Dict[str, Dict[str, CausalResult]],
    method: str,
    alpha: float,
    ground_truth: Dict,
    save_path: str,
):
    """Plot causal networks as directed graphs for each regime."""
    regimes = list(results_by_regime.keys())
    n_regimes = len(regimes)

    fig, axes = plt.subplots(1, n_regimes, figsize=(8 * n_regimes, 8))
    if n_regimes == 1:
        axes = [axes]

    # Node layout (circle)
    n = len(SECTOR_TICKERS)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) - np.pi / 2
    node_x = np.cos(angles)
    node_y = np.sin(angles)

    for idx, regime in enumerate(regimes):
        ax = axes[idx]
        result = results_by_regime[regime].get(method)
        if result is None:
            continue

        links = result.get_significant_links(alpha)
        gt_links = {(l["leader"], l["follower"]) for l in ground_truth.get(regime, [])}
        detected_links = {(c, e) for c, e, _, _, _ in links}

        # Draw edges
        asset_to_idx = {a: i for i, a in enumerate(result.assets)}
        for cause, effect, lag, strength, pval in links:
            ci = asset_to_idx[cause]
            ei = asset_to_idx[effect]
            is_gt = (cause, effect) in gt_links
            color = "#2ecc71" if is_gt else "#e74c3c"
            ax.annotate(
                "", xy=(node_x[ei], node_y[ei]), xytext=(node_x[ci], node_y[ci]),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                alpha=0.7, lw=1.8, mutation_scale=15),
            )

        # Draw ground truth links that were missed (dashed)
        for gt_link in gt_links:
            if gt_link not in detected_links:
                ci = asset_to_idx.get(gt_link[0])
                ei = asset_to_idx.get(gt_link[1])
                if ci is not None and ei is not None:
                    ax.annotate(
                        "", xy=(node_x[ei], node_y[ei]),
                        xytext=(node_x[ci], node_y[ci]),
                        arrowprops=dict(arrowstyle="-|>", color="gray",
                                        alpha=0.4, lw=1.2, linestyle="--",
                                        mutation_scale=12),
                    )

        # Draw nodes
        scores = result.get_leader_scores(alpha)
        max_count = max((s["count"] for s in scores.values()), default=1) or 1
        for i, asset in enumerate(result.assets):
            count = scores[asset]["count"]
            size = 400 + 600 * (count / max_count)
            color = plt.cm.YlOrRd(0.2 + 0.7 * count / max_count)
            ax.scatter(node_x[i], node_y[i], s=size, color=color,
                       zorder=5, edgecolors="black", linewidth=1.2)
            offset = 0.15
            ax.text(node_x[i] * (1 + offset), node_y[i] * (1 + offset),
                    asset, ha="center", va="center", fontsize=8, fontweight="bold")

        tp = len(detected_links & gt_links)
        fp = len(detected_links - gt_links)
        fn = len(gt_links - detected_links)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.set_aspect("equal")
        ax.set_title(
            f"{regime.upper()}\n"
            f"P={precision:.0%} R={recall:.0%} | "
            f"TP={tp} FP={fp} FN={fn}",
            fontsize=11, fontweight="bold",
            color=REGIME_COLORS[regime],
        )
        ax.axis("off")

    # Legend
    legend_elements = [
        mpatches.Patch(color="#2ecc71", label="True Positive"),
        mpatches.Patch(color="#e74c3c", label="False Positive"),
        mpatches.Patch(color="gray", label="Missed (FN)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=10)

    plt.suptitle(f"Causal Networks by Regime — {method}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_method_comparison(
    results_by_regime: Dict[str, Dict[str, CausalResult]],
    ground_truth: Dict,
    alpha: float,
    save_path: str,
):
    """Bar chart comparing detection accuracy across methods and regimes."""
    methods = list(next(iter(results_by_regime.values())).keys())
    regimes = list(results_by_regime.keys())

    metrics = {m: {"precision": [], "recall": [], "f1": []} for m in methods}
    x_labels = []

    for regime in regimes:
        gt_links = {(l["leader"], l["follower"]) for l in ground_truth.get(regime, [])}
        for method in methods:
            result = results_by_regime[regime].get(method)
            if result is None:
                for metric in metrics[method]:
                    metrics[method][metric].append(0)
                continue
            detected = {(c, e) for c, e, _, _, _ in result.get_significant_links(alpha)}
            tp = len(detected & gt_links)
            fp = len(detected - gt_links)
            fn = len(gt_links - detected)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            metrics[method]["precision"].append(p)
            metrics[method]["recall"].append(r)
            metrics[method]["f1"].append(f1)
        x_labels.append(regime.capitalize())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metric_names = ["precision", "recall", "f1"]
    titles = ["Precision", "Recall", "F1 Score"]
    x = np.arange(len(regimes))
    width = 0.8 / len(methods)

    for ax_idx, (metric_key, title) in enumerate(zip(metric_names, titles)):
        ax = axes[ax_idx]
        for m_idx, method in enumerate(methods):
            offset = (m_idx - len(methods) / 2 + 0.5) * width
            bars = ax.bar(x + offset, metrics[method][metric_key],
                          width, label=method, alpha=0.85)
            for bar, val in zip(bars, metrics[method][metric_key]):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                            f"{val:.0%}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel(title)
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2, axis="y")

    plt.suptitle("Causal Detection Method Comparison by Regime",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rolling_causal_with_regimes(
    gen: SectorDataGenerator,
    network,
    top_n: int,
    alpha: float,
    save_path: str,
):
    """Plot rolling causal dynamics with true regime shading."""
    stable = network.get_stable_links(alpha=alpha, min_stability=0.1)
    if stable.empty:
        return
    stable = stable.head(top_n)

    fig, axes = plt.subplots(top_n + 1, 1, figsize=(18, 3 * (top_n + 1)),
                              gridspec_kw={"height_ratios": [0.5] + [1] * top_n})

    dates = gen.macro_data.index
    regimes = gen.regime_sequence
    regime_names = gen.cfg.regime.regime_names

    # Panel 0: Regime bar (use simple colored spans instead of imshow)
    ax = axes[0]
    for r_idx, r_name in enumerate(regime_names):
        mask = regimes == r_idx
        for i in range(len(mask)):
            if mask[i]:
                ax.axvspan(i - 0.5, i + 0.5, color=REGIME_COLORS[r_name], alpha=0.8, linewidth=0)
    ax.set_xlim(0, len(dates))
    ax.set_yticks([])
    ax.set_ylabel("Regime")
    patches = [mpatches.Patch(color=REGIME_COLORS[r], label=r.capitalize())
               for r in regime_names]
    ax.legend(handles=patches, loc="upper right", ncol=3, fontsize=8)

    # Build date-to-index mapping for link panels
    date_to_idx = {d: i for i, d in enumerate(dates)}

    # Link panels
    for idx, (_, row) in enumerate(stable.iterrows()):
        link = network.links[(row["cause"], row["effect"])]
        ts = link.to_series()
        if ts.empty:
            continue
        ax = axes[idx + 1]

        # Check if this is a ground truth link for any regime
        gt_regimes = []
        for r_name, r_links in REGIME_CAUSAL_LINKS.items():
            for l, f, lag, s in r_links:
                if l == row["cause"] and f == row["effect"]:
                    gt_regimes.append(r_name)

        # Map dates to integer indices for consistent x-axis
        x_indices = [date_to_idx.get(d, 0) for d in ts.index]
        p_values = ts["p_value"].values

        ax.fill_between(x_indices, 0, 1,
                        where=p_values < alpha,
                        alpha=0.2, color="green", label="Significant")
        ax.plot(x_indices, p_values, "k-", linewidth=0.8)
        ax.axhline(alpha, color="red", linestyle="--", linewidth=0.5)
        ax.set_ylabel("p-value", fontsize=8)
        ax.set_xlim(0, len(dates))

        gt_str = f" [GT: {', '.join(gt_regimes)}]" if gt_regimes else " [NOT in GT]"
        ax.set_title(
            f"{row['cause']} → {row['effect']} "
            f"(stability={row['stability']:.0%}){gt_str}",
            fontsize=9, fontweight="bold",
        )
        ax.set_ylim(0, 0.20)

    # Set x-axis labels on bottom panel
    n_ticks = 12
    tick_positions = np.linspace(0, len(dates) - 1, n_ticks, dtype=int)
    axes[-1].set_xticks(tick_positions)
    axes[-1].set_xticklabels([dates[i].strftime("%Y-%m") for i in tick_positions],
                              rotation=45, fontsize=8)

    plt.suptitle("Rolling Lead-Lag Dynamics with True Regime Overlay",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_portfolio_comparison(
    results: List[BacktestResult],
    gen: SectorDataGenerator,
    save_path: str,
):
    """Plot portfolio value comparison with regime shading."""
    fig, axes = plt.subplots(2, 1, figsize=(18, 10),
                              gridspec_kw={"height_ratios": [3, 1]})

    dates_full = gen.macro_data.index
    regimes = gen.regime_sequence
    regime_names = gen.cfg.regime.regime_names

    # Panel 1: Portfolio values
    ax = axes[0]
    for res in results:
        if res.portfolio_value is not None and len(res.portfolio_value) > 0:
            ax.plot(res.portfolio_value.index, res.portfolio_value.values,
                    label=res.name, linewidth=1.2)
    # Add regime shading aligned to OOS period
    if results and results[0].portfolio_value is not None:
        oos_start = results[0].portfolio_value.index[0]
        oos_end = results[0].portfolio_value.index[-1]
        oos_mask = (dates_full >= oos_start) & (dates_full <= oos_end)
        oos_dates = dates_full[oos_mask]
        oos_regimes = regimes[oos_mask]
        _add_regime_shading(ax, oos_dates, oos_regimes, regime_names, alpha=0.12)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_title("Strategy Comparison with Regime Shading", fontweight="bold")
    ax.grid(True, alpha=0.2)

    # Panel 2: Drawdowns
    ax = axes[1]
    for res in results:
        if res.returns is not None and len(res.returns) > 0:
            cum = (1 + res.returns).cumprod()
            dd = (cum - cum.cummax()) / cum.cummax()
            ax.fill_between(dd.index, dd.values, 0, alpha=0.3, label=res.name)
    ax.legend(fontsize=8)
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity_heatmap(results_df: pd.DataFrame, save_path: str):
    """Plot parameter sensitivity analysis heatmaps."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    metric_cols = ["recall", "precision", "f1"]
    titles = ["Recall (Detection Rate)", "Precision (False Positive Control)", "F1 Score"]

    for ax, metric, title in zip(axes, metric_cols, titles):
        pivot = results_df.pivot_table(
            index="noise_scale", columns="lead_lag_strength",
            values=metric, aggfunc="mean",
        )
        sns.heatmap(pivot, ax=ax, cmap="RdYlGn", vmin=0, vmax=1,
                    annot=True, fmt=".0%", linewidths=0.5)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Lead-Lag Strength Multiplier")
        ax.set_ylabel("Noise Scale Multiplier")

    plt.suptitle("Detection Accuracy: Signal Strength vs. Noise Level",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sector_rotation_clock(
    results_by_regime: Dict[str, Dict[str, CausalResult]],
    method: str,
    alpha: float,
    save_path: str,
):
    """Plot sector leadership rotation across regimes as a circular diagram."""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})

    regimes = list(results_by_regime.keys())
    n_sectors = len(SECTOR_TICKERS)

    # Get leader scores per regime
    sector_angles = np.linspace(0, 2 * np.pi, n_sectors, endpoint=False)

    for r_idx, regime in enumerate(regimes):
        result = results_by_regime[regime].get(method)
        if result is None:
            continue
        scores = result.get_leader_scores(alpha)
        leader_counts = [scores.get(t, {"count": 0})["count"] for t in SECTOR_TICKERS]
        max_count = max(leader_counts) or 1
        normalized = [c / max_count for c in leader_counts]

        # Offset for each regime
        offset = r_idx * 0.25
        values = [n + offset for n in normalized]
        values.append(values[0])  # Close the polygon
        angles = np.append(sector_angles, sector_angles[0])

        ax.plot(angles, values, "o-", color=REGIME_COLORS[regime],
                label=regime.capitalize(), linewidth=2, markersize=6, alpha=0.8)
        ax.fill(angles, values, color=REGIME_COLORS[regime], alpha=0.1)

    ax.set_xticks(sector_angles)
    ax.set_xticklabels(SECTOR_TICKERS, fontsize=9, fontweight="bold")
    ax.set_title("Sector Leadership Rotation by Regime\n(normalized leader scores)",
                 fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_contagion_cascade(
    crisis_result: CausalResult,
    alpha: float,
    save_path: str,
):
    """Visualize crisis contagion as a cascade/waterfall diagram."""
    links = crisis_result.get_significant_links(alpha)
    if not links:
        return

    # Build adjacency by lag
    lag_links = {}
    for cause, effect, lag, strength, pval in links:
        lag_links.setdefault(lag, []).append((cause, effect, strength))

    max_lag = max(lag_links.keys()) if lag_links else 0

    fig, ax = plt.subplots(figsize=(16, 8))

    # Position sectors vertically
    sector_y = {t: i for i, t in enumerate(SECTOR_TICKERS)}

    for lag, pairs in sorted(lag_links.items()):
        for cause, effect, strength in pairs:
            y1 = sector_y[cause]
            y2 = sector_y[effect]
            color_intensity = min(1.0, abs(strength) / 0.5)
            ax.annotate(
                "", xy=(lag + 0.4, y2), xytext=(lag - 0.4, y1),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=plt.cm.Reds(0.3 + 0.7 * color_intensity),
                    lw=1.5 + 2.0 * color_intensity,
                    mutation_scale=15,
                ),
            )

    # Draw sector labels
    for ticker, y in sector_y.items():
        name = SECTOR_INFO[ticker]["name"]
        ax.text(-0.8, y, f"{ticker}\n{name}", ha="right", va="center",
                fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

    ax.set_xlim(-1.5, max_lag + 1)
    ax.set_ylim(-1, len(SECTOR_TICKERS))
    ax.set_xlabel("Lag (days)", fontsize=11)
    ax.set_title("Crisis Contagion Cascade\n(arrow width ∝ strength)",
                 fontsize=13, fontweight="bold")
    ax.set_yticks([])
    ax.grid(True, alpha=0.2, axis="x")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_regime_distributions(gen: SectorDataGenerator, save_path: str):
    """Show how computed features distribute differently across regimes."""
    features = gen.all_features
    regimes = gen.regime_sequence
    regime_names = gen.cfg.regime.regime_names

    feature_cols = [
        "realized_vol", "market_trend", "avg_correlation",
        "cross_section_dispersion", "momentum_spread", "avg_lagged_correlation",
    ]
    feature_titles = [
        "Realized Volatility", "Market Trend (60d)",
        "Avg Cross-Sector Correlation", "Cross-Section Dispersion",
        "Momentum Spread (Top-Bottom)", "Avg Lagged Correlation",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (col, title) in enumerate(zip(feature_cols, feature_titles)):
        ax = axes[idx]
        for r_idx, r_name in enumerate(regime_names):
            mask = regimes == r_idx
            vals = features.loc[mask, col].dropna()
            if len(vals) > 10:
                ax.hist(vals, bins=40, alpha=0.5, density=True,
                        color=REGIME_COLORS[r_name], label=r_name.capitalize())
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    plt.suptitle("Feature Distributions by Regime\n(auto-computed from sector data)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# USE CASES
# ===========================================================================

def use_case_1_sector_rotation(
    gen: SectorDataGenerator,
    causal_cfg: CausalConfig,
    output_dir: str,
) -> Dict[str, Dict[str, CausalResult]]:
    """
    USE CASE 1: Sector Rotation Detection

    Run causal analysis separately on each regime's data.
    Identify which sectors lead in each macroeconomic regime.
    Compare the detected structure to the ground truth.
    """
    print("\n" + "=" * 80)
    print("  USE CASE 1: SECTOR ROTATION DETECTION")
    print("=" * 80)

    returns_by_regime = gen.get_returns_by_regime()
    results_by_regime = {}

    for regime, ret in returns_by_regime.items():
        print(f"\n  --- {regime.upper()} ({len(ret)} days) ---")

        if len(ret) < 100:
            print(f"    Too few observations, skipping")
            continue

        results = run_causal_analysis(ret, causal_cfg)
        results_by_regime[regime] = results

        gt_links = {(l["leader"], l["follower"])
                    for l in gen.ground_truth.get(regime, [])}

        for method_name, result in results.items():
            links = result.get_significant_links(causal_cfg.significance_level)
            detected = {(c, e) for c, e, _, _, _ in links}

            tp = len(detected & gt_links)
            fp = len(detected - gt_links)
            fn = len(gt_links - detected)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            print(f"    {method_name:20s}: {len(links):2d} links | "
                  f"P={precision:.0%} R={recall:.0%} | "
                  f"TP={tp} FP={fp} FN={fn}")

        # Leader rankings
        for method_name, result in results.items():
            scores = result.get_leader_scores(causal_cfg.significance_level)
            ranked = sorted(scores.items(), key=lambda x: x[1]["count"], reverse=True)
            top_leaders = [(name, info["count"]) for name, info in ranked if info["count"] > 0]
            if top_leaders:
                leaders_str = ", ".join(f"{n}({c})" for n, c in top_leaders[:5])
                print(f"    {method_name:20s} leaders: {leaders_str}")

    # Generate plots
    for method in causal_cfg.methods:
        if any(method in r for r in results_by_regime.values()):
            plot_causal_heatmaps_by_regime(
                results_by_regime, method, causal_cfg.significance_level,
                os.path.join(output_dir, f"uc1_heatmaps_{method}.png"),
            )
            plot_causal_networks_by_regime(
                results_by_regime, method, causal_cfg.significance_level,
                gen.ground_truth,
                os.path.join(output_dir, f"uc1_networks_{method}.png"),
            )

    # Rotation clock
    primary_method = causal_cfg.methods[0]
    plot_sector_rotation_clock(
        results_by_regime, primary_method, causal_cfg.significance_level,
        os.path.join(output_dir, "uc1_rotation_clock.png"),
    )

    return results_by_regime


def use_case_2_crisis_contagion(
    gen: SectorDataGenerator,
    causal_cfg: CausalConfig,
    output_dir: str,
):
    """
    USE CASE 2: Crisis Contagion Mapping

    Focus on crisis periods. Analyze how financial shocks propagate
    from sector to sector via causal lead-lag relationships.
    """
    print("\n" + "=" * 80)
    print("  USE CASE 2: CRISIS CONTAGION MAPPING")
    print("=" * 80)

    returns_by_regime = gen.get_returns_by_regime()
    crisis_returns = returns_by_regime.get("crisis")

    if crisis_returns is None or len(crisis_returns) < 50:
        print("  Not enough crisis data for analysis")
        return

    print(f"\n  Crisis data: {len(crisis_returns)} days")

    results = run_causal_analysis(crisis_returns, causal_cfg)

    primary_method = causal_cfg.methods[0]
    result = results[primary_method]
    links = result.get_significant_links(causal_cfg.significance_level)

    print(f"\n  Crisis contagion links ({primary_method}):")
    for cause, effect, lag, strength, pval in links:
        gt_mark = ""
        for l in gen.ground_truth.get("crisis", []):
            if l["leader"] == cause and l["follower"] == effect:
                gt_mark = " [GT]"
                break
        print(f"    {cause:5s} → {effect:5s}  lag={lag}  p={pval:.4f}  str={strength:.4f}{gt_mark}")

    # Contagion source analysis
    scores = result.get_leader_scores(causal_cfg.significance_level)
    print(f"\n  Crisis contagion sources:")
    ranked = sorted(scores.items(), key=lambda x: x[1]["count"], reverse=True)
    for name, info in ranked:
        if info["count"] > 0:
            print(f"    {name}: causes {info['count']} sectors (total strength: {info['strength']:.3f})")

    # Lag structure analysis
    print(f"\n  Contagion speed (by lag):")
    lag_counts = {}
    for _, _, lag, _, _ in links:
        lag_counts[lag] = lag_counts.get(lag, 0) + 1
    for lag in sorted(lag_counts):
        print(f"    Lag {lag}: {lag_counts[lag]} relationships")

    plot_contagion_cascade(
        result, causal_cfg.significance_level,
        os.path.join(output_dir, "uc2_contagion_cascade.png"),
    )


def use_case_3_regime_portfolio(
    gen: SectorDataGenerator,
    causal_cfg: CausalConfig,
    output_dir: str,
) -> List[BacktestResult]:
    """
    USE CASE 3: Regime-Adaptive Portfolio

    Compare portfolio strategies:
    1. Equal Weight (benchmark)
    2. Inverse Volatility
    3. Minimum Variance
    4. Causal-Weighted (static)
    5. Regime-Adaptive Causal (re-estimates causal structure periodically)
    """
    print("\n" + "=" * 80)
    print("  USE CASE 3: REGIME-ADAPTIVE PORTFOLIO")
    print("=" * 80)

    prices = gen.sector_prices
    returns = gen.sector_returns

    bt_cfg = BacktestConfig(
        initial_capital=100000.0,
        commission_pct=0.001,
        slippage_pct=0.0005,
        train_fraction=0.4,
    )
    port_cfg = PortfolioConfig(rebalance_days=21, leader_tilt=0.3)

    backtest_results = []

    # a) Equal weight benchmark
    bm = backtest_equal_weight_benchmark(returns, bt_cfg)
    backtest_results.append(bm)

    # b) Standard portfolio strategies
    for method in ["equal_weight", "inverse_vol", "min_variance", "causal_weighted"]:
        port_cfg.method = method
        res = backtest_portfolio(prices, returns, bt_cfg, port_cfg, causal_cfg)
        backtest_results.append(res)

    # c) Pairs trading from causal analysis
    # Get stable links from rolling analysis
    ll_cfg = LeadLagConfig(
        rolling_window=252, rolling_step=42,
        method="granger", significance_level=0.05,
    )
    print("\n  Running rolling analysis for pair selection...")
    network = rolling_lead_lag_detection(returns, ll_cfg, causal_cfg)

    pairs_cfg = PairsConfig(max_pairs=5, entry_z=2.0, exit_z=0.5)
    pairs = select_pairs(network, pairs_cfg, ll_cfg.significance_level)
    if pairs:
        print(f"  Selected {len(pairs)} causal pairs:")
        for p in pairs:
            print(f"    {p}")
        pairs_result = backtest_pairs(prices, pairs, bt_cfg, pairs_cfg)
        backtest_results.append(pairs_result)

    # Print results
    print(f"\n  {'Strategy':<35s} {'Sharpe':>8s} {'CAGR':>8s} {'MaxDD':>8s} {'Sortino':>8s}")
    print("  " + "-" * 70)
    for res in backtest_results:
        if res.metrics:
            print(f"  {res.name:<35s} "
                  f"{res.metrics.get('sharpe', 0):>8.3f} "
                  f"{res.metrics.get('cagr', 0):>+7.2%} "
                  f"{res.metrics.get('max_drawdown', 0):>+7.2%} "
                  f"{res.metrics.get('sortino', 0):>8.3f}")

    plot_portfolio_comparison(
        backtest_results, gen,
        os.path.join(output_dir, "uc3_portfolio_comparison.png"),
    )

    return backtest_results


def use_case_4_method_comparison(
    results_by_regime: Dict[str, Dict[str, CausalResult]],
    gen: SectorDataGenerator,
    causal_cfg: CausalConfig,
    output_dir: str,
):
    """
    USE CASE 4: Method Comparison

    Compare Granger, Transfer Entropy, CCF, PCMCI-lite across regimes.
    Analyze each method's strengths and weaknesses.
    """
    print("\n" + "=" * 80)
    print("  USE CASE 4: METHOD COMPARISON")
    print("=" * 80)

    alpha = causal_cfg.significance_level

    # Overall comparison table
    print(f"\n  {'Method':<20s} {'Regime':<15s} {'Links':>6s} {'Prec':>6s} {'Recall':>6s} {'F1':>6s}")
    print("  " + "-" * 70)

    all_rows = []
    for regime, results in results_by_regime.items():
        gt_links = {(l["leader"], l["follower"])
                    for l in gen.ground_truth.get(regime, [])}

        for method_name, result in results.items():
            detected = {(c, e) for c, e, _, _, _ in result.get_significant_links(alpha)}
            tp = len(detected & gt_links)
            fp = len(detected - gt_links)
            fn = len(gt_links - detected)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

            print(f"  {method_name:<20s} {regime:<15s} {len(detected):>6d} "
                  f"{p:>5.0%} {r:>6.0%} {f1:>5.0%}")

            all_rows.append({
                "method": method_name, "regime": regime,
                "n_links": len(detected),
                "precision": p, "recall": r, "f1": f1,
            })

    # Aggregate across regimes
    agg_df = pd.DataFrame(all_rows)
    print(f"\n  --- Aggregated (mean across regimes) ---")
    agg = agg_df.groupby("method")[["precision", "recall", "f1"]].mean()
    for method, row in agg.iterrows():
        print(f"  {method:<20s} P={row['precision']:.0%}  R={row['recall']:.0%}  F1={row['f1']:.0%}")

    # Lag accuracy analysis
    print(f"\n  --- Lag Estimation Accuracy ---")
    for regime, results in results_by_regime.items():
        gt_dict = {(l["leader"], l["follower"]): l["lag"]
                   for l in gen.ground_truth.get(regime, [])}
        for method_name, result in results.items():
            links = result.get_significant_links(alpha)
            lag_errors = []
            for cause, effect, detected_lag, _, _ in links:
                true_lag = gt_dict.get((cause, effect))
                if true_lag is not None:
                    lag_errors.append(abs(detected_lag - true_lag))
            if lag_errors:
                print(f"  {method_name:<20s} {regime:<15s} "
                      f"mean lag error: {np.mean(lag_errors):.1f} days "
                      f"(exact: {sum(1 for e in lag_errors if e == 0)}/{len(lag_errors)})")

    plot_method_comparison(
        results_by_regime, gen.ground_truth, alpha,
        os.path.join(output_dir, "uc4_method_comparison.png"),
    )


def use_case_5_parameter_sensitivity(
    causal_cfg: CausalConfig,
    output_dir: str,
    fast: bool = False,
):
    """
    USE CASE 5: Parameter Sensitivity Analysis

    Sweep over lead-lag strength and noise scale to understand
    how detection power depends on signal-to-noise ratio.
    """
    print("\n" + "=" * 80)
    print("  USE CASE 5: PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 80)

    if fast:
        strengths = [0.5, 1.0, 1.5, 2.0]
        noises = [0.5, 1.0, 1.5, 2.0]
        n_obs = 1500
    else:
        strengths = [0.3, 0.6, 1.0, 1.5, 2.0]
        noises = [0.5, 0.8, 1.0, 1.5, 2.0]
        n_obs = 2000

    # Use only the fastest method for sweep
    sweep_cfg = CausalConfig(
        methods=["granger"],
        max_lag=causal_cfg.max_lag,
        significance_level=causal_cfg.significance_level,
    )

    rows = []
    total = len(strengths) * len(noises)
    count = 0

    for strength in strengths:
        for noise in noises:
            count += 1
            print(f"  [{count}/{total}] strength={strength:.1f}, noise={noise:.1f}...", end="")

            cfg = SectorGenConfig(
                n_obs=n_obs,
                lead_lag_strength=strength,
                noise_scale=noise,
                seed=42,
            )
            gen = SectorDataGenerator(cfg)
            gen.generate()

            returns_by_regime = gen.get_returns_by_regime()

            total_tp, total_fp, total_fn = 0, 0, 0
            for regime, ret in returns_by_regime.items():
                if len(ret) < 80:
                    continue
                results = run_causal_analysis(ret, sweep_cfg)
                result = results["granger"]
                detected = {(c, e) for c, e, _, _, _
                            in result.get_significant_links(sweep_cfg.significance_level)}
                gt = {(l["leader"], l["follower"])
                      for l in gen.ground_truth.get(regime, [])}
                total_tp += len(detected & gt)
                total_fp += len(detected - gt)
                total_fn += len(gt - detected)

            p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

            print(f" P={p:.0%} R={r:.0%} F1={f1:.0%}")
            rows.append({
                "lead_lag_strength": strength,
                "noise_scale": noise,
                "precision": p,
                "recall": r,
                "f1": f1,
            })

    results_df = pd.DataFrame(rows)

    plot_sensitivity_heatmap(
        results_df,
        os.path.join(output_dir, "uc5_sensitivity_heatmap.png"),
    )

    # Print summary
    print(f"\n  Best configuration: "
          f"strength={results_df.loc[results_df['f1'].idxmax(), 'lead_lag_strength']}, "
          f"noise={results_df.loc[results_df['f1'].idxmax(), 'noise_scale']}, "
          f"F1={results_df['f1'].max():.0%}")

    return results_df


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

def run_sector_pipeline(
    fast: bool = False,
    output_dir: str = "output_sectors",
):
    """Run the complete sector-based analysis with all use cases."""
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    print("\n" + "#" * 80)
    print("  SECTOR-BASED CAUSAL ANALYSIS WITH REGIME FEATURES")
    print("  SPDR Sector ETFs | Macro Regime Switching | 5 Use Cases")
    print("#" * 80)

    # ----------------------------------------------------------------
    # Configuration
    # ----------------------------------------------------------------
    if fast:
        gen_cfg = SectorGenConfig(n_obs=2000, seed=42)
        causal_cfg = CausalConfig(
            methods=["granger", "ccf"],
            max_lag=5,
            significance_level=0.05,
        )
    else:
        gen_cfg = SectorGenConfig(n_obs=3000, seed=42)
        causal_cfg = CausalConfig(
            methods=["granger", "transfer_entropy", "ccf"],
            max_lag=8,
            significance_level=0.05,
        )

    # ----------------------------------------------------------------
    # Step 0: Generate Sector Data with Regime Features
    # ----------------------------------------------------------------
    print("\n[0/7] Generating sector data with macro regimes...")
    gen = SectorDataGenerator(gen_cfg)
    prices, features, ground_truth_df = gen.generate()

    print(f"  Prices: {prices.shape} ({prices.index[0].date()} to {prices.index[-1].date()})")
    print(f"  Features: {features.shape} ({features.columns.tolist()[:8]}...)")
    print(f"  Ground truth: {len(ground_truth_df)} causal links across "
          f"{ground_truth_df['regime'].nunique()} regimes")

    # Regime statistics
    regime_stats = gen.get_regime_stats()
    print(f"\n  Regime Statistics:")
    print(f"  {'Regime':<15s} {'Days':>6s} {'Frac':>6s} {'Ann.Ret':>8s} "
          f"{'Ann.Vol':>8s} {'Corr':>6s} {'Links':>6s}")
    for _, row in regime_stats.iterrows():
        print(f"  {row['regime']:<15s} {row['n_days']:>6.0f} {row['fraction']:>5.0%} "
              f"{row['avg_annual_return']:>+7.2%} {row['avg_annual_vol']:>7.2%} "
              f"{row['avg_cross_corr']:>5.2f} {row['n_causal_links']:>6.0f}")

    # Regime periods
    periods = gen.get_regime_periods()
    print(f"\n  Regime periods: {len(periods)} transitions")
    for _, p in periods.head(10).iterrows():
        print(f"    {p['regime']:<15s} {p['start'].date()} to {p['end'].date()} ({p['duration_days']}d)")
    if len(periods) > 10:
        print(f"    ... ({len(periods) - 10} more)")

    # Ground truth
    print(f"\n  Ground Truth Causal Links:")
    for regime in ["expansion", "contraction", "crisis"]:
        links = gen.ground_truth.get(regime, [])
        print(f"    {regime}: {len(links)} links")
        for l in links:
            print(f"      {l['leader']:>5s} → {l['follower']:<5s}  "
                  f"lag={l['lag']}  strength={l['strength']:.3f}")

    # ----------------------------------------------------------------
    # Step 1: Visualize data
    # ----------------------------------------------------------------
    print("\n[1/7] Generating data visualizations...")
    plot_regime_timeline(gen, os.path.join(output_dir, "regime_timeline.png"))
    plot_sector_prices(gen, os.path.join(output_dir, "sector_prices.png"))
    plot_feature_regime_distributions(gen, os.path.join(output_dir, "feature_distributions.png"))
    print("  Saved: regime_timeline.png, sector_prices.png, feature_distributions.png")

    # ----------------------------------------------------------------
    # Step 2: Use Case 1 — Sector Rotation
    # ----------------------------------------------------------------
    print("\n[2/7] Use Case 1: Sector Rotation Detection...")
    results_by_regime = use_case_1_sector_rotation(gen, causal_cfg, output_dir)

    # ----------------------------------------------------------------
    # Step 3: Use Case 2 — Crisis Contagion
    # ----------------------------------------------------------------
    print("\n[3/7] Use Case 2: Crisis Contagion Mapping...")
    use_case_2_crisis_contagion(gen, causal_cfg, output_dir)

    # ----------------------------------------------------------------
    # Step 4: Use Case 3 — Regime-Adaptive Portfolio
    # ----------------------------------------------------------------
    print("\n[4/7] Use Case 3: Regime-Adaptive Portfolio...")
    portfolio_cfg = CausalConfig(
        methods=["granger"],
        max_lag=5,
        significance_level=0.05,
    )
    backtest_results = use_case_3_regime_portfolio(gen, portfolio_cfg, output_dir)

    # ----------------------------------------------------------------
    # Step 5: Use Case 4 — Method Comparison
    # ----------------------------------------------------------------
    print("\n[5/7] Use Case 4: Method Comparison...")
    if results_by_regime:
        use_case_4_method_comparison(results_by_regime, gen, causal_cfg, output_dir)

    # ----------------------------------------------------------------
    # Step 6: Rolling Analysis with Regime Overlay
    # ----------------------------------------------------------------
    print("\n[6/7] Rolling Lead-Lag Analysis with Regime Overlay...")
    ll_cfg = LeadLagConfig(
        rolling_window=252,
        rolling_step=42 if not fast else 63,
        method="granger",
        significance_level=0.05,
    )
    network = rolling_lead_lag_detection(gen.sector_returns, ll_cfg, causal_cfg)

    stable_links = network.get_stable_links(ll_cfg.significance_level, min_stability=0.3)
    print(f"\n  Stable links (>30% of windows): {len(stable_links)}")
    if not stable_links.empty:
        for _, row in stable_links.head(15).iterrows():
            # Check ground truth membership
            gt_regimes = []
            for r_name, links in REGIME_CAUSAL_LINKS.items():
                for l, f, lag, s in links:
                    if l == row["cause"] and f == row["effect"]:
                        gt_regimes.append(r_name)
            gt_str = f" [{', '.join(gt_regimes)}]" if gt_regimes else ""
            print(f"    {row['cause']:>5s} → {row['effect']:<5s}  "
                  f"stability={row['stability']:.0%}  "
                  f"active={row['currently_active']}{gt_str}")

    plot_rolling_causal_with_regimes(
        gen, network, top_n=min(10, max(1, len(stable_links))),
        alpha=ll_cfg.significance_level,
        save_path=os.path.join(output_dir, "rolling_causal_dynamics.png"),
    )

    # ----------------------------------------------------------------
    # Step 7: Use Case 5 — Parameter Sensitivity
    # ----------------------------------------------------------------
    print("\n[7/7] Use Case 5: Parameter Sensitivity...")
    sensitivity_df = use_case_5_parameter_sensitivity(
        causal_cfg, output_dir, fast=fast,
    )

    # ----------------------------------------------------------------
    # FINAL SUMMARY
    # ----------------------------------------------------------------
    elapsed = time.time() - t0

    print("\n" + "#" * 80)
    print("  ANALYSIS COMPLETE — KEY FINDINGS")
    print("#" * 80)

    # Detection summary
    if results_by_regime:
        primary = causal_cfg.methods[0]
        print(f"\n  1. SECTOR ROTATION (method: {primary}):")
        for regime, results in results_by_regime.items():
            result = results.get(primary)
            if result:
                scores = result.get_leader_scores(0.05)
                top = sorted(scores.items(), key=lambda x: x[1]["count"], reverse=True)
                leaders = [f"{n}({info['count']})" for n, info in top if info["count"] > 0]
                print(f"     {regime:>15s}: leaders = {', '.join(leaders[:4]) if leaders else 'none'}")

    # Portfolio summary
    if backtest_results:
        print(f"\n  2. PORTFOLIO PERFORMANCE (OOS):")
        best = max(backtest_results, key=lambda r: r.metrics.get("sharpe", -999))
        for res in backtest_results:
            marker = " ★" if res.name == best.name else ""
            if res.metrics:
                print(f"     {res.name:<35s} Sharpe={res.metrics.get('sharpe', 0):+.3f} "
                      f"CAGR={res.metrics.get('cagr', 0):+.2%}{marker}")

    # Sensitivity summary
    if sensitivity_df is not None and len(sensitivity_df) > 0:
        print(f"\n  3. SENSITIVITY (Granger detection):")
        print(f"     Best F1: {sensitivity_df['f1'].max():.0%} "
              f"(strength={sensitivity_df.loc[sensitivity_df['f1'].idxmax(), 'lead_lag_strength']:.1f}, "
              f"noise={sensitivity_df.loc[sensitivity_df['f1'].idxmax(), 'noise_scale']:.1f})")
        print(f"     Worst F1: {sensitivity_df['f1'].min():.0%} "
              f"(strength={sensitivity_df.loc[sensitivity_df['f1'].idxmin(), 'lead_lag_strength']:.1f}, "
              f"noise={sensitivity_df.loc[sensitivity_df['f1'].idxmin(), 'noise_scale']:.1f})")

    print(f"\n  4. GENERATED FEATURES: {features.shape[1]} total")
    print(f"     Regime indicators: vol_regime, trend_regime, correlation_regime, "
          f"dispersion_regime, momentum_regime")
    print(f"     Macro features: {', '.join(f'macro_{m}' for m in MACRO_NAMES)}")
    print(f"     Per-sector: momentum (21/63/126d), volatility (21/63d), "
          f"relative strength, rolling beta")
    print(f"     Cross-sector: avg_lagged_correlation, rotation_speed")

    # Save summary
    summary_data = {
        "ground_truth": ground_truth_df,
        "regime_stats": regime_stats,
        "features": features,
    }
    if sensitivity_df is not None:
        summary_data["sensitivity"] = sensitivity_df
    if backtest_results:
        bt_rows = []
        for res in backtest_results:
            if res.metrics:
                bt_rows.append({"strategy": res.name, **res.metrics})
        if bt_rows:
            summary_data["portfolio"] = pd.DataFrame(bt_rows)

    for name, df in summary_data.items():
        path = os.path.join(output_dir, f"summary_{name}.csv")
        df.to_csv(path, index=True)

    print(f"\n  Results saved to {output_dir}/")
    print(f"  Total time: {elapsed:.1f}s")
    print("#" * 80)

    return gen, results_by_regime, backtest_results


def main():
    parser = argparse.ArgumentParser(
        description="Sector-based Causal Analysis with Regime Features"
    )
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: fewer methods, smaller data")
    parser.add_argument("--output-dir", default="output_sectors",
                        help="Output directory")
    args = parser.parse_args()

    run_sector_pipeline(fast=args.fast, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
