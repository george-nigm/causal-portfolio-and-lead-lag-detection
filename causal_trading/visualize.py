"""
Visualization for causal analysis results.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import Dict, List, Optional
from .causal import CausalResult
from .lead_lag import LeadLagNetwork
from .backtest import BacktestResult
from .pairs import CausalPair


def plot_causal_matrix(
    result: CausalResult,
    alpha: float = 0.05,
    save_path: Optional[str] = None,
):
    """Plot causal p-value matrix as heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # P-value matrix
    p_df = pd.DataFrame(
        result.p_matrix,
        index=result.assets,
        columns=result.assets,
    )
    # Mask diagonal
    mask = np.eye(len(result.assets), dtype=bool)

    sns.heatmap(
        p_df, mask=mask, ax=axes[0],
        cmap="RdYlGn_r", vmin=0, vmax=0.1,
        annot=True, fmt=".3f", linewidths=0.5,
    )
    axes[0].set_title(f"{result.method} — p-values (row=effect, col=cause)")

    # Significance matrix (binary)
    sig = (p_df < alpha).astype(int)
    sns.heatmap(
        sig, mask=mask, ax=axes[1],
        cmap="Greens", vmin=0, vmax=1,
        annot=True, fmt="d", linewidths=0.5,
    )
    axes[1].set_title(f"Significant links (p < {alpha})")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_lead_lag_dynamics(
    network: LeadLagNetwork,
    top_n: int = 10,
    alpha: float = 0.05,
    save_path: Optional[str] = None,
):
    """Plot how the top lead-lag relationships evolve over time."""
    # Find most stable links
    stable = network.get_stable_links(alpha=alpha, min_stability=0.1)
    if stable.empty:
        print("  No links to plot")
        return

    stable = stable.head(top_n)

    fig, axes = plt.subplots(top_n, 1, figsize=(14, 3 * top_n), sharex=True)
    if top_n == 1:
        axes = [axes]

    for idx, (_, row) in enumerate(stable.iterrows()):
        link = network.links[(row["cause"], row["effect"])]
        ts = link.to_series()
        if ts.empty:
            continue

        ax = axes[idx]
        # Plot p-value over time
        ax.fill_between(
            ts.index, 0, 1,
            where=ts["p_value"] < alpha,
            alpha=0.2, color="green", label="significant"
        )
        ax.plot(ts.index, ts["p_value"], "k-", linewidth=0.8)
        ax.axhline(alpha, color="red", linestyle="--", linewidth=0.5)
        ax.set_ylabel("p-value")
        ax.set_title(
            f"{row['cause']} → {row['effect']} "
            f"(stability={row['stability']:.0%})",
            fontsize=10,
        )
        ax.set_ylim(0, 0.2)
        ax.legend(loc="upper right", fontsize=8)

    plt.xlabel("Date")
    plt.suptitle("Lead-Lag Relationship Dynamics", fontsize=14, y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_regime_changes(
    network: LeadLagNetwork,
    alpha: float = 0.05,
    save_path: Optional[str] = None,
):
    """Plot timeline of regime changes (links appearing/disappearing)."""
    events = network.get_regime_changes(alpha=alpha)
    if not events:
        print("  No regime changes to plot")
        return

    events_df = pd.DataFrame(events)
    events_df["pair"] = events_df["cause"] + " → " + events_df["effect"]

    fig, ax = plt.subplots(figsize=(14, 6))
    pairs_list = events_df["pair"].unique()
    pair_to_y = {p: i for i, p in enumerate(pairs_list)}

    for _, ev in events_df.iterrows():
        y = pair_to_y[ev["pair"]]
        color = "green" if ev["event"] == "appeared" else "red"
        marker = "^" if ev["event"] == "appeared" else "v"
        ax.scatter(ev["date"], y, color=color, marker=marker, s=30, alpha=0.7)

    ax.set_yticks(range(len(pairs_list)))
    ax.set_yticklabels(pairs_list, fontsize=8)
    ax.set_xlabel("Date")
    ax.set_title("Causal Link Regime Changes (green=appeared, red=disappeared)")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_pair_trading(
    pair: CausalPair,
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """Plot pair trading signals and performance."""
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1])

    # Panel 1: Prices
    ax1 = fig.add_subplot(gs[0])
    p_leader = prices[pair.leader] / prices[pair.leader].iloc[0] * 100
    p_follower = prices[pair.follower] / prices[pair.follower].iloc[0] * 100
    ax1.plot(p_leader, label=f"{pair.leader} (leader)", linewidth=1)
    ax1.plot(p_follower, label=f"{pair.follower} (follower)", linewidth=1)
    ax1.legend()
    ax1.set_title(f"Pair: {pair.leader} → {pair.follower} (lag={pair.lag})")
    ax1.set_ylabel("Normalized Price")

    # Panel 2: Z-score and signals
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    z = signals["z_score"].dropna()
    ax2.plot(z, "b-", linewidth=0.8, label="Z-score")
    ax2.axhline(2.0, color="red", linestyle="--", linewidth=0.5)
    ax2.axhline(-2.0, color="red", linestyle="--", linewidth=0.5)
    ax2.axhline(0.5, color="gray", linestyle=":", linewidth=0.5)
    ax2.axhline(-0.5, color="gray", linestyle=":", linewidth=0.5)
    ax2.axhline(0, color="black", linewidth=0.3)
    # Mark entry/exit signals
    buy = signals[signals["signal"] > 0]
    sell = signals[signals["signal"] < 0]
    ax2.scatter(buy.index, buy["z_score"], color="green", marker="^", s=30, zorder=5)
    ax2.scatter(sell.index, sell["z_score"], color="red", marker="v", s=30, zorder=5)
    ax2.set_ylabel("Z-score")
    ax2.legend(loc="upper right")

    # Panel 3: Position
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.fill_between(signals.index, signals["position"], 0, alpha=0.3)
    ax3.set_ylabel("Position")
    ax3.set_xlabel("Date")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_backtest_results(
    results: List[BacktestResult],
    save_path: Optional[str] = None,
):
    """Plot portfolio values and comparison of strategies."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

    # Panel 1: Portfolio values
    for res in results:
        if res.portfolio_value is not None:
            axes[0].plot(res.portfolio_value, label=res.name, linewidth=1.2)
    axes[0].legend(fontsize=9)
    axes[0].set_ylabel("Portfolio Value ($)")
    axes[0].set_title("Strategy Comparison")
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Drawdowns
    for res in results:
        if res.returns is not None:
            cum = (1 + res.returns).cumprod()
            dd = (cum - cum.cummax()) / cum.cummax()
            axes[1].fill_between(dd.index, dd, 0, alpha=0.3, label=res.name)
    axes[1].legend(fontsize=9)
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_leader_network(
    result: CausalResult,
    alpha: float = 0.05,
    save_path: Optional[str] = None,
):
    """Plot the causal network as a directed graph (simple version)."""
    links = result.get_significant_links(alpha=alpha)
    if not links:
        print("  No significant links to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 10))
    n = len(result.assets)

    # Arrange nodes in a circle
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)

    # Draw nodes
    scores = result.get_leader_scores(alpha=alpha)
    max_count = max((s["count"] for s in scores.values()), default=1)
    for i, asset in enumerate(result.assets):
        count = scores[asset]["count"]
        size = 300 + 500 * (count / (max_count + 1))
        color = plt.cm.YlOrRd(count / (max_count + 1))
        ax.scatter(x[i], y[i], s=size, color=color, zorder=3, edgecolors="black")
        ax.annotate(asset, (x[i], y[i]), ha="center", va="center", fontsize=7, fontweight="bold")

    # Draw edges
    asset_to_idx = {a: i for i, a in enumerate(result.assets)}
    for cause, effect, lag, strength, pval in links:
        ci = asset_to_idx[cause]
        ei = asset_to_idx[effect]
        dx = x[ei] - x[ci]
        dy = y[ei] - y[ci]
        ax.annotate(
            "", xy=(x[ei], y[ei]), xytext=(x[ci], y[ci]),
            arrowprops=dict(
                arrowstyle="->",
                color="steelblue",
                alpha=min(1.0, 0.3 + 0.7 * (1 - pval / alpha)),
                lw=1.5,
            ),
        )

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.set_title(f"Causal Network ({result.method}, {len(links)} links, alpha={alpha})")
    ax.axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)
