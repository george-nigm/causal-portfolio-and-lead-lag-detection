"""
Run causal analysis on S&P 500 small-cap stocks.

This is the main script for answering:
1. Can we find real lead-lag relationships?
2. Are they stable or transient?
3. How should we weight/trade them?

Usage:
    python -m causal_trading.run_sp500 [--fast] [--n-tickers N]
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

from .config import Config
from .data import compute_returns
from .sp500_synthetic import generate_sp500_small_synthetic, SECTORS, EMBEDDED_LEAD_LAG
from .causal import run_causal_analysis
from .lead_lag import rolling_lead_lag_detection
from .pairs import select_pairs, run_pairs_strategy
from .backtest import (
    backtest_portfolio,
    backtest_pairs,
    backtest_equal_weight_benchmark,
)
from .visualize import (
    plot_causal_matrix,
    plot_lead_lag_dynamics,
    plot_regime_changes,
    plot_pair_trading,
    plot_backtest_results,
    plot_leader_network,
)


def run_sp500_analysis(fast: bool = False, n_tickers: int = 35):
    """Run the full causal analysis on S&P 500 small-cap stocks."""
    output_dir = "output_sp500"
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    # ================================================================
    # CONFIG
    # ================================================================
    cfg = Config()
    if fast:
        cfg.causal.methods = ["granger", "ccf"]
        cfg.causal.max_lag = 5
        cfg.lead_lag.rolling_window = 126
        cfg.lead_lag.rolling_step = 63
    else:
        cfg.causal.methods = ["granger", "ccf", "transfer_entropy"]
        cfg.causal.max_lag = 10
        cfg.lead_lag.rolling_window = 252
        cfg.lead_lag.rolling_step = 22

    cfg.lead_lag.method = "granger"
    cfg.pairs.max_pairs = 8
    cfg.pairs.entry_z = 2.0
    cfg.pairs.exit_z = 0.5
    cfg.pairs.lookback = 60
    cfg.backtest.train_fraction = 0.5
    cfg.backtest.commission_pct = 0.001
    cfg.portfolio.rebalance_days = 22

    # ================================================================
    # 1. DATA
    # ================================================================
    print("\n" + "=" * 70)
    print("  CAUSAL LEAD-LAG ANALYSIS ON S&P 500 SMALL-CAP STOCKS")
    print("=" * 70)

    print("\n[1/7] Generating S&P 500 small-cap data...")
    prices, ground_truth = generate_sp500_small_synthetic()

    # Subset to requested number of tickers
    if n_tickers < prices.shape[1]:
        prices = prices.iloc[:, :n_tickers]

    returns = compute_returns(prices, method="log")
    tickers = list(prices.columns)

    print(f"  {prices.shape[1]} stocks, {prices.shape[0]} trading days")
    print(f"  Period: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"\n  Sectors represented:")
    for sector, sec_tickers in SECTORS.items():
        present = [t for t in sec_tickers if t in tickers]
        if present:
            print(f"    {sector}: {present}")

    print(f"\n  Ground truth lead-lag relationships:")
    gt_in_sample = []
    for g in ground_truth:
        if g["leader"] in tickers and g["follower"] in tickers:
            gt_in_sample.append(g)
            ptype = "persistent" if g["persistent"] else f"transient"
            print(f"    {g['leader']} -> {g['follower']} (lag={g['lag']}, "
                  f"strength={g['strength']:.2f}, {ptype})")

    # ================================================================
    # 2. STATIC CAUSAL ANALYSIS
    # ================================================================
    print("\n[2/7] Static causal analysis (full sample)...")
    causal_results = run_causal_analysis(returns, cfg.causal)

    for method_name, result in causal_results.items():
        links = result.get_significant_links(cfg.causal.significance_level)
        print(f"\n  {method_name}: {len(links)} significant links")

        # Top 15 links
        for cause, effect, lag, strength, pval in links[:15]:
            # Check if this is a ground truth link
            is_gt = any(g["leader"] == cause and g["follower"] == effect for g in gt_in_sample)
            marker = " ** GT **" if is_gt else ""
            print(f"    {cause:>6} -> {effect:<6}  lag={lag}  p={pval:.4f}  str={strength:.4f}{marker}")

        # Validation
        print(f"\n  Ground truth validation ({method_name}):")
        detected = 0
        for g in gt_in_sample:
            found = False
            for cause, effect, lag, strength, pval in links:
                if cause == g["leader"] and effect == g["follower"]:
                    found = True
                    lag_match = "EXACT" if lag == g["lag"] else f"lag={lag} vs true={g['lag']}"
                    print(f"    FOUND:  {g['leader']} -> {g['follower']} ({lag_match}, p={pval:.4f})")
                    detected += 1
                    break
            if not found:
                print(f"    MISSED: {g['leader']} -> {g['follower']} (true lag={g['lag']})")
        print(f"    Detection rate: {detected}/{len(gt_in_sample)} = {detected/len(gt_in_sample):.0%}")

        # Plots
        plot_causal_matrix(result, cfg.causal.significance_level,
                           save_path=os.path.join(output_dir, f"causal_matrix_{method_name}.png"))
        plot_leader_network(result, cfg.causal.significance_level,
                            save_path=os.path.join(output_dir, f"network_{method_name}.png"))

    # ================================================================
    # 3. LEADER RANKINGS
    # ================================================================
    print("\n[3/7] Leader rankings (who leads whom?)...")
    for method_name, result in causal_results.items():
        scores = result.get_leader_scores(cfg.causal.significance_level)
        ranked = sorted(scores.items(), key=lambda x: x[1]["count"], reverse=True)
        print(f"\n  {method_name} — top leaders:")
        for name, info in ranked[:8]:
            sector = "?"
            for s, ts in SECTORS.items():
                if name in ts:
                    sector = s
                    break
            print(f"    {name:>6} ({sector:12s}): leads {info['count']} stocks, "
                  f"total strength={info['strength']:.3f}")

    # ================================================================
    # 4. ROLLING LEAD-LAG DETECTION
    # ================================================================
    print("\n[4/7] Rolling lead-lag detection (tracking dynamics)...")
    network = rolling_lead_lag_detection(returns, cfg.lead_lag, cfg.causal)

    # Stable links
    stable = network.get_stable_links(cfg.lead_lag.significance_level, min_stability=0.3)
    print(f"\n  Stable links (>30% of windows active): {len(stable)}")
    if not stable.empty:
        for _, row in stable.head(15).iterrows():
            is_gt = any(g["leader"] == row["cause"] and g["follower"] == row["effect"]
                        for g in gt_in_sample)
            marker = " ** GT **" if is_gt else ""
            print(f"    {row['cause']:>6} -> {row['effect']:<6}  "
                  f"stability={row['stability']:.0%}  "
                  f"mean_lag={row['mean_lag']:.1f}  "
                  f"active={'YES' if row['currently_active'] else 'no'}{marker}")

    # Regime changes
    events = network.get_regime_changes(cfg.lead_lag.significance_level)
    appeared = [e for e in events if e["event"] == "appeared"]
    disappeared = [e for e in events if e["event"] == "disappeared"]
    print(f"\n  Regime changes: {len(appeared)} appeared, {len(disappeared)} disappeared")

    # Current leaders
    leaders = network.get_leaders(cfg.lead_lag.significance_level)
    print(f"\n  Current leaders (end of sample):")
    for name, row in leaders.head(8).iterrows():
        sector = "?"
        for s, ts in SECTORS.items():
            if name in ts:
                sector = s
                break
        print(f"    {name:>6} ({sector:12s}): leads {int(row['n_followers'])} stocks")

    # Plots
    n_plot = min(10, len(stable))
    if n_plot > 0:
        plot_lead_lag_dynamics(network, top_n=n_plot,
                               alpha=cfg.lead_lag.significance_level,
                               save_path=os.path.join(output_dir, "lead_lag_dynamics.png"))
    plot_regime_changes(network, alpha=cfg.lead_lag.significance_level,
                        save_path=os.path.join(output_dir, "regime_changes.png"))

    # ================================================================
    # 5. PAIR SELECTION
    # ================================================================
    print("\n[5/7] Selecting causal pairs for trading...")
    pairs = select_pairs(network, cfg.pairs, cfg.lead_lag.significance_level, min_stability=0.3)

    if pairs:
        print(f"  Selected {len(pairs)} pairs:")
        for p in pairs:
            is_gt = any(g["leader"] == p.leader and g["follower"] == p.follower for g in gt_in_sample)
            marker = " ** GT **" if is_gt else ""
            print(f"    {p}{marker}")
    else:
        print("  No pairs selected with stability>30%, trying lower threshold...")
        pairs = select_pairs(network, cfg.pairs, 0.10, min_stability=0.15)
        for p in pairs:
            print(f"    {p}")

    # ================================================================
    # 6. BACKTESTING
    # ================================================================
    print("\n[6/7] Backtesting strategies...")
    bt_results = []

    # Benchmark
    bm = backtest_equal_weight_benchmark(returns, cfg.backtest)
    bt_results.append(bm)
    print(bm.summary())

    # Portfolio strategies
    for method in ["min_variance", "inverse_vol", "causal_weighted"]:
        pcfg = cfg.portfolio
        pcfg.method = method
        res = backtest_portfolio(prices, returns, cfg.backtest, pcfg, cfg.causal)
        bt_results.append(res)
        print(res.summary())

    # Pair trading
    if pairs:
        pairs_res = backtest_pairs(prices, pairs, cfg.backtest, cfg.pairs)
        bt_results.append(pairs_res)
        print(pairs_res.summary())

        # Plot individual pairs
        T = len(prices)
        train_end = int(T * cfg.backtest.train_fraction)
        oos_prices = prices.iloc[train_end:]
        pair_signals = run_pairs_strategy(oos_prices, pairs, cfg.pairs)
        for pair_name, signals in pair_signals.items():
            pair_obj = next(p for p in pairs if f"{p.leader}->{p.follower}" == pair_name)
            fname = f"pair_{pair_name.replace('->', '_')}.png"
            plot_pair_trading(pair_obj, signals, oos_prices,
                              save_path=os.path.join(output_dir, fname))

    # ================================================================
    # 7. SUMMARY
    # ================================================================
    print("\n[7/7] Final summary...")
    plot_backtest_results(bt_results, save_path=os.path.join(output_dir, "strategy_comparison.png"))

    summary_rows = []
    for res in bt_results:
        summary_rows.append({"strategy": res.name, **res.metrics})
    summary_df = pd.DataFrame(summary_rows)

    if not summary_df.empty:
        print("\n" + "=" * 80)
        print("  STRATEGY COMPARISON")
        print("=" * 80)
        display_df = summary_df.copy()
        for c in ["total_return", "cagr", "annual_vol", "max_drawdown"]:
            if c in display_df.columns:
                display_df[c] = display_df[c].apply(lambda x: f"{x:+.2%}")
        for c in ["sharpe", "calmar", "sortino"]:
            if c in display_df.columns:
                display_df[c] = display_df[c].apply(lambda x: f"{x:+.3f}")
        print(display_df.to_string(index=False))
        print("=" * 80)

        summary_df.to_csv(os.path.join(output_dir, "summary.csv"), index=False)

    # Key insights
    print("\n" + "-" * 70)
    print("  KEY FINDINGS")
    print("-" * 70)
    print(f"  - Causal methods detected {len(gt_in_sample)} ground truth relationships")
    for method_name, result in causal_results.items():
        links = result.get_significant_links(cfg.causal.significance_level)
        detected = sum(1 for g in gt_in_sample
                       if any(c == g["leader"] and e == g["follower"] for c, e, *_ in links))
        print(f"    {method_name}: {detected}/{len(gt_in_sample)} detected")
    if not stable.empty:
        gt_stable = sum(1 for _, r in stable.iterrows()
                        if any(g["leader"] == r["cause"] and g["follower"] == r["effect"]
                               for g in gt_in_sample))
        print(f"  - Rolling detection found {len(stable)} stable links, {gt_stable} are ground truth")
    print(f"  - Best Sharpe: {max((r.metrics.get('sharpe', 0) for r in bt_results)):.3f}")
    print(f"  - Results saved to {output_dir}/")
    print("-" * 70)

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="S&P 500 Causal Analysis")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--n-tickers", type=int, default=35)
    args = parser.parse_args()
    run_sp500_analysis(fast=args.fast, n_tickers=args.n_tickers)


if __name__ == "__main__":
    main()
