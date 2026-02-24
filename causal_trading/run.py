"""
Main entry point: run the full causal trading pipeline.

Usage:
    python -m causal_trading.run [--fast] [--no-rolling] [--output-dir DIR]
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from .config import Config
from .data import load_data, compute_returns, get_ground_truth_pairs
from .causal import run_causal_analysis
from .lead_lag import rolling_lead_lag_detection
from .pairs import select_pairs, run_pairs_strategy
from .portfolio import optimize_portfolio
from .backtest import (
    backtest_portfolio,
    backtest_pairs,
    backtest_equal_weight_benchmark,
    BacktestResult,
)
from .visualize import (
    plot_causal_matrix,
    plot_lead_lag_dynamics,
    plot_regime_changes,
    plot_pair_trading,
    plot_backtest_results,
    plot_leader_network,
)


def run_pipeline(cfg: Config, output_dir: str = "output", fast: bool = False, skip_rolling: bool = False):
    """Run the complete pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    # ---- 1. Load data ----
    print("\n[1/6] Loading data...")
    prices = load_data(cfg.data)
    returns = compute_returns(prices, method="log")
    print(f"  Prices shape: {prices.shape} ({prices.index[0].date()} to {prices.index[-1].date()})")
    print(f"  Returns shape: {returns.shape}")

    # Ground truth for validation (synthetic only)
    gt_pairs = get_ground_truth_pairs(cfg.data)
    if gt_pairs:
        print(f"  Ground truth pairs: {gt_pairs}")

    # ---- 2. Static causal analysis (full sample) ----
    print("\n[2/6] Running causal inference (full sample)...")
    if fast:
        cfg.causal.methods = ["granger", "ccf"]
        cfg.causal.max_lag = 5
    causal_results = run_causal_analysis(returns, cfg.causal)

    for method_name, result in causal_results.items():
        links = result.get_significant_links(cfg.causal.significance_level)
        print(f"\n  {method_name}: {len(links)} significant links found")
        for cause, effect, lag, strength, pval in links[:10]:
            print(f"    {cause} -> {effect}  (lag={lag}, p={pval:.4f}, str={strength:.4f})")

        # Validate against ground truth
        if gt_pairs:
            print(f"\n  Validation against ground truth ({method_name}):")
            detected_pairs = {(c, e) for c, e, _, _, _ in links}
            for leader, follower, true_lag in gt_pairs:
                found = (leader, follower) in detected_pairs
                detected_lag = None
                for c, e, l, s, p in links:
                    if c == leader and e == follower:
                        detected_lag = l
                        break
                status = "FOUND" if found else "MISSED"
                lag_info = f", detected lag={detected_lag}" if detected_lag else ""
                print(f"    {status}: {leader} -> {follower} (true lag={true_lag}{lag_info})")

        # Plots
        plot_causal_matrix(result, cfg.causal.significance_level,
                           save_path=os.path.join(output_dir, f"causal_matrix_{method_name}.png"))
        plot_leader_network(result, cfg.causal.significance_level,
                            save_path=os.path.join(output_dir, f"causal_network_{method_name}.png"))

    # Leader ranking
    print("\n  Leader rankings:")
    for method_name, result in causal_results.items():
        scores = result.get_leader_scores(cfg.causal.significance_level)
        ranked = sorted(scores.items(), key=lambda x: x[1]["count"], reverse=True)
        print(f"\n  {method_name}:")
        for name, info in ranked[:5]:
            print(f"    {name}: leads {info['count']} assets (strength={info['strength']:.4f})")

    # ---- 3. Rolling lead-lag detection ----
    print("\n[3/6] Rolling lead-lag detection...")
    if skip_rolling:
        print("  Skipped (--no-rolling)")
        network = None
    else:
        if fast:
            cfg.lead_lag.rolling_window = 126
            cfg.lead_lag.rolling_step = 63
            cfg.lead_lag.method = "granger"
        network = rolling_lead_lag_detection(returns, cfg.lead_lag, cfg.causal)

        # Report on dynamics
        stable_links = network.get_stable_links(cfg.lead_lag.significance_level)
        print(f"\n  Stable links found: {len(stable_links)}")
        if not stable_links.empty:
            print(stable_links.to_string(index=False))

        regime_events = network.get_regime_changes(cfg.lead_lag.significance_level)
        print(f"\n  Regime change events: {len(regime_events)}")
        if regime_events:
            events_df = pd.DataFrame(regime_events)
            appeared = events_df[events_df["event"] == "appeared"]
            disappeared = events_df[events_df["event"] == "disappeared"]
            print(f"    Links appeared:    {len(appeared)}")
            print(f"    Links disappeared: {len(disappeared)}")

        leaders = network.get_leaders(cfg.lead_lag.significance_level)
        print(f"\n  Current leaders (rolling):")
        print(leaders.head(5).to_string())

        # Plots
        plot_lead_lag_dynamics(network, top_n=min(8, len(stable_links)),
                               alpha=cfg.lead_lag.significance_level,
                               save_path=os.path.join(output_dir, "lead_lag_dynamics.png"))
        plot_regime_changes(network, alpha=cfg.lead_lag.significance_level,
                            save_path=os.path.join(output_dir, "regime_changes.png"))

    # ---- 4. Pair selection ----
    print("\n[4/6] Selecting causal pairs for trading...")
    if network is not None:
        pairs = select_pairs(network, cfg.pairs, cfg.lead_lag.significance_level)
    else:
        # Use static analysis to pick pairs
        first_result = next(iter(causal_results.values()))
        links = first_result.get_significant_links(cfg.causal.significance_level)
        from .pairs import CausalPair
        pairs = []
        for cause, effect, lag, strength, pval in links[:cfg.pairs.max_pairs]:
            pairs.append(CausalPair(cause, effect, lag, strength))

    if pairs:
        print(f"  Selected {len(pairs)} pairs:")
        for p in pairs:
            print(f"    {p}")
    else:
        print("  No pairs selected — causal signals too weak")

    # ---- 5. Backtesting ----
    print("\n[5/6] Backtesting...")

    backtest_results = []

    # a) Equal weight benchmark
    bm_result = backtest_equal_weight_benchmark(returns, cfg.backtest)
    backtest_results.append(bm_result)
    print(bm_result.summary())

    # b) Portfolio strategies
    for method in ["equal_weight", "inverse_vol", "min_variance", "causal_weighted"]:
        port_cfg = cfg.portfolio
        port_cfg.method = method
        port_result = backtest_portfolio(prices, returns, cfg.backtest, port_cfg, cfg.causal)
        backtest_results.append(port_result)
        print(port_result.summary())

    # c) Pair trading
    if pairs:
        pairs_result = backtest_pairs(prices, pairs, cfg.backtest, cfg.pairs)
        backtest_results.append(pairs_result)
        print(pairs_result.summary())

        # Plot individual pairs
        T = len(prices)
        train_end = int(T * cfg.backtest.train_fraction)
        oos_prices = prices.iloc[train_end:]
        pair_signals = run_pairs_strategy(oos_prices, pairs, cfg.pairs)
        for pair_name, signals in pair_signals.items():
            pair_obj = next(p for p in pairs if f"{p.leader}->{p.follower}" == pair_name)
            plot_pair_trading(pair_obj, signals, oos_prices,
                              save_path=os.path.join(output_dir, f"pair_{pair_name.replace('->', '_')}.png"))

    # ---- 6. Summary & comparison ----
    print("\n[6/6] Summary...")
    plot_backtest_results(backtest_results,
                          save_path=os.path.join(output_dir, "strategy_comparison.png"))

    # Summary table
    summary_rows = []
    for res in backtest_results:
        summary_rows.append({"strategy": res.name, **res.metrics})
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        print("\n" + "=" * 80)
        print("  STRATEGY COMPARISON")
        print("=" * 80)
        cols_to_fmt = ["total_return", "cagr", "annual_vol", "max_drawdown"]
        for c in cols_to_fmt:
            if c in summary_df.columns:
                summary_df[c] = summary_df[c].apply(lambda x: f"{x:+.2%}")
        print(summary_df.to_string(index=False))
        print("=" * 80)

        # Save
        summary_df.to_csv(os.path.join(output_dir, "summary.csv"), index=False)
        print(f"\n  Results saved to {output_dir}/")

    elapsed = time.time() - t0
    print(f"\n  Pipeline completed in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Causal Portfolio & Lead-Lag Detection")
    parser.add_argument("--fast", action="store_true", help="Fast mode: fewer methods, smaller windows")
    parser.add_argument("--no-rolling", action="store_true", help="Skip rolling lead-lag detection")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--n-assets", type=int, default=10, help="Number of synthetic assets")
    parser.add_argument("--n-obs", type=int, default=2000, help="Number of observations")
    args = parser.parse_args()

    cfg = Config()
    cfg.data.source = "synthetic"
    cfg.data.synthetic_n_assets = args.n_assets
    cfg.data.synthetic_n_obs = args.n_obs

    run_pipeline(cfg, output_dir=args.output_dir, fast=args.fast, skip_rolling=args.no_rolling)


if __name__ == "__main__":
    main()
