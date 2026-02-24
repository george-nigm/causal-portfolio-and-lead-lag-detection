"""
Backtesting engine for both portfolio strategies and pair trading.

Simple, transparent vectorized backtest — no external dependencies.
Computes returns, drawdowns, Sharpe, and other standard metrics.

Design principles:
- Walk-forward only: weights computed from past data, applied to future
- No look-ahead bias: signals at t use data up to t-1
- Transaction costs: proportional to turnover on rebalance days
- Weight drift: between rebalances, weights drift with returns (realistic)
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from .config import BacktestConfig, PortfolioConfig, PairsConfig, CausalConfig
from .causal import run_causal_analysis, CausalResult
from .portfolio import optimize_portfolio
from .pairs import CausalPair, generate_pair_signals


class BacktestResult:
    """Container for backtest results."""

    def __init__(self, name: str):
        self.name = name
        self.portfolio_value: Optional[pd.Series] = None
        self.returns: Optional[pd.Series] = None
        self.weights_history: Optional[pd.DataFrame] = None
        self.metrics: Dict[str, float] = {}
        self.pair_signals: Optional[Dict[str, pd.DataFrame]] = None

    def compute_metrics(self, risk_free_rate: float = 0.0):
        """Compute standard performance metrics."""
        if self.returns is None or len(self.returns) == 0:
            return

        ret = self.returns.dropna()
        total_days = len(ret)
        years = total_days / 252.0

        # Total return
        cum_ret = (1 + ret).prod() - 1
        self.metrics["total_return"] = cum_ret

        # CAGR
        if years > 0 and (1 + cum_ret) > 0:
            self.metrics["cagr"] = (1 + cum_ret) ** (1.0 / years) - 1
        else:
            self.metrics["cagr"] = 0.0

        # Annualized volatility
        ann_vol = ret.std() * np.sqrt(252)
        self.metrics["annual_vol"] = ann_vol

        # Sharpe ratio
        excess = ret.mean() - risk_free_rate / 252.0
        if ret.std() > 0:
            self.metrics["sharpe"] = np.sqrt(252) * excess / ret.std()
        else:
            self.metrics["sharpe"] = 0.0

        # Max drawdown
        cum_returns = (1 + ret).cumprod()
        peak = cum_returns.cummax()
        drawdown = (cum_returns - peak) / peak
        self.metrics["max_drawdown"] = drawdown.min()

        # Calmar ratio
        if self.metrics["max_drawdown"] != 0:
            self.metrics["calmar"] = self.metrics["cagr"] / abs(self.metrics["max_drawdown"])
        else:
            self.metrics["calmar"] = 0.0

        # Win rate
        self.metrics["win_rate"] = (ret > 0).mean()

        # Sortino ratio
        downside = ret[ret < 0]
        if len(downside) > 0:
            downside_std = downside.std() * np.sqrt(252)
            self.metrics["sortino"] = (self.metrics["cagr"] - risk_free_rate) / (downside_std + 1e-12)
        else:
            self.metrics["sortino"] = 0.0

        # Number of trades (for pairs)
        self.metrics["n_days"] = total_days

    def summary(self) -> str:
        """Formatted summary string."""
        lines = [f"\n{'=' * 60}", f"  {self.name}", f"{'=' * 60}"]
        for k, v in self.metrics.items():
            if "return" in k or "cagr" in k or "vol" in k or "drawdown" in k:
                lines.append(f"  {k:20s}: {v:+.2%}")
            else:
                lines.append(f"  {k:20s}: {v:+.4f}")
        lines.append(f"{'=' * 60}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Portfolio backtest
# ---------------------------------------------------------------------------

def backtest_portfolio(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    bt_cfg: BacktestConfig,
    port_cfg: PortfolioConfig,
    causal_cfg: CausalConfig,
) -> BacktestResult:
    """
    Backtest a portfolio strategy with periodic rebalancing.

    Walk-forward design:
    1. At each rebalance date, use ONLY historical data to estimate causal structure
    2. Optimize portfolio weights from that historical window
    3. Apply weights to NEXT period's returns (no look-ahead)
    4. Account for transaction costs proportional to turnover
    5. Between rebalances, weights drift naturally with asset returns
    """
    result = BacktestResult(f"Portfolio ({port_cfg.method})")

    T = len(returns)
    train_end = int(T * bt_cfg.train_fraction)
    rebal_interval = port_cfg.rebalance_days
    lookback = min(252, train_end)

    oos_returns = returns.iloc[train_end:]

    if len(oos_returns) < 10:
        print("  Not enough out-of-sample data for portfolio backtest")
        return result

    rebal_indices = list(range(0, len(oos_returns), rebal_interval))
    if not rebal_indices:
        rebal_indices = [0]

    weights_history = []
    current_weights = None
    port_returns = pd.Series(0.0, index=oos_returns.index)

    # Only run causal analysis for methods that need it
    needs_causal = port_cfg.method == "causal_weighted"

    for idx in rebal_indices:
        est_end = train_end + idx
        est_start = max(0, est_end - lookback)
        est_returns = returns.iloc[est_start:est_end]

        if len(est_returns) < 30:
            continue

        # Only run expensive causal analysis when the method requires it
        causal_result = None
        if needs_causal:
            causal_results = run_causal_analysis(est_returns, causal_cfg)
            causal_result = next(iter(causal_results.values())) if causal_results else None

        new_weights = optimize_portfolio(est_returns, causal_result, port_cfg)
        new_weights = new_weights.clip(lower=0)
        new_weights = new_weights / (new_weights.sum() + 1e-12)

        # Transaction costs: proportional to turnover (both buying and selling)
        if current_weights is not None:
            # Align indices for turnover calc (handles potential mismatches)
            aligned_old = current_weights.reindex(new_weights.index, fill_value=0)
            turnover = (new_weights - aligned_old).abs().sum()
            tc = turnover * bt_cfg.commission_pct
        else:
            tc = bt_cfg.commission_pct  # initial investment cost

        current_weights = new_weights
        weights_history.append((oos_returns.index[idx], new_weights.copy()))

        # Apply weights to returns until next rebalance
        next_rebal = rebal_indices[rebal_indices.index(idx) + 1] if idx != rebal_indices[-1] else len(oos_returns)
        period_returns = oos_returns.iloc[idx:next_rebal]

        for t_offset in range(len(period_returns)):
            t_global = idx + t_offset
            # Daily portfolio return = sum(weight_i * return_i)
            daily_ret = (period_returns.iloc[t_offset] * current_weights).sum()
            if t_offset == 0:
                daily_ret -= tc  # deduct TC on rebalance day
            port_returns.iloc[t_global] = daily_ret

            # Drift weights to reflect price changes (buy-and-hold between rebalances)
            if t_offset < len(period_returns) - 1:
                drifted = current_weights * (1 + period_returns.iloc[t_offset])
                current_weights = drifted / (drifted.sum() + 1e-12)

    result.returns = port_returns
    result.portfolio_value = bt_cfg.initial_capital * (1 + port_returns).cumprod()
    if weights_history:
        wh_dates, wh_vals = zip(*weights_history)
        result.weights_history = pd.DataFrame(list(wh_vals), index=list(wh_dates))
    result.compute_metrics()

    return result


# ---------------------------------------------------------------------------
# Pairs backtest
# ---------------------------------------------------------------------------

def backtest_pairs(
    prices: pd.DataFrame,
    pairs: List[CausalPair],
    bt_cfg: BacktestConfig,
    pairs_cfg: PairsConfig,
) -> BacktestResult:
    """
    Backtest pair trading strategy.

    For each pair:
    - Compute spread and z-scores using rolling lookback (no future data)
    - Generate entry/exit signals based on z-score thresholds
    - P&L: position is set at close of signal day, applied to NEXT day's return
    - Beta for hedge ratio is lagged by 1 day to avoid look-ahead
    """
    result = BacktestResult("Pair Trading (causal)")

    T = len(prices)
    train_end = int(T * bt_cfg.train_fraction)
    oos_prices = prices.iloc[train_end:]

    if len(oos_prices) < 100:
        print("  Not enough OOS data for pairs backtest")
        return result

    pair_pnls = {}
    all_signals = {}

    for pair in pairs:
        if pair.leader not in oos_prices.columns or pair.follower not in oos_prices.columns:
            continue

        signals = generate_pair_signals(oos_prices, pair, pairs_cfg)
        all_signals[f"{pair.leader}->{pair.follower}"] = signals

        follower_ret = oos_prices[pair.follower].pct_change()
        leader_ret = oos_prices[pair.leader].pct_change()
        position = signals["position"]

        # Use LAGGED beta to avoid look-ahead: hedge ratio known at t-1
        beta = signals["beta"].ffill().shift(1)

        # position.shift(1): decision made at t-1 close, applied to t returns
        spread_ret = position.shift(1) * (follower_ret - beta * leader_ret)

        # Transaction costs: charged when position changes
        position_change = position.diff().abs().fillna(0)
        tc = position_change * bt_cfg.commission_pct * 2  # both legs

        spread_ret = spread_ret - tc
        pair_pnls[f"{pair.leader}->{pair.follower}"] = spread_ret

    if not pair_pnls:
        print("  No valid pairs for backtesting")
        return result

    pnl_df = pd.DataFrame(pair_pnls).fillna(0)
    combined_returns = pnl_df.mean(axis=1)

    result.returns = combined_returns
    result.portfolio_value = bt_cfg.initial_capital * (1 + combined_returns).cumprod()
    result.pair_signals = all_signals
    result.compute_metrics()

    return result


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def backtest_equal_weight_benchmark(
    returns: pd.DataFrame,
    bt_cfg: BacktestConfig,
) -> BacktestResult:
    """
    Equal-weight buy-and-hold benchmark.
    Includes initial transaction cost for fair comparison.
    """
    result = BacktestResult("Equal Weight (benchmark)")

    T = len(returns)
    train_end = int(T * bt_cfg.train_fraction)
    oos_returns = returns.iloc[train_end:]

    port_returns = oos_returns.mean(axis=1).copy()
    # Deduct initial TC on first day for fair comparison with active strategies
    port_returns.iloc[0] -= bt_cfg.commission_pct

    result.returns = port_returns
    result.portfolio_value = bt_cfg.initial_capital * (1 + port_returns).cumprod()
    result.compute_metrics()

    return result
