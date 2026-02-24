"""
Causal pair trading strategy.

Uses causal lead-lag relationships to:
1. Select pairs where one asset causally leads another
2. Generate trading signals when the spread deviates
3. Use the causal direction to determine which side to trade

Key insight: traditional pairs trading uses correlation to find pairs.
We use causal inference to find DIRECTED pairs — if A leads B with lag k,
we can predict B's future moves from A's current/recent moves.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from .config import PairsConfig
from .lead_lag import LeadLagNetwork


class CausalPair:
    """A single causal pair for trading."""

    def __init__(self, leader: str, follower: str, lag: int, strength: float):
        self.leader = leader
        self.follower = follower
        self.lag = lag
        self.strength = strength

    def __repr__(self):
        return f"CausalPair({self.leader} -> {self.follower}, lag={self.lag}, str={self.strength:.3f})"


def select_pairs(
    network: LeadLagNetwork,
    cfg: PairsConfig,
    alpha: float = 0.05,
    min_stability: float = 0.3,
) -> List[CausalPair]:
    """
    Select trading pairs from the causal network.
    Criteria:
    - Link must be currently active (significant)
    - Link should be reasonably stable over time
    - Prefer stronger, more stable relationships
    """
    stable = network.get_stable_links(alpha=alpha, min_stability=min_stability)
    if stable.empty:
        print("  No stable causal pairs found. Lowering thresholds...")
        stable = network.get_stable_links(alpha=0.10, min_stability=0.1)
    if stable.empty:
        return []

    # Sort by stability, take top pairs
    stable = stable.sort_values("stability", ascending=False)
    pairs = []
    used_assets = set()

    for _, row in stable.iterrows():
        if len(pairs) >= cfg.max_pairs:
            break
        leader = row["cause"]
        follower = row["effect"]
        # Avoid reusing assets too much
        if leader in used_assets and follower in used_assets:
            continue
        pairs.append(CausalPair(
            leader=leader,
            follower=follower,
            lag=int(row["mean_lag"]) if pd.notna(row["mean_lag"]) else 1,
            strength=row["stability"],
        ))
        used_assets.add(leader)
        used_assets.add(follower)

    return pairs


def compute_spread(
    prices: pd.DataFrame,
    pair: CausalPair,
    lookback: int = 60,
) -> pd.DataFrame:
    """
    Compute the spread between leader and follower.

    Two approaches:
    1. Price ratio (log): log(follower) - beta * log(leader), with rolling beta
    2. Return prediction residual: follower_return - predicted_from_leader

    We use approach 1 (more standard for pairs trading).
    """
    leader_log = np.log(prices[pair.leader])
    follower_log = np.log(prices[pair.follower])

    # Rolling OLS: follower = alpha + beta * leader
    spread = pd.Series(np.nan, index=prices.index, name="spread")
    z_score = pd.Series(np.nan, index=prices.index, name="z_score")
    beta_series = pd.Series(np.nan, index=prices.index, name="beta")

    for t in range(lookback, len(prices)):
        window_leader = leader_log.iloc[t - lookback: t]
        window_follower = follower_log.iloc[t - lookback: t]

        # Simple OLS
        x = window_leader.values
        y = window_follower.values
        x_mean = x.mean()
        beta = np.sum((x - x_mean) * (y - y.mean())) / (np.sum((x - x_mean) ** 2) + 1e-12)
        alpha = y.mean() - beta * x_mean

        # Current spread
        s = follower_log.iloc[t] - beta * leader_log.iloc[t] - alpha
        spread.iloc[t] = s
        beta_series.iloc[t] = beta

        # Z-score from rolling window spread stats
        window_spread = follower_log.iloc[t - lookback: t] - beta * leader_log.iloc[t - lookback: t] - alpha
        mu = window_spread.mean()
        sigma = window_spread.std()
        z_score.iloc[t] = (s - mu) / (sigma + 1e-12)

    return pd.DataFrame({
        "spread": spread,
        "z_score": z_score,
        "beta": beta_series,
    })


def generate_pair_signals(
    prices: pd.DataFrame,
    pair: CausalPair,
    cfg: PairsConfig,
) -> pd.DataFrame:
    """
    Generate trading signals for a causal pair.

    Signal logic:
    - When z_score > entry_z: short the follower, long the leader (spread will revert)
    - When z_score < -entry_z: long the follower, short the leader
    - When |z_score| < exit_z: close positions

    The causal direction adds an asymmetry:
    - If leader moves first and follower hasn't followed yet,
      that's a stronger signal than the reverse.
    """
    spread_data = compute_spread(prices, pair, lookback=cfg.lookback)
    z = spread_data["z_score"]

    # Position: +1 = long follower/short leader, -1 = reverse, 0 = flat
    position = pd.Series(0.0, index=prices.index, name="position")
    signal = pd.Series(0.0, index=prices.index, name="signal")

    current_pos = 0.0
    for t in range(1, len(prices)):
        if pd.isna(z.iloc[t]):
            continue

        z_val = z.iloc[t]

        if current_pos == 0:
            # Entry signals
            if z_val > cfg.entry_z:
                current_pos = -1.0  # spread too high -> short spread
                signal.iloc[t] = -1.0
            elif z_val < -cfg.entry_z:
                current_pos = 1.0   # spread too low -> long spread
                signal.iloc[t] = 1.0
        else:
            # Exit signals
            if abs(z_val) < cfg.exit_z:
                signal.iloc[t] = -current_pos  # reverse to close
                current_pos = 0.0
            # Also exit on crossing zero after extreme
            elif current_pos > 0 and z_val > cfg.entry_z:
                signal.iloc[t] = -current_pos - 1.0
                current_pos = -1.0
            elif current_pos < 0 and z_val < -cfg.entry_z:
                signal.iloc[t] = -current_pos + 1.0
                current_pos = 1.0

        position.iloc[t] = current_pos

    # Causal enhancement: weight signals by leader's recent move
    if cfg.use_causal_direction:
        leader_ret = prices[pair.leader].pct_change()
        # If leader moved strongly in the predicted direction, increase confidence
        for t in range(pair.lag, len(prices)):
            if signal.iloc[t] != 0:
                leader_move = leader_ret.iloc[t - pair.lag: t].sum()
                # Amplify signal if leader confirms direction
                if (signal.iloc[t] > 0 and leader_move > 0) or \
                   (signal.iloc[t] < 0 and leader_move < 0):
                    position.iloc[t] *= 1.2  # 20% boost

    return pd.DataFrame({
        "spread": spread_data["spread"],
        "z_score": z,
        "beta": spread_data["beta"],
        "signal": signal,
        "position": position,
    })


def run_pairs_strategy(
    prices: pd.DataFrame,
    pairs: List[CausalPair],
    cfg: PairsConfig,
) -> Dict[str, pd.DataFrame]:
    """Run pair trading signals for all selected pairs."""
    results = {}
    for pair in pairs:
        if pair.leader in prices.columns and pair.follower in prices.columns:
            signals = generate_pair_signals(prices, pair, cfg)
            results[f"{pair.leader}->{pair.follower}"] = signals
    return results
