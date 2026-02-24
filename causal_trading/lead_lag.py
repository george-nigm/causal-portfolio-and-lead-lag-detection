"""
Rolling lead-lag detection.

Core question: How do causal relationships between assets evolve over time?
- When do they appear?
- When do they disappear?
- Are they stable or transient?

This module runs causal inference in rolling windows and tracks the dynamics.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from .config import LeadLagConfig, CausalConfig
from .causal import METHODS, CausalResult


class LeadLagLink:
    """A single directed causal link tracked over time."""

    def __init__(self, cause: str, effect: str):
        self.cause = cause
        self.effect = effect
        # Time series of (window_end_date, p_value, strength, lag)
        self.history: List[Tuple[pd.Timestamp, float, float, int]] = []

    def add_observation(self, date: pd.Timestamp, p_val: float, strength: float, lag: int):
        self.history.append((date, p_val, strength, lag))

    def is_active(self, alpha: float = 0.05) -> bool:
        """Is the link currently significant?"""
        if not self.history:
            return False
        return self.history[-1][1] < alpha

    def active_periods(self, alpha: float = 0.05) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Return list of (start, end) periods where the link was significant."""
        periods = []
        in_period = False
        start = None
        for date, p, _, _ in self.history:
            if p < alpha and not in_period:
                start = date
                in_period = True
            elif p >= alpha and in_period:
                periods.append((start, date))
                in_period = False
        if in_period:
            periods.append((start, self.history[-1][0]))
        return periods

    def stability_score(self, alpha: float = 0.05) -> float:
        """Fraction of windows where the link was significant."""
        if not self.history:
            return 0.0
        active = sum(1 for _, p, _, _ in self.history if p < alpha)
        return active / len(self.history)

    def mean_lag(self, alpha: float = 0.05) -> Optional[float]:
        """Mean lag during active periods."""
        lags = [lag for _, p, _, lag in self.history if p < alpha]
        return np.mean(lags) if lags else None

    def to_series(self) -> pd.DataFrame:
        """Convert history to DataFrame."""
        if not self.history:
            return pd.DataFrame()
        df = pd.DataFrame(
            self.history,
            columns=["date", "p_value", "strength", "lag"]
        )
        df.set_index("date", inplace=True)
        return df


class LeadLagNetwork:
    """
    Dynamic network of causal lead-lag relationships.
    Tracks how the full causal graph evolves over time.
    """

    def __init__(self, assets: List[str]):
        self.assets = assets
        self.links: Dict[Tuple[str, str], LeadLagLink] = {}
        # Initialize all possible directed links
        for cause in assets:
            for effect in assets:
                if cause != effect:
                    self.links[(cause, effect)] = LeadLagLink(cause, effect)

    def get_snapshot(self, alpha: float = 0.05) -> pd.DataFrame:
        """Current state of all links as adjacency matrix of p-values."""
        n = len(self.assets)
        mat = np.ones((n, n))
        for i, effect in enumerate(self.assets):
            for j, cause in enumerate(self.assets):
                link = self.links.get((cause, effect))
                if link and link.history:
                    mat[i, j] = link.history[-1][1]  # latest p-value
        return pd.DataFrame(mat, index=self.assets, columns=self.assets)

    def get_stable_links(self, alpha: float = 0.05, min_stability: float = 0.5):
        """Return links that are significant in at least min_stability fraction of windows."""
        stable = []
        for (cause, effect), link in self.links.items():
            score = link.stability_score(alpha)
            if score >= min_stability:
                mean_lag = link.mean_lag(alpha)
                stable.append({
                    "cause": cause,
                    "effect": effect,
                    "stability": score,
                    "mean_lag": mean_lag,
                    "currently_active": link.is_active(alpha),
                })
        stable.sort(key=lambda x: x["stability"], reverse=True)
        return pd.DataFrame(stable)

    def get_leaders(self, alpha: float = 0.05) -> pd.DataFrame:
        """Rank assets by how many others they lead (currently)."""
        leader_counts = {a: 0 for a in self.assets}
        leader_strength = {a: 0.0 for a in self.assets}
        for (cause, effect), link in self.links.items():
            if link.is_active(alpha):
                leader_counts[cause] += 1
                if link.history:
                    leader_strength[cause] += abs(link.history[-1][2])

        df = pd.DataFrame({
            "n_followers": leader_counts,
            "total_strength": leader_strength,
        })
        df.sort_values("n_followers", ascending=False, inplace=True)
        return df

    def get_regime_changes(self, alpha: float = 0.05) -> List[dict]:
        """
        Detect points where links appear or disappear.
        Returns list of regime change events.
        """
        events = []
        for (cause, effect), link in self.links.items():
            if len(link.history) < 2:
                continue
            for k in range(1, len(link.history)):
                prev_active = link.history[k - 1][1] < alpha
                curr_active = link.history[k][1] < alpha
                if prev_active != curr_active:
                    events.append({
                        "date": link.history[k][0],
                        "cause": cause,
                        "effect": effect,
                        "event": "appeared" if curr_active else "disappeared",
                        "p_value": link.history[k][1],
                        "strength": link.history[k][2],
                    })
        events.sort(key=lambda x: x["date"])
        return events


def rolling_lead_lag_detection(
    returns: pd.DataFrame,
    ll_cfg: LeadLagConfig,
    causal_cfg: CausalConfig,
) -> LeadLagNetwork:
    """
    Run causal inference in rolling windows to track lead-lag dynamics.

    This is the core function for understanding how causal relationships
    evolve over time - when they appear, when they disappear, and how stable they are.
    """
    assets = list(returns.columns)
    network = LeadLagNetwork(assets)

    method = ll_cfg.method
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}. Available: {list(METHODS.keys())}")

    causal_fn = METHODS[method]
    window = ll_cfg.rolling_window
    step = ll_cfg.rolling_step
    T = len(returns)

    n_windows = (T - window) // step + 1
    print(f"  Rolling analysis: {n_windows} windows, size={window}, step={step}")

    for w in range(n_windows):
        start = w * step
        end = start + window
        if end > T:
            break

        window_returns = returns.iloc[start:end]
        window_end_date = returns.index[end - 1]

        # Run causal analysis on this window
        result = causal_fn(window_returns, causal_cfg)

        # Update network
        n = len(assets)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                cause = assets[j]
                effect = assets[i]
                network.links[(cause, effect)].add_observation(
                    date=window_end_date,
                    p_val=result.p_matrix[i, j],
                    strength=result.val_matrix[i, j],
                    lag=int(result.lag_matrix[i, j]),
                )

        if (w + 1) % 10 == 0 or w == 0:
            active = sum(
                1 for link in network.links.values() if link.history and link.history[-1][1] < ll_cfg.significance_level
            )
            print(f"    Window {w + 1}/{n_windows} ({window_end_date.date()}): {active} active links")

    return network
