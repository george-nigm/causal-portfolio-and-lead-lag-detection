"""
Data loading and synthetic data generation.
Generates data with known lead-lag relationships for testing.
"""
import numpy as np
import pandas as pd
from .config import DataConfig


def load_data(cfg: DataConfig) -> pd.DataFrame:
    """Load price data from configured source. Returns DataFrame of close prices."""
    if cfg.source == "yfinance":
        return _load_yfinance(cfg)
    elif cfg.source == "synthetic":
        return _generate_synthetic(cfg)
    elif cfg.source == "csv":
        raise NotImplementedError("CSV loading: pass a path via DataConfig")
    else:
        raise ValueError(f"Unknown data source: {cfg.source}")


def _load_yfinance(cfg: DataConfig) -> pd.DataFrame:
    """Download data from Yahoo Finance."""
    import yfinance as yf
    data = yf.download(cfg.tickers, start=cfg.start_date, end=cfg.end_date)
    prices = data["Close"] if "Close" in data.columns.get_level_values(0) else data
    prices = prices.dropna(axis=1, how="all").ffill().dropna()
    return prices


def _generate_synthetic(cfg: DataConfig) -> pd.DataFrame:
    """
    Generate synthetic price data with known lead-lag causal relationships.

    The ground truth is embedded: some assets' returns are partially driven
    by lagged returns of other assets. This allows us to validate that
    our causal inference methods can recover these relationships.
    """
    np.random.seed(42)
    n = cfg.synthetic_n_obs
    k = cfg.synthetic_n_assets
    strength = cfg.synthetic_lead_lag_strength

    # Generate base innovations (idiosyncratic noise)
    innovations = np.random.randn(n, k) * 0.02  # ~2% daily vol

    # Add AR(1) structure to each series for realism
    returns = np.zeros((n, k))
    for t in range(1, n):
        returns[t] = 0.02 * returns[t - 1] + innovations[t]

    # Embed lead-lag relationships
    for leader_idx, follower_idx, lag in cfg.synthetic_lead_lag:
        if leader_idx < k and follower_idx < k:
            for t in range(lag, n):
                returns[t, follower_idx] += strength * returns[t - lag, leader_idx]

    # Add a sector factor structure: groups of 3 share a common factor
    n_groups = k // 3
    for g in range(n_groups):
        factor = np.random.randn(n) * 0.01
        for j in range(3):
            idx = g * 3 + j
            if idx < k:
                loading = 0.3 + 0.2 * np.random.rand()
                returns[:, idx] += loading * factor

    # Convert returns to prices
    prices = 100 * np.exp(np.cumsum(returns, axis=0))

    # Create DataFrame with date index
    dates = pd.bdate_range(start="2015-01-02", periods=n, freq="B")
    names = [f"Asset_{i}" for i in range(k)]
    df = pd.DataFrame(prices, index=dates, columns=names)
    df.index.name = "Date"

    return df


def compute_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """Compute returns from prices."""
    if method == "log":
        return np.log(prices / prices.shift(1)).dropna()
    elif method == "simple":
        return prices.pct_change().dropna()
    else:
        raise ValueError(f"Unknown return method: {method}")


def get_ground_truth_pairs(cfg: DataConfig):
    """Return the known lead-lag pairs from synthetic config for validation."""
    if cfg.source != "synthetic":
        return None
    pairs = []
    for leader_idx, follower_idx, lag in cfg.synthetic_lead_lag:
        leader = f"Asset_{leader_idx}"
        follower = f"Asset_{follower_idx}"
        pairs.append((leader, follower, lag))
    return pairs
