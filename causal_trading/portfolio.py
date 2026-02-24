"""
Simple portfolio optimization methods.

Implements:
1. Equal weight - baseline
2. Inverse volatility - weight inversely to vol
3. Minimum variance - classic MVO
4. Causal-weighted - tilt towards causal leaders
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional
from .config import PortfolioConfig
from .causal import CausalResult


def equal_weight(returns: pd.DataFrame) -> pd.Series:
    """Equal weight across all assets."""
    n = returns.shape[1]
    w = pd.Series(1.0 / n, index=returns.columns, name="weight")
    return w


def inverse_volatility(returns: pd.DataFrame) -> pd.Series:
    """Weight inversely proportional to volatility."""
    vols = returns.std()
    inv_vol = 1.0 / (vols + 1e-12)
    w = inv_vol / inv_vol.sum()
    w.name = "weight"
    return w


def min_variance(returns: pd.DataFrame) -> pd.Series:
    """
    Minimum variance portfolio via quadratic optimization.
    Long-only, fully invested.
    """
    cov = returns.cov().values
    n = cov.shape[0]

    # Make positive semi-definite
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-8)
    cov = eigvecs @ np.diag(eigvals) @ eigvecs.T

    def objective(w):
        return w @ cov @ w

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n
    x0 = np.ones(n) / n

    res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if res.success:
        w = pd.Series(res.x, index=returns.columns, name="weight")
    else:
        w = equal_weight(returns)
    return w


def causal_weighted(
    returns: pd.DataFrame,
    causal_result: Optional[CausalResult],
    cfg: PortfolioConfig,
) -> pd.Series:
    """
    Causal-weighted portfolio:
    - Start with minimum variance weights
    - Tilt towards assets that are causal leaders
    - Reduce weight on assets that are pure followers

    Idea: leaders contain more 'original' information and tend to
    be more robust signal sources. Followers are more noise-driven
    and may reverse when the causal link breaks.
    """
    # Start with min-var base
    base_w = min_variance(returns)

    if causal_result is None:
        return base_w

    # Get leader scores
    scores = causal_result.get_leader_scores(alpha=0.05)
    leader_counts = pd.Series(
        {name: info["count"] for name, info in scores.items()}
    )

    if leader_counts.sum() == 0:
        return base_w

    # Normalize to [0, 1]
    leader_norm = leader_counts / (leader_counts.max() + 1e-12)

    # Tilt weights: w_new = (1 - tilt) * base + tilt * leader_score
    tilt = cfg.leader_tilt
    leader_w = leader_norm / (leader_norm.sum() + 1e-12)

    # Align indices
    leader_w = leader_w.reindex(base_w.index, fill_value=0)

    w = (1 - tilt) * base_w + tilt * leader_w
    # Re-normalize
    w = w / w.sum()
    w.name = "weight"
    return w


PORTFOLIO_METHODS = {
    "equal_weight": lambda ret, cr, cfg: equal_weight(ret),
    "inverse_vol": lambda ret, cr, cfg: inverse_volatility(ret),
    "min_variance": lambda ret, cr, cfg: min_variance(ret),
    "causal_weighted": causal_weighted,
}


def optimize_portfolio(
    returns: pd.DataFrame,
    causal_result: Optional[CausalResult],
    cfg: PortfolioConfig,
) -> pd.Series:
    """Dispatch to configured portfolio optimization method."""
    method = cfg.method
    if method not in PORTFOLIO_METHODS:
        raise ValueError(f"Unknown method: {method}. Available: {list(PORTFOLIO_METHODS.keys())}")
    return PORTFOLIO_METHODS[method](returns, causal_result, cfg)
