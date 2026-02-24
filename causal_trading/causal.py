"""
Causal inference methods for detecting lead-lag relationships.

Methods implemented:
1. Granger Causality - classic VAR-based test
2. Transfer Entropy - information-theoretic, captures non-linear dependencies
3. Cross-Correlation Function (CCF) - simple lag-based correlation
4. Lightweight PCMCI - partial correlation with conditional independence testing
"""
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
from typing import Dict, Tuple, Optional
from .config import CausalConfig


class CausalResult:
    """Container for causal analysis results."""

    def __init__(self, method: str, assets: list):
        self.method = method
        self.assets = assets
        n = len(assets)
        # p_matrix[i, j] = p-value that j -> i (j Granger-causes i)
        self.p_matrix = np.ones((n, n))
        # val_matrix[i, j] = strength of j -> i relationship
        self.val_matrix = np.zeros((n, n))
        # lag_matrix[i, j] = optimal lag for j -> i
        self.lag_matrix = np.zeros((n, n), dtype=int)

    def get_significant_links(self, alpha: float = 0.05):
        """Return list of (cause, effect, lag, strength, pval) tuples."""
        links = []
        n = len(self.assets)
        for i in range(n):
            for j in range(n):
                if i != j and self.p_matrix[i, j] < alpha:
                    links.append((
                        self.assets[j],   # cause
                        self.assets[i],   # effect
                        int(self.lag_matrix[i, j]),
                        float(self.val_matrix[i, j]),
                        float(self.p_matrix[i, j])
                    ))
        links.sort(key=lambda x: x[4])  # sort by p-value
        return links

    def get_leader_scores(self, alpha: float = 0.05):
        """
        Score each asset by how many others it causally leads.
        Higher score = more assets follow this asset = stronger leader.
        """
        scores = {}
        n = len(self.assets)
        for j, name in enumerate(self.assets):
            # Count how many assets j Granger-causes
            count = sum(1 for i in range(n) if i != j and self.p_matrix[i, j] < alpha)
            strength = sum(
                abs(self.val_matrix[i, j])
                for i in range(n) if i != j and self.p_matrix[i, j] < alpha
            )
            scores[name] = {"count": count, "strength": strength}
        return scores

    def to_dataframe(self):
        """Return p-value matrix as labeled DataFrame."""
        return pd.DataFrame(
            self.p_matrix,
            index=self.assets,
            columns=self.assets
        )


# ---------------------------------------------------------------------------
# Method 1: Granger Causality
# ---------------------------------------------------------------------------

def granger_causality(returns: pd.DataFrame, cfg: CausalConfig) -> CausalResult:
    """
    Bivariate Granger causality test for all pairs.
    Tests whether past values of X improve prediction of Y beyond Y's own past.
    """
    assets = list(returns.columns)
    result = CausalResult("granger", assets)
    n = len(assets)
    max_lag = cfg.max_lag
    test_name = cfg.granger_test

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            y_col = assets[i]  # effect
            x_col = assets[j]  # potential cause
            data = returns[[y_col, x_col]].dropna()

            if len(data) < max_lag + 10:
                continue

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    test_result = grangercausalitytests(
                        data.values, maxlag=max_lag, verbose=False
                    )
                # Get p-values for each lag
                p_values = [
                    test_result[lag + 1][0][test_name][1]
                    for lag in range(max_lag)
                ]
                best_lag = int(np.argmin(p_values))
                result.p_matrix[i, j] = p_values[best_lag]
                result.lag_matrix[i, j] = best_lag + 1

                # Use F-statistic as strength measure
                f_stat = test_result[best_lag + 1][0][test_name][0]
                result.val_matrix[i, j] = f_stat
            except Exception:
                pass  # keep defaults (p=1, val=0)

    return result


# ---------------------------------------------------------------------------
# Method 2: Transfer Entropy
# ---------------------------------------------------------------------------

def _discretize(x: np.ndarray, bins: int) -> np.ndarray:
    """Discretize a continuous series into bins."""
    percentiles = np.linspace(0, 100, bins + 1)
    edges = np.percentile(x, percentiles)
    edges[0] -= 1e-10
    edges[-1] += 1e-10
    return np.digitize(x, edges[1:-1])


def _compute_transfer_entropy(
    x: np.ndarray, y: np.ndarray, lag: int, k: int, bins: int
) -> float:
    """
    Compute transfer entropy from X to Y: TE(X->Y).
    TE(X->Y) = H(Y_t | Y_{t-1:t-k}) - H(Y_t | Y_{t-1:t-k}, X_{t-lag})

    Uses discrete (binned) estimation.
    """
    n = len(x)
    if n < lag + k + 1:
        return 0.0

    xd = _discretize(x, bins)
    yd = _discretize(y, bins)

    # Build joint distributions
    # Y_future, Y_past (k-dim embedded), X_lagged
    y_fut = yd[lag + k:]
    y_past = np.column_stack([yd[lag + k - i - 1: n - i - 1] for i in range(k)])
    x_lag = xd[k: n - lag]

    min_len = min(len(y_fut), len(y_past), len(x_lag))
    y_fut = y_fut[:min_len]
    y_past = y_past[:min_len]
    x_lag = x_lag[:min_len]

    # Encode states as integers for counting
    def encode_states(*arrays):
        result = arrays[0].copy().astype(int)
        for a in arrays[1:]:
            if a.ndim == 1:
                result = result * (bins + 1) + a.astype(int)
            else:
                for col in range(a.shape[1]):
                    result = result * (bins + 1) + a[:, col].astype(int)
        return result

    # H(Y_fut | Y_past) - H(Y_fut | Y_past, X_lag)
    # = H(Y_fut, Y_past) - H(Y_past) - H(Y_fut, Y_past, X_lag) + H(Y_past, X_lag)

    def entropy(labels):
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs + 1e-12))

    if y_past.ndim == 1:
        y_past = y_past.reshape(-1, 1)

    # Joint entropies
    yf_yp = encode_states(y_fut, y_past)
    yp_only = encode_states(y_past[:, 0]) if y_past.shape[1] == 1 else encode_states(*[y_past[:, c] for c in range(y_past.shape[1])])
    yf_yp_xl = encode_states(y_fut, y_past, x_lag)
    yp_xl = encode_states(y_past[:, 0], x_lag) if y_past.shape[1] == 1 else encode_states(*[y_past[:, c] for c in range(y_past.shape[1])], x_lag)

    te = entropy(yf_yp) - entropy(yp_only) - entropy(yf_yp_xl) + entropy(yp_xl)
    return max(te, 0.0)


def _transfer_entropy_significance(
    x: np.ndarray, y: np.ndarray, lag: int, k: int, bins: int,
    te_observed: float, n_surrogates: int = 100
) -> float:
    """Compute p-value for TE via time-shifted surrogates."""
    te_surrogates = []
    for _ in range(n_surrogates):
        x_shuffled = np.random.permutation(x)
        te_s = _compute_transfer_entropy(x_shuffled, y, lag, k, bins)
        te_surrogates.append(te_s)
    te_surrogates = np.array(te_surrogates)
    p_value = np.mean(te_surrogates >= te_observed)
    return p_value


def transfer_entropy(returns: pd.DataFrame, cfg: CausalConfig) -> CausalResult:
    """
    Transfer entropy for all pairs across multiple lags.
    Non-parametric, captures non-linear causal relationships.
    """
    assets = list(returns.columns)
    result = CausalResult("transfer_entropy", assets)
    n = len(assets)
    vals = returns.values

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            best_te = 0.0
            best_lag = 1
            best_pval = 1.0

            for lag in range(1, cfg.max_lag + 1):
                te = _compute_transfer_entropy(
                    vals[:, j], vals[:, i],
                    lag=lag, k=cfg.te_k, bins=cfg.te_bins
                )
                if te > best_te:
                    best_te = te
                    best_lag = lag

            # Significance test for best lag only (saves time)
            if best_te > 0:
                best_pval = _transfer_entropy_significance(
                    vals[:, j], vals[:, i],
                    lag=best_lag, k=cfg.te_k, bins=cfg.te_bins,
                    te_observed=best_te, n_surrogates=100
                )

            result.val_matrix[i, j] = best_te
            result.p_matrix[i, j] = best_pval
            result.lag_matrix[i, j] = best_lag

    return result


# ---------------------------------------------------------------------------
# Method 3: Cross-Correlation Function (CCF)
# ---------------------------------------------------------------------------

def cross_correlation(returns: pd.DataFrame, cfg: CausalConfig) -> CausalResult:
    """
    Cross-correlation at various lags.
    Simple and fast; not truly causal but useful for detecting lead-lag.
    Tests significance against white noise null (Bartlett's formula).
    """
    assets = list(returns.columns)
    result = CausalResult("ccf", assets)
    n = len(assets)
    T = len(returns)
    vals = returns.values

    # Standardize
    means = vals.mean(axis=0)
    stds = vals.std(axis=0)
    z = (vals - means) / (stds + 1e-12)

    # Significance threshold under white noise null
    se = 1.0 / np.sqrt(T)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            best_corr = 0.0
            best_lag = 0
            for lag in range(1, cfg.max_lag + 1):
                # corr(Y_t, X_{t-lag}) — does X lead Y?
                corr = np.corrcoef(z[lag:, i], z[:-lag, j])[0, 1]
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag

            result.val_matrix[i, j] = best_corr
            result.lag_matrix[i, j] = best_lag
            # Approximate p-value using Fisher z-transform
            fisher_z = np.arctanh(np.clip(best_corr, -0.999, 0.999))
            p_val = 2 * (1 - stats.norm.cdf(abs(fisher_z) / se))
            result.p_matrix[i, j] = p_val

    return result


# ---------------------------------------------------------------------------
# Method 4: Lightweight PCMCI-style (partial correlation with conditioning)
# ---------------------------------------------------------------------------

def _partial_correlation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[float, float]:
    """Compute partial correlation between x and y, controlling for z."""
    if z.size == 0:
        r, p = stats.pearsonr(x, y)
        return r, p
    # Regress x on z, y on z, correlate residuals
    from numpy.linalg import lstsq
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    z_aug = np.column_stack([z, np.ones(len(z))])

    coef_x, _, _, _ = lstsq(z_aug, x, rcond=None)
    coef_y, _, _, _ = lstsq(z_aug, y, rcond=None)
    res_x = x - z_aug @ coef_x
    res_y = y - z_aug @ coef_y

    if res_x.std() < 1e-12 or res_y.std() < 1e-12:
        return 0.0, 1.0
    r, p = stats.pearsonr(res_x, res_y)
    return r, p


def pcmci_lite(returns: pd.DataFrame, cfg: CausalConfig) -> CausalResult:
    """
    Simplified PCMCI-like algorithm:
    1. For each pair (X->Y), test lagged partial correlations
    2. Condition on other variables' contemporaneous values
    3. Select best lag and compute significance

    This is a lightweight version that doesn't require tigramite.
    """
    assets = list(returns.columns)
    result = CausalResult("pcmci_lite", assets)
    n = len(assets)
    vals = returns.values
    T = len(returns)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            best_val = 0.0
            best_pval = 1.0
            best_lag = 1

            for lag in range(1, cfg.max_lag + 1):
                y_target = vals[lag:, i]
                x_cause = vals[:-lag, j]
                min_len = min(len(y_target), len(x_cause))
                y_target = y_target[:min_len]
                x_cause = x_cause[:min_len]

                # Condition on Y's own past and other variables' contemporaneous
                conditions = []
                # Y's own past (1 lag)
                y_own_past = vals[lag - 1: lag - 1 + min_len, i]
                conditions.append(y_own_past)
                # Other variables' contemporaneous
                for k_idx in range(n):
                    if k_idx != i and k_idx != j:
                        other = vals[lag: lag + min_len, k_idx]
                        conditions.append(other)

                if conditions:
                    z = np.column_stack(conditions)
                else:
                    z = np.array([])

                r, p = _partial_correlation(x_cause, y_target, z)
                if abs(r) > abs(best_val):
                    best_val = r
                    best_pval = p
                    best_lag = lag

            result.val_matrix[i, j] = best_val
            result.p_matrix[i, j] = best_pval
            result.lag_matrix[i, j] = best_lag

    return result


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

METHODS = {
    "granger": granger_causality,
    "transfer_entropy": transfer_entropy,
    "ccf": cross_correlation,
    "pcmci_lite": pcmci_lite,
}


def run_causal_analysis(
    returns: pd.DataFrame, cfg: CausalConfig
) -> Dict[str, CausalResult]:
    """Run all configured causal methods and return results."""
    results = {}
    for method_name in cfg.methods:
        if method_name in METHODS:
            print(f"  Running {method_name}...")
            results[method_name] = METHODS[method_name](returns, cfg)
        else:
            print(f"  Unknown method: {method_name}, skipping")
    return results
