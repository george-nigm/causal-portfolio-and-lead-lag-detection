"""
Default configuration for causal trading pipeline.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    # Stock tickers to analyze
    tickers: List[str] = field(default_factory=lambda: [
        "XLF", "XLE", "XLK", "XLV", "XLI",
        "XLP", "XLU", "XLB", "XLY", "XLRE"
    ])
    start_date: str = "2015-01-01"
    end_date: str = "2023-12-31"
    # 'yfinance' or 'synthetic'
    source: str = "synthetic"
    # For synthetic: number of assets with embedded lead-lag
    synthetic_n_assets: int = 10
    synthetic_n_obs: int = 2000
    # Known lead-lag pairs for synthetic data: list of (leader_idx, follower_idx, lag)
    synthetic_lead_lag: List[tuple] = field(default_factory=lambda: [
        (0, 1, 2), (2, 3, 3), (4, 5, 1), (0, 6, 5)
    ])
    synthetic_lead_lag_strength: float = 0.3


@dataclass
class CausalConfig:
    # Methods to run: 'granger', 'transfer_entropy', 'ccf', 'pcmci_lite'
    methods: List[str] = field(default_factory=lambda: [
        "granger", "transfer_entropy", "ccf"
    ])
    max_lag: int = 10
    significance_level: float = 0.05
    # Granger
    granger_test: str = "ssr_chi2test"
    # Transfer entropy
    te_bins: int = 10
    te_k: int = 1  # embedding dimension for TE


@dataclass
class LeadLagConfig:
    # Rolling window for dynamic lead-lag detection
    rolling_window: int = 252  # ~1 year of trading days
    rolling_step: int = 22     # ~1 month step
    # Significance threshold for declaring a link active
    significance_level: float = 0.05
    # Minimum consecutive windows for a relationship to be "stable"
    min_stable_windows: int = 3
    # Method for rolling analysis
    method: str = "granger"


@dataclass
class PairsConfig:
    # Z-score entry threshold
    entry_z: float = 2.0
    # Z-score exit threshold
    exit_z: float = 0.5
    # Lookback for spread mean/std
    lookback: int = 60
    # Max pairs to trade simultaneously
    max_pairs: int = 5
    # Use causal direction for asymmetric signals
    use_causal_direction: bool = True


@dataclass
class PortfolioConfig:
    # 'equal_weight', 'inverse_vol', 'min_variance', 'causal_weighted'
    method: str = "causal_weighted"
    rebalance_days: int = 22
    # For causal_weighted: how much to tilt towards causal leaders
    leader_tilt: float = 0.3


@dataclass
class BacktestConfig:
    initial_capital: float = 100000.0
    commission_pct: float = 0.001  # 10 bps
    slippage_pct: float = 0.0005   # 5 bps
    # Out-of-sample start (fraction of data)
    train_fraction: float = 0.6


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    causal: CausalConfig = field(default_factory=CausalConfig)
    lead_lag: LeadLagConfig = field(default_factory=LeadLagConfig)
    pairs: PairsConfig = field(default_factory=PairsConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
