"""
Sector-based synthetic data generation with macro regime features.

Implements SPDR-style sector ETF simulation with:
- Macro regime switching (Expansion, Contraction, Crisis)
- Regime-dependent causal lead-lag relationships between sectors
- Auto-computed regime indicators and macro features
- Configurable parameters that propagate through the entire system

The key idea: macro regimes drive both sector returns AND the causal
structure between sectors. In expansion, Tech leads Consumer Discretionary;
in crisis, Financials contagion spreads to all sectors. Changing any
parameter (regime duration, transition probabilities, macro sensitivities)
automatically cascades through the entire data generation process.

Sectors modeled after SPDR (Spider) ETFs based on GICS classification.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


# ===========================================================================
# SECTOR DEFINITIONS (SPDR / GICS)
# ===========================================================================

SECTOR_INFO = {
    "XLK": {"name": "Technology",              "gics": 45, "base_vol": 0.018, "market_beta": 1.15},
    "XLF": {"name": "Financials",              "gics": 40, "base_vol": 0.016, "market_beta": 1.10},
    "XLE": {"name": "Energy",                  "gics": 10, "base_vol": 0.022, "market_beta": 1.05},
    "XLV": {"name": "Healthcare",              "gics": 35, "base_vol": 0.014, "market_beta": 0.85},
    "XLY": {"name": "Consumer Discretionary",  "gics": 25, "base_vol": 0.017, "market_beta": 1.10},
    "XLP": {"name": "Consumer Staples",        "gics": 30, "base_vol": 0.010, "market_beta": 0.65},
    "XLI": {"name": "Industrials",             "gics": 20, "base_vol": 0.015, "market_beta": 1.00},
    "XLB": {"name": "Materials",               "gics": 15, "base_vol": 0.018, "market_beta": 1.05},
    "XLU": {"name": "Utilities",               "gics": 55, "base_vol": 0.012, "market_beta": 0.55},
    "XLRE": {"name": "Real Estate",            "gics": 60, "base_vol": 0.016, "market_beta": 0.80},
    "XLC": {"name": "Communication Services",  "gics": 50, "base_vol": 0.017, "market_beta": 1.05},
}

SECTOR_TICKERS = list(SECTOR_INFO.keys())

# ===========================================================================
# MACRO SENSITIVITY MATRIX
# ===========================================================================
# How each macro variable affects each sector's returns.
# Columns: [vix, rate_change, credit_spread, dollar_strength, oil_change, yield_curve_slope]
# Positive = benefits from increase, Negative = hurt by increase.

MACRO_SENSITIVITIES = {
    #             vix    rate   credit  dollar   oil   yield_curve
    "XLK":  [-0.15, -0.08, -0.10, -0.05, -0.02,  0.05],
    "XLF":  [-0.20,  0.15, -0.25,  0.05, -0.03,  0.20],
    "XLE":  [-0.10,  0.02, -0.05, -0.12,  0.35,  0.03],
    "XLV":  [-0.08, -0.05, -0.05,  0.03, -0.02,  0.02],
    "XLY":  [-0.18, -0.10, -0.12, -0.03, -0.08,  0.08],
    "XLP":  [-0.05, -0.03, -0.02,  0.02, -0.03,  0.01],
    "XLI":  [-0.15, -0.05, -0.10, -0.08, -0.05,  0.10],
    "XLB":  [-0.12, -0.03, -0.08, -0.15,  0.10,  0.08],
    "XLU":  [-0.06, -0.15, -0.08,  0.03, -0.02, -0.10],
    "XLRE": [-0.15, -0.20, -0.20,  0.02, -0.02, -0.15],
    "XLC":  [-0.14, -0.07, -0.08, -0.04, -0.02,  0.04],
}

MACRO_NAMES = ["vix", "rate_change", "credit_spread",
               "dollar_strength", "oil_change", "yield_curve_slope"]

# ===========================================================================
# REGIME-DEPENDENT CAUSAL LINKS
# ===========================================================================
# Format: (leader, follower, lag_days, strength)
# Strength is the coefficient: follower_ret += strength * leader_ret[t-lag]

REGIME_CAUSAL_LINKS = {
    "expansion": [
        # Tech innovation leads consumer spending & communication
        ("XLK", "XLY",  2, 0.22),
        ("XLK", "XLC",  1, 0.18),
        # Easy financial conditions lead industrial activity & real estate
        ("XLF", "XLI",  3, 0.20),
        ("XLF", "XLRE", 2, 0.24),
        # Consumer demand leads industrial production
        ("XLY", "XLI",  2, 0.16),
        # Energy prices lead materials costs
        ("XLE", "XLB",  2, 0.15),
    ],
    "contraction": [
        # Energy cost pressure leads materials & industrial slowdown
        ("XLE", "XLB",  2, 0.25),
        ("XLE", "XLI",  3, 0.18),
        # Financial tightening leads real estate & consumer stress
        ("XLF", "XLRE", 1, 0.30),
        ("XLF", "XLY",  2, 0.20),
        # Defensive rotation: Staples lead Utilities
        ("XLP", "XLU",  2, 0.22),
        # Industrial slowdown leads consumer cutback
        ("XLI", "XLY",  3, 0.18),
        # Tech selloff leads communication
        ("XLK", "XLC",  1, 0.20),
    ],
    "crisis": [
        # Financial contagion spreads to all cyclical sectors
        ("XLF", "XLI",  1, 0.38),
        ("XLF", "XLRE", 1, 0.42),
        ("XLF", "XLY",  1, 0.32),
        ("XLF", "XLB",  2, 0.25),
        # Energy crash leads materials & industrials
        ("XLE", "XLB",  1, 0.30),
        ("XLE", "XLI",  2, 0.28),
        # Tech crash leads communication
        ("XLK", "XLC",  1, 0.30),
        # Consumer crash leads communication
        ("XLY", "XLC",  1, 0.22),
        # Healthcare relatively leads utilities (flight to safety)
        ("XLV", "XLU",  1, 0.15),
    ],
}


# ===========================================================================
# CONFIGURATION
# ===========================================================================

@dataclass
class RegimeConfig:
    """Configuration for macro regime switching."""
    # Transition probability matrix [from_regime, to_regime]
    # Rows: expansion, contraction, crisis; Cols: same
    transition_matrix: np.ndarray = field(default_factory=lambda: np.array([
        [0.96, 0.03, 0.01],  # From expansion
        [0.04, 0.92, 0.04],  # From contraction
        [0.03, 0.07, 0.90],  # From crisis
    ]))
    # Labels
    regime_names: List[str] = field(default_factory=lambda: [
        "expansion", "contraction", "crisis"
    ])
    # Starting regime
    initial_regime: int = 0  # expansion


@dataclass
class MacroConfig:
    """Configuration for macro variable generation per regime."""
    # Mean and volatility of macro variable CHANGES per regime
    # Shape: {macro_name: {regime_name: (mean, vol)}}
    distributions: Dict = field(default_factory=lambda: {
        "vix": {
            "expansion":   (-0.02, 0.08),   # VIX declining
            "contraction": ( 0.03, 0.12),   # VIX rising
            "crisis":      ( 0.08, 0.20),   # VIX spiking
        },
        "rate_change": {
            "expansion":   ( 0.005, 0.01),  # Rates rising (tightening)
            "contraction": (-0.003, 0.01),  # Rates falling
            "crisis":      (-0.010, 0.02),  # Emergency rate cuts
        },
        "credit_spread": {
            "expansion":   (-0.01, 0.03),   # Spreads tightening
            "contraction": ( 0.02, 0.05),   # Spreads widening
            "crisis":      ( 0.06, 0.10),   # Spreads blowing out
        },
        "dollar_strength": {
            "expansion":   ( 0.001, 0.005),
            "contraction": ( 0.003, 0.008),  # Dollar strengthening (safe haven)
            "crisis":      ( 0.005, 0.012),  # Dollar strengthening more
        },
        "oil_change": {
            "expansion":   ( 0.003, 0.025),  # Oil rising with demand
            "contraction": (-0.005, 0.030),  # Oil falling
            "crisis":      (-0.015, 0.045),  # Oil crashing
        },
        "yield_curve_slope": {
            "expansion":   ( 0.002, 0.008),  # Curve steepening
            "contraction": (-0.004, 0.010),  # Curve flattening
            "crisis":      (-0.008, 0.015),  # Curve inverting
        },
    })
    # AR(1) persistence for macro variables (smoothing)
    macro_persistence: float = 0.92


@dataclass
class SectorGenConfig:
    """Full configuration for sector data generation."""
    n_obs: int = 3000          # Number of trading days (~12 years)
    start_date: str = "2012-01-03"
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    macro: MacroConfig = field(default_factory=MacroConfig)
    # Global lead-lag strength multiplier (scales all causal links)
    lead_lag_strength: float = 1.0
    # Noise multiplier (scales idiosyncratic volatility)
    noise_scale: float = 1.0
    # Market factor volatility
    market_vol: float = 0.012
    # Market factor AR(1) persistence
    market_persistence: float = 0.05
    # Random seed
    seed: int = 42


# ===========================================================================
# GENERATION ENGINE
# ===========================================================================

class SectorDataGenerator:
    """
    Generates synthetic sector return data with regime-dependent dynamics.

    The generation follows a cascading dependency structure:
    1. Regime sequence → determined by transition matrix
    2. Macro variables → conditioned on current regime
    3. Market factor → GARCH-like, regime-dependent volatility
    4. Sector returns → market + macro + causal lead-lag + idiosyncratic
    5. Regime features → auto-computed from generated data

    Changing ANY parameter automatically propagates through all downstream
    computations. For example:
    - Changing transition_matrix → different regime sequence → different macro
      variables → different sector returns → different causal structure
    - Changing lead_lag_strength → same regimes but different detectability
    """

    def __init__(self, config: SectorGenConfig):
        self.cfg = config
        self.rng = np.random.RandomState(config.seed)

        # Storage for generated data
        self.regime_sequence: Optional[np.ndarray] = None
        self.macro_data: Optional[pd.DataFrame] = None
        self.sector_returns: Optional[pd.DataFrame] = None
        self.sector_prices: Optional[pd.DataFrame] = None
        self.regime_features: Optional[pd.DataFrame] = None
        self.all_features: Optional[pd.DataFrame] = None
        self.ground_truth: Optional[Dict] = None

    def generate(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run the full generation pipeline.

        Returns:
            prices: sector prices (n_obs x 11)
            features: all computed features including regime indicators
            ground_truth_df: DataFrame of true causal links per regime
        """
        self._generate_regime_sequence()
        self._generate_macro_variables()
        self._generate_sector_returns()
        self._compute_regime_features()
        self._compute_all_features()
        self._build_ground_truth()

        return self.sector_prices, self.all_features, self._ground_truth_df()

    # ---- Step 1: Regime Sequence ----

    def _generate_regime_sequence(self):
        """Generate Markov chain regime sequence."""
        n = self.cfg.n_obs
        tm = self.cfg.regime.transition_matrix
        regimes = np.zeros(n, dtype=int)
        regimes[0] = self.cfg.regime.initial_regime

        for t in range(1, n):
            probs = tm[regimes[t - 1]]
            regimes[t] = self.rng.choice(len(probs), p=probs)

        self.regime_sequence = regimes

    # ---- Step 2: Macro Variables ----

    def _generate_macro_variables(self):
        """Generate macro variables conditioned on regime."""
        n = self.cfg.n_obs
        regimes = self.regime_sequence
        regime_names = self.cfg.regime.regime_names
        phi = self.cfg.macro.macro_persistence
        distributions = self.cfg.macro.distributions

        dates = pd.bdate_range(start=self.cfg.start_date, periods=n, freq="B")
        macro_data = {}

        for macro_name in MACRO_NAMES:
            series = np.zeros(n)
            for t in range(1, n):
                regime = regime_names[regimes[t]]
                mu, sigma = distributions[macro_name][regime]
                innovation = self.rng.randn() * sigma
                series[t] = phi * series[t - 1] + (1 - phi) * mu + innovation
            macro_data[macro_name] = series

        self.macro_data = pd.DataFrame(macro_data, index=dates)
        self.macro_data.index.name = "Date"

    # ---- Step 3: Sector Returns ----

    def _generate_sector_returns(self):
        """
        Generate sector returns using the multi-factor model:

        r_i,t = β_market_i * r_market,t
              + Σ_m (γ_i,m * Δmacro_m,t)
              + Σ_j (δ_ij,regime(t) * r_j,t-lag)   [causal lead-lag]
              + ε_i,t                                [idiosyncratic]

        The causal lead-lag term is regime-dependent: different regimes
        activate different sets of causal relationships.
        """
        n = self.cfg.n_obs
        dates = self.macro_data.index
        regimes = self.regime_sequence
        regime_names = self.cfg.regime.regime_names

        # --- Market factor ---
        market_vol = self.cfg.market_vol
        regime_vol_scale = {"expansion": 1.0, "contraction": 1.5, "crisis": 2.5}
        market_returns = np.zeros(n)
        current_vol = market_vol

        for t in range(1, n):
            regime = regime_names[regimes[t]]
            target_vol = market_vol * regime_vol_scale[regime]
            # GARCH-like: slow vol adjustment
            current_vol = 0.95 * current_vol + 0.05 * target_vol
            market_returns[t] = (self.cfg.market_persistence * market_returns[t - 1]
                                 + self.rng.randn() * current_vol)

        # --- Build sector returns ---
        sector_returns = np.zeros((n, len(SECTOR_TICKERS)))
        macro_vals = self.macro_data.values  # (n, 6)

        # Macro sensitivity matrix
        sensitivity_matrix = np.array([MACRO_SENSITIVITIES[s] for s in SECTOR_TICKERS])  # (11, 6)

        for t in range(1, n):
            for s_idx, ticker in enumerate(SECTOR_TICKERS):
                info = SECTOR_INFO[ticker]

                # Market component
                mkt_component = info["market_beta"] * market_returns[t]

                # Macro component: sensitivity * macro variable CHANGE
                macro_changes = macro_vals[t] - macro_vals[t - 1]
                macro_component = sensitivity_matrix[s_idx] @ macro_changes

                # Idiosyncratic component
                idio_vol = info["base_vol"] * self.cfg.noise_scale
                regime = regime_names[regimes[t]]
                if regime == "contraction":
                    idio_vol *= 1.3
                elif regime == "crisis":
                    idio_vol *= 2.0
                idio_component = self.rng.randn() * idio_vol

                sector_returns[t, s_idx] = mkt_component + macro_component + idio_component

        # --- Embed causal lead-lag relationships ---
        for t in range(1, n):
            regime = regime_names[regimes[t]]
            links = REGIME_CAUSAL_LINKS.get(regime, [])
            for leader, follower, lag, strength in links:
                if t >= lag:
                    l_idx = SECTOR_TICKERS.index(leader)
                    f_idx = SECTOR_TICKERS.index(follower)
                    sector_returns[t, f_idx] += (
                        strength * self.cfg.lead_lag_strength * sector_returns[t - lag, l_idx]
                    )

        # Store market returns for feature computation
        self._market_returns = market_returns

        # Convert to DataFrame
        self.sector_returns = pd.DataFrame(
            sector_returns, index=dates, columns=SECTOR_TICKERS
        )
        self.sector_returns.index.name = "Date"

        # Convert to prices
        self.sector_prices = 100 * np.exp(self.sector_returns.cumsum())
        self.sector_prices.index.name = "Date"

    # ---- Step 4: Regime Features (auto-computed) ----

    def _compute_regime_features(self):
        """
        Compute regime indicator features from the generated data.
        These are observable features that reflect the underlying regime.
        They update automatically whenever parameters change.
        """
        n = self.cfg.n_obs
        dates = self.macro_data.index
        returns = self.sector_returns.values

        features = {}

        # --- A. Volatility Regime ---
        # Rolling realized volatility of market (proxy for VIX)
        window = 21
        market_ret = self._market_returns
        realized_vol = pd.Series(market_ret, index=dates).rolling(window).std() * np.sqrt(252)
        features["realized_vol"] = realized_vol.values

        # Classify into low/medium/high volatility
        vol_33 = np.nanpercentile(realized_vol.dropna(), 33)
        vol_67 = np.nanpercentile(realized_vol.dropna(), 67)
        vol_regime = np.where(realized_vol < vol_33, 0,
                     np.where(realized_vol < vol_67, 1, 2))
        features["vol_regime"] = vol_regime  # 0=low, 1=medium, 2=high

        # --- B. Trend Regime ---
        # Rolling cumulative market return (60-day)
        cum_market = pd.Series(market_ret, index=dates).rolling(60).sum()
        features["market_trend"] = cum_market.values
        trend_regime = np.where(cum_market > 0.02, 0,       # Bull
                       np.where(cum_market > -0.02, 1, 2))  # Sideways / Bear
        features["trend_regime"] = trend_regime  # 0=bull, 1=sideways, 2=bear

        # --- C. Correlation Regime ---
        # Average cross-sector correlation (rolling 63-day)
        rolling_corr = pd.Series(0.0, index=dates)
        ret_df = self.sector_returns
        for i in range(len(SECTOR_TICKERS)):
            for j in range(i + 1, len(SECTOR_TICKERS)):
                pair_corr = ret_df.iloc[:, i].rolling(63).corr(ret_df.iloc[:, j])
                rolling_corr += pair_corr
        n_pairs = len(SECTOR_TICKERS) * (len(SECTOR_TICKERS) - 1) / 2
        rolling_corr /= n_pairs
        features["avg_correlation"] = rolling_corr.values
        corr_median = np.nanmedian(rolling_corr.dropna())
        features["correlation_regime"] = np.where(
            rolling_corr > corr_median, 1, 0  # 1=high corr, 0=low corr
        )

        # --- D. Dispersion Regime ---
        # Cross-sectional return standard deviation
        cross_section_std = ret_df.rolling(21).std().mean(axis=1)
        features["cross_section_dispersion"] = cross_section_std.values
        disp_median = np.nanmedian(cross_section_std.dropna())
        features["dispersion_regime"] = np.where(
            cross_section_std > disp_median, 1, 0  # 1=high disp, 0=low disp
        )

        # --- E. Momentum Regime ---
        # Cross-sectional momentum spread (top minus bottom)
        mom_window = 63
        mom = ret_df.rolling(mom_window).sum()
        momentum_spread = mom.max(axis=1) - mom.min(axis=1)
        features["momentum_spread"] = momentum_spread.values
        mom_median = np.nanmedian(momentum_spread.dropna())
        features["momentum_regime"] = np.where(
            momentum_spread > mom_median, 1, 0  # 1=wide, 0=narrow
        )

        # --- F. Macro Level Features (cumulative / smoothed) ---
        for macro_name in MACRO_NAMES:
            features[f"macro_{macro_name}"] = self.macro_data[macro_name].values

        # --- G. True regime labels (for validation) ---
        features["true_regime"] = self.regime_sequence

        self.regime_features = pd.DataFrame(features, index=dates)
        self.regime_features.index.name = "Date"

    # ---- Step 5: All Features Combined ----

    def _compute_all_features(self):
        """
        Combine sector-specific features with regime features.
        Produces a comprehensive feature matrix for analysis.
        """
        dates = self.macro_data.index
        ret_df = self.sector_returns
        features = {}

        # --- Per-sector features ---
        for ticker in SECTOR_TICKERS:
            ret = ret_df[ticker]

            # Rolling momentum (21d, 63d, 126d)
            features[f"{ticker}_mom_21d"] = ret.rolling(21).sum().values
            features[f"{ticker}_mom_63d"] = ret.rolling(63).sum().values
            features[f"{ticker}_mom_126d"] = ret.rolling(126).sum().values

            # Rolling volatility
            features[f"{ticker}_vol_21d"] = ret.rolling(21).std().values * np.sqrt(252)
            features[f"{ticker}_vol_63d"] = ret.rolling(63).std().values * np.sqrt(252)

            # Relative strength vs market
            market_ret = pd.Series(self._market_returns, index=dates)
            rs_21 = ret.rolling(21).sum() - market_ret.rolling(21).sum()
            features[f"{ticker}_rel_strength_21d"] = rs_21.values

            # Rolling beta (vs market)
            cov_rm = ret.rolling(126).cov(market_ret)
            var_m = market_ret.rolling(126).var()
            beta = cov_rm / (var_m + 1e-12)
            features[f"{ticker}_beta_126d"] = beta.values

        # --- Cross-sector features ---
        # Causal density: fraction of significant lagged correlations (proxy)
        # Use absolute lagged correlations as proxy
        lag_corr_sum = pd.Series(0.0, index=dates)
        n_pairs_total = 0
        for i in range(len(SECTOR_TICKERS)):
            for j in range(len(SECTOR_TICKERS)):
                if i != j:
                    lag_corr = ret_df.iloc[:, i].rolling(63).corr(
                        ret_df.iloc[:, j].shift(1)
                    ).abs()
                    lag_corr_sum += lag_corr.fillna(0)
                    n_pairs_total += 1
        features["avg_lagged_correlation"] = (lag_corr_sum / n_pairs_total).values

        # Sector rotation speed: rank correlation of momentum from one month to next
        mom_63 = ret_df.rolling(63).sum()
        rank_current = mom_63.rank(axis=1)
        rank_prev = mom_63.shift(21).rank(axis=1)
        rotation_speed = 1 - rank_current.corrwith(rank_prev, axis=1)
        features["rotation_speed"] = rotation_speed.values

        sector_features = pd.DataFrame(features, index=dates)

        # Combine with regime features
        self.all_features = pd.concat(
            [self.regime_features, sector_features], axis=1
        )
        self.all_features.index.name = "Date"

    # ---- Step 6: Ground Truth ----

    def _build_ground_truth(self):
        """Build ground truth causal relationships per regime."""
        self.ground_truth = {}
        for regime, links in REGIME_CAUSAL_LINKS.items():
            self.ground_truth[regime] = [
                {
                    "leader": leader,
                    "follower": follower,
                    "lag": lag,
                    "strength": strength * self.cfg.lead_lag_strength,
                }
                for leader, follower, lag, strength in links
            ]

    def _ground_truth_df(self) -> pd.DataFrame:
        """Return ground truth as a flat DataFrame."""
        rows = []
        for regime, links in self.ground_truth.items():
            for link in links:
                rows.append({"regime": regime, **link})
        return pd.DataFrame(rows)

    # ---- Utilities ----

    def get_regime_periods(self) -> pd.DataFrame:
        """Return start/end dates for each regime period."""
        dates = self.macro_data.index
        regimes = self.regime_sequence
        regime_names = self.cfg.regime.regime_names

        periods = []
        current_regime = regimes[0]
        start_idx = 0

        for t in range(1, len(regimes)):
            if regimes[t] != current_regime:
                periods.append({
                    "regime": regime_names[current_regime],
                    "start": dates[start_idx],
                    "end": dates[t - 1],
                    "duration_days": t - start_idx,
                })
                current_regime = regimes[t]
                start_idx = t

        # Last period
        periods.append({
            "regime": regime_names[current_regime],
            "start": dates[start_idx],
            "end": dates[-1],
            "duration_days": len(regimes) - start_idx,
        })

        return pd.DataFrame(periods)

    def get_regime_stats(self) -> pd.DataFrame:
        """Return statistics for each regime."""
        regimes = self.regime_sequence
        regime_names = self.cfg.regime.regime_names
        ret = self.sector_returns

        stats = []
        for r_idx, r_name in enumerate(regime_names):
            mask = regimes == r_idx
            n_days = mask.sum()
            frac = n_days / len(regimes)
            r_returns = ret.loc[mask]
            avg_ret = r_returns.mean().mean() * 252
            avg_vol = r_returns.std().mean() * np.sqrt(252)

            # Cross-sector correlation during this regime
            corr_matrix = r_returns.corr()
            n_sectors = len(SECTOR_TICKERS)
            mask_upper = np.triu(np.ones((n_sectors, n_sectors), dtype=bool), k=1)
            avg_corr = corr_matrix.values[mask_upper].mean()

            # Active causal links
            n_links = len(REGIME_CAUSAL_LINKS.get(r_name, []))

            stats.append({
                "regime": r_name,
                "n_days": n_days,
                "fraction": frac,
                "avg_annual_return": avg_ret,
                "avg_annual_vol": avg_vol,
                "avg_cross_corr": avg_corr,
                "n_causal_links": n_links,
            })

        return pd.DataFrame(stats)

    def get_returns_by_regime(self) -> Dict[str, pd.DataFrame]:
        """Split returns by regime for per-regime analysis."""
        regimes = self.regime_sequence
        regime_names = self.cfg.regime.regime_names
        ret = self.sector_returns

        result = {}
        for r_idx, r_name in enumerate(regime_names):
            mask = regimes == r_idx
            result[r_name] = ret.loc[mask]

        return result

    def get_macro_by_regime(self) -> Dict[str, pd.DataFrame]:
        """Split macro data by regime."""
        regimes = self.regime_sequence
        regime_names = self.cfg.regime.regime_names

        result = {}
        for r_idx, r_name in enumerate(regime_names):
            mask = regimes == r_idx
            result[r_name] = self.macro_data.loc[mask]

        return result
