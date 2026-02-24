"""
Generate realistic synthetic data mimicking S&P 500 small-cap equities.
Embeds sector structure, lead-lag relationships, and regime changes.

Since we can't download real data in this environment, this generates
data with properties that match real equity markets:
- Sector factor structure (financials lead, energy lags, etc.)
- Cross-sector lead-lag (e.g., financials -> industrials)
- Intra-sector lead-lag (large -> small within sector)
- Regime changes: some lead-lag relationships are transient
- Fat tails, volatility clustering (GARCH-like)
- Correlation structure similar to real S&P 500 small caps
"""
import numpy as np
import pandas as pd
from typing import List, Tuple


# Sector definitions mimicking real S&P 500 small-cap
SECTORS = {
    "Financials": ["ZION", "CMA", "CFG", "RF", "HBAN", "KEY", "FITB"],
    "Energy":     ["MRO", "DVN", "HAL", "APA", "OXY", "FANG"],
    "Industrials": ["GNRC", "RHI", "CHRW", "JBHT", "DAL", "UAL"],
    "Consumer":   ["HAS", "BWA", "POOL", "RL", "TPR", "PVH"],
    "Healthcare": ["VTRS", "OGN", "CRL", "TECH", "BIO"],
    "Tech":       ["SWKS", "QRVO", "FFIV", "JNPR", "ENPH"],
}

# Known lead-lag relationships to embed (leader, follower, lag_days, strength, active_regime)
# active_regime: (start_frac, end_frac) of data where the relationship is active
EMBEDDED_LEAD_LAG = [
    # Cross-sector: financials tend to lead industrials (credit → real economy)
    ("CFG", "DAL", 2, 0.25, (0.0, 1.0)),     # persistent
    ("KEY", "JBHT", 3, 0.20, (0.0, 1.0)),     # persistent
    # Cross-sector: energy leads materials/industrials
    ("DVN", "CHRW", 1, 0.22, (0.0, 0.6)),     # disappears mid-sample
    ("OXY", "RHI", 2, 0.18, (0.3, 1.0)),      # appears mid-sample
    # Intra-sector: within financials, larger leads smaller
    ("FITB", "ZION", 1, 0.30, (0.0, 1.0)),    # strong, persistent
    ("RF", "CMA", 2, 0.20, (0.0, 1.0)),       # persistent
    # Intra-sector: within tech
    ("FFIV", "QRVO", 1, 0.25, (0.0, 0.7)),    # disappears
    ("JNPR", "SWKS", 3, 0.15, (0.2, 0.8)),    # transient
    # Cross-sector: tech -> consumer (technology adoption)
    ("ENPH", "POOL", 4, 0.18, (0.0, 1.0)),    # persistent
    # Healthcare internal
    ("CRL", "OGN", 2, 0.22, (0.0, 1.0)),      # persistent
]


def generate_sp500_small_synthetic(
    start: str = "2018-01-01",
    end: str = "2024-12-31",
    seed: int = 42,
) -> Tuple[pd.DataFrame, List[dict]]:
    """
    Generate synthetic price data mimicking S&P 500 small-cap stocks.

    Returns:
        prices: DataFrame of close prices
        ground_truth: list of embedded lead-lag relationships for validation
    """
    np.random.seed(seed)

    all_tickers = []
    ticker_sectors = {}
    for sector, tickers in SECTORS.items():
        all_tickers.extend(tickers)
        for t in tickers:
            ticker_sectors[t] = sector

    dates = pd.bdate_range(start=start, end=end, freq="B")
    n = len(dates)
    k = len(all_tickers)

    # ---- 1. Market factor (affects everyone) ----
    market = np.zeros(n)
    market_vol = 0.01
    for t in range(1, n):
        # GARCH(1,1)-like: volatility clustering
        market_vol = 0.0001 + 0.85 * market_vol + 0.10 * market[t-1]**2
        market[t] = np.sqrt(max(market_vol, 0.00005)) * np.random.randn()

    # ---- 2. Sector factors ----
    sector_names = list(SECTORS.keys())
    sector_factors = {}
    for sector in sector_names:
        sf = np.zeros(n)
        sf_vol = 0.008
        for t in range(1, n):
            sf_vol = 0.00005 + 0.80 * sf_vol + 0.12 * sf[t-1]**2
            sf[t] = np.sqrt(max(sf_vol, 0.00003)) * np.random.randn()
        sector_factors[sector] = sf

    # ---- 3. Build returns ----
    returns = np.zeros((n, k))
    for j, ticker in enumerate(all_tickers):
        sector = ticker_sectors[ticker]

        # Market beta: 0.8 to 1.4 for small caps
        market_beta = 0.8 + 0.6 * np.random.rand()
        # Sector loading
        sector_loading = 0.3 + 0.4 * np.random.rand()
        # Idiosyncratic vol: 1.5% to 3% daily
        idio_vol = 0.015 + 0.015 * np.random.rand()

        idio = np.random.randn(n) * idio_vol
        # Add some AR(1) for persistence
        for t in range(1, n):
            idio[t] += 0.03 * idio[t-1]

        returns[:, j] = (
            market_beta * market
            + sector_loading * sector_factors[sector]
            + idio
        )

    # ---- 4. Embed lead-lag relationships ----
    ticker_to_idx = {t: i for i, t in enumerate(all_tickers)}
    ground_truth = []

    for leader, follower, lag, strength, (regime_start, regime_end) in EMBEDDED_LEAD_LAG:
        if leader not in ticker_to_idx or follower not in ticker_to_idx:
            continue
        li = ticker_to_idx[leader]
        fi = ticker_to_idx[follower]

        t_start = int(n * regime_start)
        t_end = int(n * regime_end)

        for t in range(max(lag, t_start), t_end):
            returns[t, fi] += strength * returns[t - lag, li]

        ground_truth.append({
            "leader": leader,
            "follower": follower,
            "lag": lag,
            "strength": strength,
            "regime_start": dates[t_start],
            "regime_end": dates[min(t_end, n-1)],
            "persistent": (regime_start == 0.0 and regime_end == 1.0),
        })

    # ---- 5. Convert to prices ----
    prices_array = 50 + 50 * np.random.rand(k)  # random starting prices $50-$100
    prices_matrix = np.zeros((n, k))
    prices_matrix[0] = prices_array
    for t in range(1, n):
        prices_matrix[t] = prices_matrix[t-1] * np.exp(returns[t])

    prices = pd.DataFrame(prices_matrix, index=dates, columns=all_tickers)
    prices.index.name = "Date"

    return prices, ground_truth


if __name__ == "__main__":
    prices, gt = generate_sp500_small_synthetic()
    print(f"Prices shape: {prices.shape}")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"\nSectors:")
    for sector, tickers in SECTORS.items():
        print(f"  {sector}: {tickers}")
    print(f"\nGround truth lead-lag ({len(gt)} relationships):")
    for g in gt:
        ptype = "persistent" if g["persistent"] else f"regime: {g['regime_start'].date()} to {g['regime_end'].date()}"
        print(f"  {g['leader']} -> {g['follower']} (lag={g['lag']}, str={g['strength']}, {ptype})")
