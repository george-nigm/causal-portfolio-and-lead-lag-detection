"""
Download S&P 500 small-cap data via Yahoo Finance CSV API.
Falls back to synthetic data if download fails.
"""
import os
import time
import urllib.request
import urllib.error
import numpy as np
import pandas as pd
from io import StringIO


# S&P 500 bottom ~100 by market cap (small caps within the index)
# These are historically interesting for lead-lag because:
# - Smaller, less liquid -> potentially lag behind sector leaders
# - More idiosyncratic risk -> more room for causal signals
SP500_SMALL = [
    # Financials
    "ZION", "CMA", "FHN", "CFG", "RF", "HBAN", "KEY", "FITB", "MTB",
    # Energy
    "MRO", "DVN", "HAL", "APA", "OXY", "FANG",
    # Industrials
    "GNRC", "RHI", "NDSN", "CHRW", "JBHT", "DAL", "UAL", "AAL",
    # Consumer
    "HAS", "BWA", "POOL", "BBWI", "RL", "TPR", "PVH",
    # Healthcare
    "VTRS", "OGN", "CRL", "TECH", "BIO",
    # Tech
    "SEDG", "ENPH", "SWKS", "QRVO", "FFIV", "JNPR",
    # Materials
    "ALB", "MOS", "CF", "FMC", "SEE", "IFF",
    # REITs
    "VNO", "SLG", "BXP", "REG", "KIM",
    # Utilities
    "AES", "NRG", "PNW", "EVRG",
]


def _download_yahoo_csv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download historical prices from Yahoo Finance via CSV API."""
    start_ts = int(pd.Timestamp(start).timestamp())
    end_ts = int(pd.Timestamp(end).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}"
        f"?period1={start_ts}&period2={end_ts}&interval=1d"
        f"&events=history&includeAdjustedClose=true"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read().decode("utf-8")
        df = pd.read_csv(StringIO(data), parse_dates=["Date"], index_col="Date")
        return df
    except Exception as e:
        return pd.DataFrame()


def download_sp500_small(
    start: str = "2015-01-01",
    end: str = "2024-12-31",
    cache_path: str = "cleaned/sp500_small.pkl",
    max_tickers: int = 50,
) -> pd.DataFrame:
    """
    Download S&P 500 small-cap stocks. Uses pickle cache if available.
    Returns DataFrame of adjusted close prices.
    """
    if os.path.exists(cache_path):
        print(f"  Loading cached data from {cache_path}")
        return pd.read_pickle(cache_path)

    tickers = SP500_SMALL[:max_tickers]
    print(f"  Downloading {len(tickers)} tickers from Yahoo Finance...")

    all_prices = {}
    failed = []
    for i, ticker in enumerate(tickers):
        df = _download_yahoo_csv(ticker, start, end)
        if not df.empty and "Adj Close" in df.columns:
            all_prices[ticker] = df["Adj Close"]
            if (i + 1) % 10 == 0:
                print(f"    Downloaded {i + 1}/{len(tickers)}")
        else:
            failed.append(ticker)
        time.sleep(0.3)  # rate limit

    if not all_prices:
        print("  Download failed! No data retrieved.")
        return pd.DataFrame()

    prices = pd.DataFrame(all_prices)
    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.8))  # drop columns with >20% NaN
    prices = prices.ffill().dropna()

    print(f"  Downloaded {prices.shape[1]} tickers, {prices.shape[0]} days")
    if failed:
        print(f"  Failed tickers: {failed}")

    # Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    prices.to_pickle(cache_path)
    print(f"  Cached to {cache_path}")

    return prices


if __name__ == "__main__":
    prices = download_sp500_small()
    print(f"\nResult: {prices.shape}")
    print(prices.head())
