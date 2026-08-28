"""Prepare NIFTY-50 returns, regime labels, windows, and graph targets."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WINDOW_SIZE = 60
CRISIS_PERCENTILE = 85
CORR_THRESHOLD = 0.5
VOL_WINDOW = 20

NIFTY50_TICKERS = [
    "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJAJFINSV.NS", "BAJFINANCE.NS", "BHARTIARTL.NS",
    "BPCL.NS", "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DIVISLAB.NS",
    "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS",
    "HDFCLIFE.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS",
    "ICICIBANK.NS", "INDUSINDBK.NS", "INFY.NS", "ITC.NS", "JSWSTEEL.NS",
    "KOTAKBANK.NS", "LT.NS", "LTI.NS", "M&M.NS", "MARUTI.NS", "NESTLEIND.NS",
    "NTPC.NS", "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS",
    "SBIN.NS", "SHREECEM.NS", "SUNPHARMA.NS", "TATACONSUM.NS", "TATAMOTORS.NS",
    "TATASTEEL.NS", "TCS.NS", "TECHM.NS", "TITAN.NS", "ULTRACEMCO.NS",
    "UPL.NS", "WIPRO.NS",
]


class DataProcessor:
    def __init__(self, tickers: Sequence[str] = NIFTY50_TICKERS,
                 start: str = "2010-01-01", end: str = "2024-12-31",
                 cache_path: Path | None = None) -> None:
        self.tickers = list(tickers)
        self.start, self.end, self.cache_path = start, end, cache_path

    def fetch_data(self) -> pd.DataFrame:
        if self.cache_path and Path(self.cache_path).exists():
            prices = pd.read_csv(self.cache_path, index_col=0, parse_dates=True)
            return prices.loc[self.start:self.end]
        import yfinance as yf

        data = yf.download(self.tickers, start=self.start, end=self.end, progress=False)
        prices = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
        prices.columns = [str(column) for column in prices.columns]
        return prices.dropna(axis=1, how="all")

    @staticmethod
    def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
        prices = prices.apply(pd.to_numeric, errors="coerce").ffill().bfill()
        return np.log(prices / prices.shift(1)).dropna(how="any")

    @staticmethod
    def identify_regimes(returns: pd.DataFrame, vol_window: int = VOL_WINDOW,
                         crisis_percentile: int = CRISIS_PERCENTILE) -> np.ndarray:
        rolling_vol = returns.rolling(vol_window).std().mean(axis=1)
        threshold = np.percentile(rolling_vol.dropna(), crisis_percentile)
        regimes = (rolling_vol > threshold).astype(np.int64).to_numpy()
        regimes[:vol_window - 1] = 0
        return regimes

    @staticmethod
    def create_windows(returns: pd.DataFrame, regimes: np.ndarray,
                       window_size: int = WINDOW_SIZE) -> tuple[np.ndarray, np.ndarray]:
        values = returns.to_numpy(dtype=np.float32)
        if len(values) < window_size:
            raise ValueError("Not enough observations to create a return window")
        windows = np.stack([values[i:i + window_size] for i in range(len(values) - window_size + 1)])
        labels = np.array([regimes[i:i + window_size].max() for i in range(len(windows))], dtype=np.int64)
        return windows, labels

    @staticmethod
    def build_adjacency_from_window(window: np.ndarray,
                                    corr_threshold: float = CORR_THRESHOLD) -> np.ndarray:
        correlation = np.nan_to_num(np.corrcoef(window.T), nan=0.0)
        adjacency = (np.abs(correlation) >= corr_threshold).astype(np.float32)
        np.fill_diagonal(adjacency, 0.0)
        return adjacency

    def run(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        returns = self.compute_log_returns(self.fetch_data())
        regimes = self.identify_regimes(returns)
        windows, window_regimes = self.create_windows(returns, regimes)
        adjacency = np.stack([self.build_adjacency_from_window(window) for window in windows])
        np.save(output_dir / "returns.npy", returns.to_numpy(dtype=np.float32))
        np.save(output_dir / "regimes.npy", regimes)
        np.save(output_dir / "windows.npy", windows)
        np.save(output_dir / "window_regimes.npy", window_regimes)
        np.save(output_dir / "adj_matrices.npy", adjacency)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    DataProcessor(cache_path=root / "data/raw/nifty50_close.csv").run(root / "data/raw")