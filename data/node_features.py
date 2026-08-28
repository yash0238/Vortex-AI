"""Engineer 6-dimensional node features per stock per window per masterplan.

Feature vector for stock i at time t:
    [r_i,t, |r_i,t|, vol_i,t, sector_onehot, dist_to_band]

Sector mapping follows NIFTY-50 sector classification.
Distance-to-band measures how close the stock is to its circuit filter limit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

NIFTY50_SECTORS: dict[str, int] = {
    "RELIANCE.NS": 0, "TCS.NS": 1, "HDFCBANK.NS": 2, "INFY.NS": 1,
    "HINDUNILVR.NS": 3, "ICICIBANK.NS": 2, "KOTAKBANK.NS": 2, "BHARTIARTL.NS": 4,
    "ITC.NS": 3, "AXISBANK.NS": 2, "SBIN.NS": 2, "LT.NS": 5, "BAJFINANCE.NS": 6,
    "HCLTECH.NS": 1, "ASIANPAINT.NS": 7, "MARUTI.NS": 8, "SUNPHARMA.NS": 9,
    "TITAN.NS": 7, "ULTRACEMCO.NS": 10, "NESTLEIND.NS": 3, "WIPRO.NS": 1,
    "POWERGRID.NS": 11, "NTPC.NS": 11, "M&M.NS": 8, "TECHM.NS": 1,
    "TATAMOTORS.NS": 8, "TATASTEEL.NS": 12, "JSWSTEEL.NS": 12, "BAJAJ-AUTO.NS": 8,
    "CIPLA.NS": 9, "DRREDDY.NS": 9, "DIVISLAB.NS": 9, "HEROMOTOCO.NS": 8,
    "ONGC.NS": 13, "COALINDIA.NS": 14, "BPCL.NS": 13, "GRASIM.NS": 10,
    "ADANIPORTS.NS": 5, "EICHERMOT.NS": 8, "APOLLOHOSP.NS": 15, "HINDALCO.NS": 16,
    "TATACONSUM.NS": 3, "BRITANNIA.NS": 3, "SHREECEM.NS": 10, "UPL.NS": 17,
    "BAJAJFINSV.NS": 6, "SBILIFE.NS": 6, "HDFCLIFE.NS": 6, "INDUSINDBK.NS": 2,
    "LTI.NS": 1,
}

NUM_SECTORS = 18

BAND_A_STOCKS = {"SBIN.NS", "TATAMOTORS.NS", "TATASTEEL.NS"}
DEFAULT_BAND = 0.10


def _band_limit(ticker: str) -> float:
    if ticker in BAND_A_STOCKS:
        return 0.05
    return DEFAULT_BAND


def load_ticker_list() -> list[str]:
    close_path = RAW_DATA_DIR / "nifty50_close.csv"
    if close_path.exists():
        df = pd.read_csv(close_path, index_col=0, nrows=0)
        return [c for c in df.columns if c in NIFTY50_SECTORS]
    return [t for t in NIFTY50_SECTORS]


def compute_node_features(
    returns: np.ndarray,
    volume: np.ndarray | None = None,
    tickers: Sequence[str] | None = None,
) -> np.ndarray:
    """Compute 6-dim node features from returns and optional volume.

    Args:
        returns: (T, N) array of log-returns.
        volume: (T, N) array of traded volume (z-normalized internally).
        tickers: list of ticker strings for band/sector lookup.

    Returns:
        features: (T, N, 6) array — [r, |r|, vol, sector_onehot(18), dist_to_band].
        Note: sector_onehot expands the last 3 dims, so output is (T, N, 23).
        For strict 6-dim, use compute_node_features_compact which averages sector.
    """
    T, N = returns.shape
    if tickers is None:
        tickers = [f"STOCK_{i}" for i in range(N)]

    feat = np.zeros((T, N, 3 + NUM_SECTORS + 1), dtype=np.float32)

    feat[:, :, 0] = returns
    feat[:, :, 1] = np.abs(returns)

    if volume is not None:
        vol_clean = np.nan_to_num(volume, nan=0.0)
        vol_mean = vol_clean.mean(axis=0, keepdims=True)
        vol_std = vol_clean.std(axis=0, keepdims=True) + 1e-8
        feat[:, :, 2] = (vol_clean - vol_mean) / vol_std
    else:
        feat[:, :, 2] = 0.0

    for j, tk in enumerate(tickers):
        sector_idx = NIFTY50_SECTORS.get(tk, 0)
        feat[:, j, 3 + sector_idx] = 1.0
        band = _band_limit(tk)
        feat[:, j, 3 + NUM_SECTORS] = band - np.abs(returns[:, j])

    return feat


def compute_node_features_compact(
    returns: np.ndarray,
    volume: np.ndarray | None = None,
    tickers: Sequence[str] | None = None,
) -> np.ndarray:
    """Compute strict 6-dim node features per masterplan specification.

    Features: [r_i,t, |r_i,t|, vol_i,t, sector_id (normalized), dist_to_band]
    Sector is encoded as a single normalized scalar (sector_id / NUM_SECTORS).
    """
    T, N = returns.shape
    if tickers is None:
        tickers = [f"STOCK_{i}" for i in range(N)]

    feat = np.zeros((T, N, 6), dtype=np.float32)

    feat[:, :, 0] = returns
    feat[:, :, 1] = np.abs(returns)

    if volume is not None:
        vol_clean = np.nan_to_num(volume, nan=0.0)
        vol_mean = vol_clean.mean(axis=0, keepdims=True)
        vol_std = vol_clean.std(axis=0, keepdims=True) + 1e-8
        feat[:, :, 2] = (vol_clean - vol_mean) / vol_std
    else:
        feat[:, :, 2] = 0.0

    for j, tk in enumerate(tickers):
        sector_idx = NIFTY50_SECTORS.get(tk, 0)
        feat[:, j, 3] = sector_idx / NUM_SECTORS
        band = _band_limit(tk)
        feat[:, j, 4] = band
        feat[:, j, 5] = band - np.abs(returns[:, j])

    return feat


def build_window_node_features(
    returns: np.ndarray,
    volume: np.ndarray | None = None,
    tickers: Sequence[str] | None = None,
    window_size: int = 60,
    stride: int = 1,
) -> np.ndarray:
    """Build node features for every sliding window.

    Returns:
        window_features: (N_windows, T, N, 6)
    """
    T_full, N = returns.shape
    features_full = compute_node_features_compact(returns, volume, tickers)

    n_windows = (T_full - window_size) // stride + 1
    out = np.zeros((n_windows, window_size, N, 6), dtype=np.float32)

    for idx, start in enumerate(range(0, T_full - window_size + 1, stride)):
        out[idx] = features_full[start:start + window_size]

    return out


def regenerate_window_regimes() -> np.ndarray:
    """Regenerate window-level regime labels using last-day labeling.

    The existing window_regimes.npy uses .max() over the window (73% crisis).
    The masterplan requires ~15% crisis rate, achieved by labeling each
    window with its final day's regime.
    """
    returns = np.load(RAW_DATA_DIR / "returns.npy")
    regimes = np.load(RAW_DATA_DIR / "regimes.npy")
    T = returns.shape[0]
    window_size = 60

    window_labels = []
    for i in range(window_size, T):
        window_labels.append(regimes[i - 1])

    return np.array(window_labels, dtype=np.int64)


if __name__ == "__main__":
    returns = np.load(RAW_DATA_DIR / "returns.npy")
    volume_path = RAW_DATA_DIR / "nifty50_volume.csv"
    volume = None
    if volume_path.exists():
        vol_df = pd.read_csv(volume_path, index_col=0, parse_dates=True)
        volume = vol_df.values.astype(np.float32)

    tickers = load_ticker_list()
    print(f"Tickers loaded: {len(tickers)}")

    wf = build_window_node_features(returns, volume, tickers, window_size=60)
    print(f"Window node features shape: {wf.shape}")

    new_regimes = regenerate_window_regimes()
    print(f"Regenerated window regimes: {new_regimes.shape}")
    print(f"Crisis rate: {new_regimes.mean():.1%}")

    out_path = RAW_DATA_DIR / "window_node_features.npy"
    np.save(out_path, wf)
    print(f"Saved to {out_path}")

    regimes_path = RAW_DATA_DIR / "window_regimes_v2.npy"
    np.save(regimes_path, new_regimes)
    print(f"Saved corrected regimes to {regimes_path}")
