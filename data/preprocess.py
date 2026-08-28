from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"


def compute_log_returns(close_df: pd.DataFrame) -> pd.DataFrame:
    """Coerces numeric values, fills missing prices, and calculates log-returns."""
    close_df = close_df.apply(pd.to_numeric, errors="coerce")
    close_df = close_df.ffill().bfill()
    returns = np.log(close_df / close_df.shift(1))
    return returns.fillna(0.0)


def load_volume() -> pd.DataFrame | None:
    """Load and clean volume data from CSV."""
    vol_path = RAW_DATA_DIR / "nifty50_volume.csv"
    if not vol_path.exists():
        return None
    vol_df = pd.read_csv(vol_path, index_col=0, parse_dates=True)
    vol_df = vol_df.apply(pd.to_numeric, errors="coerce")
    vol_df = vol_df.ffill().bfill()
    vol_df = vol_df.fillna(0.0)
    return vol_df


def label_regimes(
    returns_df: pd.DataFrame, vol_window: int = 20, crisis_pct: float = 85.0) -> pd.Series:
    """Constructs a composite crisis score using realized vol, drawdowns,
    and circuit filter proximity, then labels the top 15% as crisis (1).
    """
    realized_vol = (returns_df.std(axis=1).rolling(vol_window, min_periods=1).mean())
    rolling_5d = returns_df.rolling(5, min_periods=1).sum()
    drawdown_frac = (rolling_5d < -0.05).mean(axis=1)
    circuit_prox = (returns_df.abs() > 0.095).mean(axis=1)

    vol_q99 = realized_vol.quantile(0.99)
    if pd.isna(vol_q99) or vol_q99 == 0:
        vol_normalized = realized_vol
    else:
        vol_normalized = realized_vol / vol_q99

    crisis_score = (0.5 * vol_normalized + 0.3 * drawdown_frac + 0.2 * circuit_prox)
    crisis_score = crisis_score.fillna(0.0)
    threshold = np.percentile(crisis_score, crisis_pct)
    labels = (crisis_score >= threshold).astype(int)

    print(f"Crisis days labeled: {labels.sum()} / {len(labels)} ({labels.mean():.1%})")
    return labels


def make_windows(returns_df: pd.DataFrame, labels: pd.Series, T: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Slices returns into sliding windows of length T (60 days) paired with
    the regime label corresponding to the window's final day.
    """
    windows, window_labels = [], []
    arr = returns_df.values
    lab = labels.values

    for i in range(T, len(arr)):
        windows.append(arr[i - T : i])
        window_labels.append(lab[i - 1])

    return np.array(windows, dtype=np.float32), np.array(window_labels, dtype=np.int64)


def make_volume_windows(volume_df: pd.DataFrame, T: int = 60) -> np.ndarray:
    """Slices volume into sliding windows of length T (60 days).

    Returns:
        volume_windows: (N_windows, T, N_stocks)
    """
    vol = volume_df.values
    windows = []
    for i in range(T, len(vol)):
        windows.append(vol[i - T : i])
    return np.array(windows, dtype=np.float32)


def preprocess():
    close_path = RAW_DATA_DIR / "nifty50_close.csv"
    if not close_path.exists():
        raise FileNotFoundError(f"Missing {close_path}. Run 'python data/download.py' first.")

    close_df = pd.read_csv(close_path, index_col=0, parse_dates=True)

    returns_df = compute_log_returns(close_df)
    labels = label_regimes(returns_df, vol_window=20, crisis_pct=85.0)

    windows, window_labels = make_windows(returns_df, labels, T=60)

    np.save(RAW_DATA_DIR / "returns.npy", returns_df.values)
    np.save(RAW_DATA_DIR / "regimes.npy", labels.values)
    np.save(RAW_DATA_DIR / "windows.npy", windows)
    np.save(RAW_DATA_DIR / "window_regimes.npy", window_labels)

    print(f"Preprocessed returns matrix saved: {returns_df.shape}")
    print(f"Windowed data shape: {windows.shape} (Windows, T, Stocks)")

    volume_df = load_volume()
    if volume_df is not None:
        volume_windows = make_volume_windows(volume_df, T=60)
        np.save(RAW_DATA_DIR / "volume_windows.npy", volume_windows)
        print(f"Volume windows saved: {volume_windows.shape}")
    else:
        print("No volume data found, skipping volume windows.")


if __name__ == "__main__":
    preprocess()
