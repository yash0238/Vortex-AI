from pathlib import Path
import numpy as np
import pandas as pd

# Dynamic path resolution pointing to project root
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"


def compute_log_returns(close_df: pd.DataFrame) -> pd.DataFrame:
    """Coerces numeric values, fills missing prices, and calculates log-returns."""
    close_df = close_df.apply(pd.to_numeric, errors="coerce")

    # Forward-fill then backward-fill missing stock prices (holidays/delisted gaps)
    close_df = close_df.ffill().bfill()

    # Compute daily log-returns: ln(S_t / S_t-1)
    returns = np.log(close_df / close_df.shift(1))

    # Fill initial shift NaN with 0.0
    return returns.fillna(0.0)


def label_regimes(
    returns_df: pd.DataFrame, vol_window: int = 20, crisis_pct: float = 85.0) -> pd.Series:
    """Constructs a composite crisis score using realized vol, drawdowns,
    and circuit filter proximity, then labels the top 15% as crisis (1).
    """
    # 1. Realized Volatility: 20-day rolling mean of cross-sectional std dev
    realized_vol = (returns_df.std(axis=1).rolling(vol_window, min_periods=1).mean())

    # 2. Drawdown Fraction: Proportion of stocks down >5% in 5 days
    rolling_5d = returns_df.rolling(5, min_periods=1).sum()
    drawdown_frac = (rolling_5d < -0.05).mean(axis=1)

    # 3. Circuit Proximity: Proportion of stocks moving >9.5% in a single day
    circuit_prox = (returns_df.abs() > 0.095).mean(axis=1)

    # Normalize volatility securely
    vol_q99 = realized_vol.quantile(0.99)
    if pd.isna(vol_q99) or vol_q99 == 0:
        vol_normalized = realized_vol
    else:
        vol_normalized = realized_vol / vol_q99

    # Combine composite crisis score
    crisis_score = (0.5 * vol_normalized + 0.3 * drawdown_frac + 0.2 * circuit_prox)
    crisis_score = crisis_score.fillna(0.0)

    # 85th percentile threshold isolates ~15% highest volatility windows
    threshold = np.percentile(crisis_score, crisis_pct)
    labels = (crisis_score >= threshold).astype(int)

    print(f"Crisis days labeled: {labels.sum()} / {len(labels)}" f" ({labels.mean():.1%})")
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

def preprocess():
    close_path = RAW_DATA_DIR / "nifty50_close.csv"
    if not close_path.exists():
        raise FileNotFoundError(f"Missing {close_path}. Run 'python data/download.py' first.")

    # Load CSV, treating first column as Date index
    close_df = pd.read_csv(close_path, index_col=0, parse_dates=True)

    # Compute log returns and regime labels
    returns_df = compute_log_returns(close_df)
    labels = label_regimes(returns_df, vol_window=20, crisis_pct=85.0)

    # Construct 60-day windowed arrays
    windows, window_labels = make_windows(returns_df, labels, T=60)

    # Save arrays for SDE & GAT model training
    np.save(RAW_DATA_DIR / "returns.npy", returns_df.values)
    np.save(RAW_DATA_DIR / "regimes.npy", labels.values)
    np.save(RAW_DATA_DIR / "windows.npy", windows)
    np.save(RAW_DATA_DIR / "window_regimes.npy", window_labels)

    print(f"Preprocessed returns matrix saved: {returns_df.shape}")
    print(f"Windowed data shape: {windows.shape} (Windows, T, Stocks)")


if __name__ == "__main__":
    preprocess()