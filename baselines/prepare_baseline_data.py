from __future__ import annotations

import json
import sys
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
OUT_DIR = BASE_DIR / "baselines" / "data"


def _ensure_preprocessed() -> None:
    required = [
        RAW_DATA_DIR / "returns.npy",
        RAW_DATA_DIR / "windows.npy",
        RAW_DATA_DIR / "window_regimes.npy",
    ]
    if all(p.exists() for p in required):
        return

    close_path = RAW_DATA_DIR / "nifty50_close.csv"
    if not close_path.exists():
        raise FileNotFoundError(
            f"Missing {close_path}. Run: python data/download.py"
        )

    # Reuse the project's preprocessing pipeline if arrays are missing.
    sys.path.insert(0, str(BASE_DIR))
    from data.preprocess import preprocess

    preprocess()


def main(window: int = 60) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_preprocessed()

    returns = np.load(RAW_DATA_DIR / "returns.npy")
    windows = np.load(RAW_DATA_DIR / "windows.npy")
    window_regimes = np.load(RAW_DATA_DIR / "window_regimes.npy")

    if windows.shape[1] != window:
        raise ValueError(
            f"Expected window length {window}, found {windows.shape[1]} in windows.npy"
        )

    # TimeGAN baseline uses full multivariate windowed sequences.
    np.save(OUT_DIR / "timegan_windows.npy", windows)
    np.save(OUT_DIR / "timegan_window_regimes.npy", window_regimes)

    # QuantGAN baseline starts with a 1D market series (equal-weight index return).
    market_returns = returns.mean(axis=1).astype(np.float32)
    quant_windows = []
    for i in range(window, len(market_returns)):
        quant_windows.append(market_returns[i - window:i])
    quant_windows = np.asarray(quant_windows, dtype=np.float32)

    np.save(OUT_DIR / "quantgan_market_returns.npy", market_returns)
    np.save(OUT_DIR / "quantgan_market_windows.npy", quant_windows)

    # Convenience CSV for repos that expect tabular input.
    market_df = pd.DataFrame({"market_return": market_returns})
    market_df.to_csv(OUT_DIR / "quantgan_market_returns.csv", index=False)

    metadata = {
        "created_utc": datetime.now(UTC).isoformat(),
        "source_returns_shape": list(returns.shape),
        "timegan_windows_shape": list(windows.shape),
        "timegan_window_regimes_shape": list(window_regimes.shape),
        "quantgan_market_returns_shape": list(market_returns.shape),
        "quantgan_market_windows_shape": list(quant_windows.shape),
        "window": window,
        "stocks": int(returns.shape[1]),
    }
    with open(OUT_DIR / "baseline_data_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Baseline datasets prepared in baselines/data")
    print(f"TimeGAN windows: {windows.shape}")
    print(f"QuantGAN market windows: {quant_windows.shape}")


if __name__ == "__main__":
    main(window=60)
