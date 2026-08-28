"""Validate generated Vortex-AI arrays and their dimensions."""

from pathlib import Path
import numpy as np

RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"
FILES = ["returns.npy", "regimes.npy", "windows.npy", "window_regimes.npy", "adj_matrices.npy"]


def inspect_pipeline() -> bool:
    missing = [name for name in FILES if not (RAW_DIR / name).exists()]
    if missing:
        print(f"Missing generated files: {', '.join(missing)}")
        return False
    returns, regimes = np.load(RAW_DIR / FILES[0]), np.load(RAW_DIR / FILES[1])
    windows, labels = np.load(RAW_DIR / FILES[2]), np.load(RAW_DIR / FILES[3])
    adjacency = np.load(RAW_DIR / FILES[4])
    checks = [
        (returns.shape[0] == regimes.shape[0], "returns/regimes length mismatch"),
        (windows.shape[0] == labels.shape[0] == adjacency.shape[0], "window length mismatch"),
        (windows.shape[1] == 60, "window length must be 60"),
        (windows.shape[2] == adjacency.shape[1] == adjacency.shape[2], "asset dimension mismatch"),
    ]
    for passed, message in checks:
        if not passed:
            print(f"Pipeline check failed: {message}")
            return False
    print("All pipeline checks passed.")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if inspect_pipeline() else 1)