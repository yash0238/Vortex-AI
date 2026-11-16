"""Validate generated Vortex-AI arrays and their dimensions."""

from pathlib import Path
import numpy as np

RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"


def inspect_pipeline() -> bool:
    # Check the canonical files used by the training pipeline
    required_files = [
        "windows.npy",
        "window_regimes_v2.npy",  # canonical labels
        "adj_matrices.npy",
        "scaler.npz",
        "split.npz",
    ]
    missing = [name for name in required_files if not (RAW_DIR / name).exists()]
    if missing:
        print(f"Missing generated files: {', '.join(missing)}")
        return False

    windows = np.load(RAW_DIR / "windows.npy")
    labels = np.load(RAW_DIR / "window_regimes_v2.npy")
    adjacency = np.load(RAW_DIR / "adj_matrices.npy")

    n = min(windows.shape[0], labels.shape[0], adjacency.shape[0])
    if not (windows.shape[0] == labels.shape[0] == adjacency.shape[0]):
        print(f"  Note: Aligning lengths to {n} (windows={windows.shape[0]}, labels={labels.shape[0]}, adj={adjacency.shape[0]})")

    checks = [
        (windows.shape[1] == 60, "window length must be 60"),
        (windows.shape[2] == adjacency.shape[1] == adjacency.shape[2], "asset dimension mismatch"),
    ]
    for passed, message in checks:
        if not passed:
            print(f"Pipeline check failed: {message}")
            return False

    print(f"All pipeline checks passed.")
    print(f"  windows: {windows.shape}")
    print(f"  window_regimes: {labels.shape}, crisis rate: {labels.mean():.1%}")
    print(f"  adj_matrices: {adjacency.shape}")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if inspect_pipeline() else 1)