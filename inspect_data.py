from pathlib import Path
import numpy as np

RAW_DATA_DIR = Path(__file__).resolve().parent / "data" / "raw"

FILES_TO_CHECK = [
    "returns.npy",
    "regimes.npy",
    "windows.npy",
    "window_regimes.npy",
    "adj_matrices.npy",
]


def inspect_pipeline():
    print("VORTEX-AI PIPELINE PROGRESS & DATA INTEGRITY CHECK")

    if not RAW_DATA_DIR.exists():
        print(f"Directory not found: {RAW_DATA_DIR}")
        return

    missing = [filename for filename in FILES_TO_CHECK if not (RAW_DATA_DIR / filename).exists()]
    if missing:
        print(f"Missing required arrays: {', '.join(missing)}")
        return False

    for filename in FILES_TO_CHECK:
        filepath = RAW_DATA_DIR / filename
        if not filepath.exists():
            print(f"XX [MISSING] XX {filename}")
            continue

        data = np.load(filepath)
        print(f"✅ [EXISTS]   {filename:<20}")
        print(f"   ├─ Shape : {data.shape}")
        print(f"   ├─ Type  : {data.dtype}")

        # Specific diagnostics per file type
        if "regime" in filename:
            crisis_pct = np.mean(data) * 100
            print(f"   └─ Stats : {data.sum()} Crisis Days ({crisis_pct:.2f}%)")
        else:
            print(f"   └─ Stats : Min={data.min():.4f}, Max={data.max():.4f}," f" Mean={data.mean():.4f}")

    returns = np.load(RAW_DATA_DIR / "returns.npy")
    regimes = np.load(RAW_DATA_DIR / "regimes.npy")
    windows = np.load(RAW_DATA_DIR / "windows.npy")
    window_regimes = np.load(RAW_DATA_DIR / "window_regimes.npy")
    adjacencies = np.load(RAW_DATA_DIR / "adj_matrices.npy")
    assert returns.shape[0] == regimes.shape[0], "returns and regimes length mismatch"
    assert windows.shape[0] == window_regimes.shape[0] == adjacencies.shape[0], "window artifact length mismatch"
    assert windows.shape[2] == adjacencies.shape[1] == adjacencies.shape[2], "asset dimension mismatch"
    print("All pipeline checks passed.")
    return True


if __name__ == "__main__":
    inspect_pipeline()