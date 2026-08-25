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


if __name__ == "__main__":
    inspect_pipeline()