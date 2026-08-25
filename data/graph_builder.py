from pathlib import Path
import warnings
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"


def compute_pearson_adjacency(window_returns: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """Computes pairwise Pearson correlation matrix and applies edge thresholding."""
    corr = np.corrcoef(window_returns, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)

    # Threshold connections where absolute correlation < 0.3 and remove self-loops
    adj = np.where(np.abs(corr) >= threshold, np.abs(corr), 0.0)
    np.fill_diagonal(adj, 0.0)
    return adj


def compute_granger_adjacency(window_returns: np.ndarray, maxlag: int = 2, p_threshold: float = 0.05) -> np.ndarray:
    """Computes directional Granger causality p-values between all stock pairs."""
    N = window_returns.shape[1]
    granger_adj = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue

            # Check if stock i Granger-causes stock j
            pair_data = window_returns[:, [j, i]]

            if (np.std(pair_data[:, 0]) < 1e-8 or np.std(pair_data[:, 1]) < 1e-8):
                continue
            try:
                res = grangercausalitytests(pair_data, maxlag=maxlag, verbose=False)
                p_val = res[maxlag][0]["ssr_ftest"][1]

                # Map significant p-values to directional edge weights
                if p_val < p_threshold:
                    granger_adj[i, j] = 1.0 - p_val
            except Exception:
                pass

    return granger_adj


def build_adjacency_matrices(T: int = 60, alpha: float = 0.5, stride: int = 1):
    """Generates combined target empirical adjacency matrices A_t^emp across sliding windows."""
    returns_path = RAW_DATA_DIR / "returns.npy"
    if not returns_path.exists():
        raise FileNotFoundError(f"Missing {returns_path}. Run 'python data/preprocess.py' first.")

    returns = np.load(returns_path)
    N_days, N_stocks = returns.shape
    N_windows = (N_days - T) // stride + 1

    print(f"Building target adjacency matrices for {N_windows} windows (T={T}," f" Stocks={N_stocks})...")

    adj_matrices = []

    for idx in range(0, N_days - T + 1, stride):
        window = returns[idx : idx + T]

        # 1. Pearson Correlation Graph (Concurrent spillovers)
        A_pearson = compute_pearson_adjacency(window, threshold=0.3)

        # 2. Granger Causality Graph (Directional lead-lag spillovers)
        A_granger = compute_granger_adjacency(window, maxlag=2)

        # Combine both metrics into empirical target matrix A_emp
        A_emp = alpha * A_pearson + (1.0 - alpha) * A_granger
        adj_matrices.append(A_emp)

        if (len(adj_matrices)) % 500 == 0:
            print(f"Processed {len(adj_matrices)} / {N_windows} windows...")

    adj_matrices = np.array(adj_matrices, dtype=np.float32)

    # (N_windows, 50, 50)
    output_path = RAW_DATA_DIR / "adj_matrices.npy"
    np.save(output_path, adj_matrices)

    print(f"Empirical adjacency matrices saved to {output_path} with shape" f" {adj_matrices.shape}")


if __name__ == "__main__":
    build_adjacency_matrices(T=60, alpha=0.5, stride=1)