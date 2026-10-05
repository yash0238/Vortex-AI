from __future__ import annotations

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import numpy as np
from numpy.linalg import lstsq

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"


def compute_pearson_adjacency(window_returns: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """Computes pairwise Pearson correlation matrix and applies edge thresholding."""
    corr = np.corrcoef(window_returns, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    adj = np.where(np.abs(corr) >= threshold, np.abs(corr), 0.0)
    np.fill_diagonal(adj, 0.0)
    return adj


def _granger_causality_fast(x: np.ndarray, y: np.ndarray, maxlag: int = 2) -> float:
    """Test if y Granger-causes x using OLS-based F-test.

    Returns p-value. Much faster than statsmodels.grangercausalitytests
    because it computes only the target lag, not all lags 1..maxlag.
    """
    T = len(x)
    if T < 2 * maxlag + 2:
        return 1.0

    x = x - x.mean()
    y = y - y.mean()

    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 1.0

    x_lags = np.column_stack([x[maxlag - k: T - k] for k in range(1, maxlag + 1)])
    y_lags = np.column_stack([y[maxlag - k: T - k] for k in range(1, maxlag + 1)])

    n_obs = len(x_lags)
    x_target = x[maxlag:]

    X_restricted = np.column_stack([x_lags, np.ones(n_obs)])
    X_unrestricted = np.column_stack([x_lags, y_lags, np.ones(n_obs)])

    try:
        beta_r, res_r, _, _ = lstsq(X_restricted, x_target, rcond=None)
        beta_u, res_u, _, _ = lstsq(X_unrestricted, x_target, rcond=None)
    except Exception:
        return 1.0

    rss_r = res_r[0] if len(res_r) > 0 else np.sum((x_target - X_restricted @ beta_r) ** 2)
    rss_u = res_u[0] if len(res_u) > 0 else np.sum((x_target - X_unrestricted @ beta_u) ** 2)

    if rss_u < 1e-15:
        return 1.0

    df_num = maxlag
    df_den = n_obs - 2 * maxlag - 1
    if df_den <= 0:
        return 1.0

    f_stat = ((rss_r - rss_u) / df_num) / (rss_u / df_den)

    from scipy.stats import f as f_dist
    p_val = 1.0 - f_dist.cdf(f_stat, df_num, df_den)
    return float(p_val)


def compute_granger_adjacency(
    window_returns: np.ndarray,
    maxlag: int = 2,
    p_threshold: float = 0.05,
) -> np.ndarray:
    """Computes directional Granger causality p-values between all stock pairs.

    Uses optimized OLS-based F-test instead of statsmodels full test suite.
    """
    N = window_returns.shape[1]
    granger_adj = np.zeros((N, N), dtype=np.float32)

    stds = np.std(window_returns, axis=0)
    valid_mask = stds > 1e-8

    for i in range(N):
        if not valid_mask[i]:
            continue
        for j in range(N):
            if i == j or not valid_mask[j]:
                continue
            p_val = _granger_causality_fast(
                window_returns[:, i], window_returns[:, j], maxlag=maxlag
            )
            if p_val < p_threshold:
                granger_adj[i, j] = 1.0 - p_val

    return granger_adj


def _process_single_window(args: tuple) -> np.ndarray:
    """Process a single window for parallel execution."""
    window_returns, T, alpha = args
    A_pearson = compute_pearson_adjacency(window_returns, threshold=0.3)
    A_granger = compute_granger_adjacency(window_returns, maxlag=2)
    A_emp = alpha * A_pearson + (1.0 - alpha) * A_granger
    return A_emp


def build_adjacency_matrices(
    T: int = 60,
    alpha: float = 0.5,
    stride: int = 1,
    n_workers: int = 4,
    subset: int | None = None,
):
    """Generates combined target empirical adjacency matrices A_t^emp across sliding windows.

    Uses parallel processing for Granger causality computation.
    3.2: alpha is configurable (was effectively hardcoded via __main__);
    subset=N builds only the first N windows for fast smoke/dev runs.
    Numerics are unchanged for any (T, alpha, stride) triple.
    """
    returns_path = RAW_DATA_DIR / "returns.npy"
    if not returns_path.exists():
        raise FileNotFoundError(f"Missing {returns_path}. Run 'python data/preprocess.py' first.")

    returns = np.load(returns_path)
    N_days, N_stocks = returns.shape
    N_windows = (N_days - T) // stride + 1
    if subset is not None:
        N_windows = min(N_windows, subset)

    print(f"Building target adjacency matrices for {N_windows} windows (T={T}, Stocks={N_stocks}, alpha={alpha})...")
    print(f"Using {n_workers} parallel workers...")

    window_args = [
        (returns[start:start + T].copy(), T, alpha)
        for start in range(0, N_days - T + 1, stride)[:N_windows]
    ]

    adj_matrices = [None] * N_windows
    completed = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_process_single_window, arg): idx
            for idx, arg in enumerate(window_args)
        }
        for future in as_completed(futures):
            idx = futures[future]
            adj_matrices[idx] = future.result()
            completed += 1
            if completed % 500 == 0:
                print(f"Processed {completed} / {N_windows} windows...")

    adj_matrices_arr = np.array(adj_matrices, dtype=np.float32)

    # Never let a subset/dev run overwrite the full artifact.
    output_path = RAW_DATA_DIR / (
        "adj_matrices.npy" if subset is None else f"adj_matrices_subset{subset}.npy"
    )
    np.save(output_path, adj_matrices_arr)

    print(f"Empirical adjacency matrices saved to {output_path} with shape {adj_matrices_arr.shape}")


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Build empirical adjacency targets")
    parser.add_argument("--T", type=int, default=60)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--subset", type=int, default=None,
                        help="Build only first N windows (dev smoke; saved separately)")
    args = parser.parse_args()
    t0 = time.time()
    build_adjacency_matrices(T=args.T, alpha=args.alpha, stride=args.stride,
                             n_workers=args.n_workers, subset=args.subset)
    print(f"Elapsed: {time.time() - t0:.1f}s")
