"""Data validation before training - validates all training artifacts."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch


def validate_training_data(data_dir: Path) -> dict:
    """Validate all training artifacts before training starts.
    
    Returns:
        dict with validation results
    Raises:
        ValueError only for truly critical issues that prevent training
    """
    data_dir = Path(data_dir)
    
    # Load all arrays
    windows = np.load(data_dir / "windows.npy")
    regimes = np.load(data_dir / "window_regimes_v2.npy")
    adj = np.load(data_dir / "adj_matrices.npy")
    node_feats = np.load(data_dir / "window_node_features.npy")
    
    checks = {}
    
    # Shape info (may differ by ±1 due to alignment in VortexDataset)
    n_windows = len(windows)
    n_regimes = len(regimes)
    n_adj = len(adj)
    n_node_feats = len(node_feats)
    
    checks["n_windows"] = n_windows
    checks["n_regimes"] = n_regimes
    checks["n_adj"] = n_adj
    checks["n_node_feats"] = n_node_feats
    checks["window_shape"] = windows.shape
    checks["regimes_shape"] = regimes.shape
    checks["adj_shape"] = adj.shape
    checks["node_feats_shape"] = node_feats.shape
    
    # Shape consistency - allow ±1 difference (VortexDataset aligns with min())
    shape_diff = max(n_windows, n_regimes, n_adj, n_node_feats) - min(n_windows, n_regimes, n_adj, n_node_feats)
    checks["shape_diff"] = shape_diff
    checks["shapes_alignable"] = shape_diff <= 1
    
    # NaN checks - windows/adj/regimes must be clean; node_feats NaN handled by nan_to_num in dataset
    checks["no_nan_windows"] = not np.isnan(windows).any()
    checks["no_nan_adj"] = not np.isnan(adj).any()
    checks["no_nan_regimes"] = not np.isnan(regimes).any()
    checks["nan_in_node_feats"] = np.isnan(node_feats).any()
    
    # Data quality
    checks["regime_balance"] = float(regimes.mean())
    checks["returns_range"] = (float(windows.min()), float(windows.max()))
    checks["returns_mean"] = float(windows.mean())
    checks["returns_std"] = float(windows.std())
    
    # Adjacency matrix properties - diagonals are 0.0 in our data (set by data_processor.py)
    checks["adj_symmetry"] = np.allclose(adj, adj.transpose(0, 2, 1), atol=1e-5)
    checks["adj_diagonal_is_zero"] = np.allclose(
        np.diagonal(adj, axis1=1, axis2=2), 0.0, atol=1e-3
    )
    checks["adj_range"] = (float(adj.min()), float(adj.max()))
    
    # Node features
    checks["node_feats_range"] = (float(np.nanmin(node_feats)), float(np.nanmax(node_feats)))
    checks["node_feats_mean"] = float(np.nanmean(node_feats))
    checks["node_feats_std"] = float(np.nanstd(node_feats))
    
    # Check for dead stocks (zero variance in returns)
    stock_returns = windows  # (N, T, stocks)
    stock_vars = stock_returns.var(axis=(0, 1))  # per-stock variance
    dead_stocks = np.where(stock_vars < 1e-10)[0].tolist()
    checks["dead_stock_indices"] = dead_stocks
    checks["dead_stock_count"] = len(dead_stocks)
    
    # Crisis rate checks
    crisis_rate = regimes.mean()
    checks["crisis_rate"] = crisis_rate
    checks["crisis_rate_reasonable"] = 0.05 <= crisis_rate <= 0.25
    
    # Critical failures - only things that actually prevent training
    critical_failures = []
    if not checks["no_nan_windows"]:
        critical_failures.append("nan_in_windows")
    if not checks["no_nan_adj"]:
        critical_failures.append("nan_in_adj")
    if not checks["no_nan_regimes"]:
        critical_failures.append("nan_in_regimes")
    if not checks["shapes_alignable"]:
        critical_failures.append("shapes_not_alignable")
    if not checks["crisis_rate_reasonable"]:
        critical_failures.append("crisis_rate_unreasonable")
    
    checks["critical_failures"] = critical_failures
    checks["passed"] = len(critical_failures) == 0
    
    if critical_failures:
        raise ValueError(f"Data validation failed: {critical_failures}")
    
    # Warnings for known issues
    warnings = []
    if dead_stocks:
        warnings.append(f"Dead stocks (zero variance): {dead_stocks}")
    if not checks["adj_symmetry"]:
        warnings.append("Adjacency matrices not symmetric")
    if not checks["adj_diagonal_is_zero"]:
        warnings.append("Adjacency diagonals not near 0.0 (expected 0.0 from data_processor)")
    if checks["nan_in_node_feats"]:
        warnings.append("NaN in node_feats (will be nan_to_num'd to 0 in dataset)")
    if shape_diff > 0:
        warnings.append(f"Shape mismatch (max-min={shape_diff}): windows={n_windows}, regimes={n_regimes}, adj={n_adj}, node_feats={n_node_feats} (aligns at load)")
    
    checks["warnings"] = warnings
    
    return checks


def validate_batch(batch: dict[str, torch.Tensor]) -> dict:
    """Validate a single training batch."""
    checks = {}
    
    x = batch["returns"]
    adj = batch["adj_matrix"]
    node_feats = batch["node_features"]
    regimes = batch["regimes"]
    
    checks["batch_size"] = x.shape[0]
    checks["returns_shape"] = tuple(x.shape)
    checks["adj_shape"] = tuple(adj.shape)
    checks["node_feats_shape"] = tuple(node_feats.shape)
    checks["regimes_shape"] = tuple(regimes.shape)
    
    checks["no_nan_returns"] = not torch.isnan(x).any().item()
    checks["no_nan_adj"] = not torch.isnan(adj).any().item()
    checks["no_nan_node_feats"] = not torch.isnan(node_feats).any().item()
    
    checks["returns_range"] = (x.min().item(), x.max().item())
    checks["regime_distribution"] = {
        "crisis": int((regimes == 1).sum().item()),
        "normal": int((regimes == 0).sum().item()),
    }
    
    return checks


if __name__ == "__main__":
    import sys
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw")
    result = validate_training_data(data_dir)
    print(f"Validation {'PASSED' if result['passed'] else 'FAILED'}")
    print(f"Windows: {result['n_windows']}")
    print(f"Crisis rate: {result['crisis_rate']:.3f}")
    print(f"Dead stocks: {result['dead_stock_count']} ({result['dead_stock_indices']})")
    if result["warnings"]:
        print(f"Warnings: {result['warnings']}")