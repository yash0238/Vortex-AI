"""Kurtosis reference: block bootstrap over ALL 3427 real windows; pooled + median rules."""
import numpy as np
import sys
sys.path.insert(0, ".")
from eval.statistical import _kurtosis_pearson, _per_stock_kurt_median, block_bootstrap_kurt_interval

rng = np.random.default_rng(11)
W = np.load("data/raw/windows.npy").astype(float)
lab = np.load("data/raw/window_regimes_v2.npy")
N, BS, PH = len(W), 100, 30
bounds = list(range(BS, N, BS))
drop = set()
for b in bounds:
    for j in range(max(0, b - PH), min(N, b + PH)):
        drop.add(j)
blocks = [list(range(k * BS, min((k + 1) * BS, N))) for k in range((N + BS - 1) // BS)]
kept = [[j for j in blk if j not in drop] for blk in blocks]
A_idx, B_idx = [], []
for p in range(0, len(blocks) - 1, 2):
    if rng.random() < 0.5:
        A_idx.extend(kept[p])
        B_idx.extend(kept[p + 1])
    else:
        B_idx.extend(kept[p])
        A_idx.extend(kept[p + 1])
if len(blocks) % 2 == 1:
    B_idx.extend(kept[-1])
A_idx, B_idx = np.array(A_idx), np.array(B_idx)
A, B = W[A_idx], W[B_idx]
n = min(500, len(A), len(B))
G = rng.normal(A[:n].mean(axis=(0, 1)), A[:n].std(axis=(0, 1)), size=(n, W.shape[1], W.shape[2]))

lo, hi = block_bootstrap_kurt_interval(W, n_draws=1000, block=100, seed=0, use_median=False)
print(f"POOLED ref interval (all windows): [{lo:.2f}, {hi:.2f}]")
for name, X in [("A", A), ("B", B), ("Gaussian", G)]:
    k = _kurtosis_pearson(X)
    print(f"  pooled kurt {name}: {k:.2f} inside={bool(lo <= k <= hi)}")

mlo, mhi = block_bootstrap_kurt_interval(W, n_draws=1000, block=100, seed=0, use_median=True)
print(f"MEDIAN ref interval (all windows): [{mlo:.2f}, {mhi:.2f}]")
for name, X in [("A", A), ("B", B), ("Gaussian", G)]:
    m = _per_stock_kurt_median(X)
    print(f"  median kurt {name}: {m:.2f} inside={bool(mlo <= m <= mhi)}")
