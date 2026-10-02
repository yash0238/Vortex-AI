"""Precompute pooled train correlation per regime (train only)."""
import numpy as np

W = np.load("data/raw/windows.npy").astype(np.float64)
lab = np.load("data/raw/window_regimes_v2.npy")
sp = np.load("data/raw/split.npz")
tr = np.asarray(sp["train_idx"])
out = {}
for rg, nm in [(1, "crisis"), (0, "normal")]:
    X = W[tr][lab[tr] == rg].reshape(-1, W.shape[-1])
    out[nm] = np.corrcoef(X.T)
    print(nm, "windows:", int((lab[tr] == rg).sum()), "mean|offdiag|:",
          round(float(np.abs(out[nm][~np.eye(47, dtype=bool)]).mean()), 4))
np.savez("data/raw/regime_corr.npz", **out)
print("saved data/raw/regime_corr.npz")
