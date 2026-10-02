"""Diag4 remainder 2: temporal PPCA grid (CPU numpy only)."""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

W = np.load("data/raw/windows.npy").astype(np.float64)
sc = np.load("data/raw/scaler.npz")
sp = np.load("data/raw/split.npz")
tr, va = np.asarray(sp["train_idx"]), np.asarray(sp["val_idx"])
px_idx = pd.read_csv("data/raw/nifty50_close.csv", index_col=0, parse_dates=True).index
px_idx = px_idx[px_idx >= "2010-11-08"][1:]
tr_dates = px_idx[[59 + j for j in tr]]
fit_mask = tr_dates < "2017-01-01"
Xfit = ((W[tr][fit_mask].astype(np.float64) - sc["mean"]) / sc["std"])
xv = (W[va].astype(np.float64) - sc["mean"]) / sc["std"]
print("temporal fit: pre-2017 train windows n =", int(fit_mask.sum()))
F = Xfit.reshape(-1, 47)
mu_f = F.mean(0)
Fc = F - mu_f
eig = np.linalg.eigvalsh(np.cov(Fc.T))[::-1]
for K in [4, 8, 16]:
    pca = PCA(n_components=K).fit(Fc)
    B = pca.components_.T
    base = B @ np.diag(pca.explained_variance_) @ B.T
    for s in [0.0, 0.1, 0.3]:
        sig2 = eig[K:].mean()
        C = (1 - s) * (base + sig2 * np.eye(47)) + s * np.trace(base + sig2 * np.eye(47)) / 47 * np.eye(47)
        sign, logdet = np.linalg.slogdet(C)
        Ci = np.linalg.inv(C)
        R = xv.reshape(-1, 47) - mu_f
        nll = float(0.5 * (47 * np.log(2 * np.pi) + logdet + np.einsum("ij,jk,ik->i", R, Ci, R).mean()))
        print(f"K={K} shrink={s}: val NLL={nll:.2f} gain={62.67 - nll:.2f}")
print("DONE-PPCA")
