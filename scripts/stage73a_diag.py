"""7.3a diagnostic: KL curve, real vol ratios, posterior vs prior sigma, tails, NLL baselines, disc."""
import glob
import numpy as np
import pandas as pd
import torch
from scipy import stats
import sys
sys.path.insert(0, ".")
from training.vortex_model import VORTEXModel

DEV = "cuda"
CKPT = "models/checkpoints/vortex-epoch=05-val/total=62.6602.ckpt"
torch.manual_seed(7)
np.random.seed(7)
model = VORTEXModel.load_from_checkpoint(CKPT, strict=False).to(DEV).eval()

# 1. KL per epoch from log + ACF(r2) from eval json
cands = sorted(glob.glob("runs/*stage73a*/version_0/metrics.csv"))
f = cands[-1]
print("log:", f)
df = pd.read_csv(f).groupby("epoch").last()
print("cols:", [c for c in df.columns if "kl" in c])
print("1. KL train per epoch (val KL not logged for hetero runs; computed post-hoc below):")
print(df["train/kl_epoch"].to_string())
import json
ev = json.load(open("results/stage73a_eval.json"))
print("ACF(r2): gen", round(ev["styl"]["acf_sq_gen"], 4), "real", round(ev["styl"]["acf_sq_real"], 4))
pm, plv = model.prior_mu.detach(), model.prior_logvar.detach()
sep = float((pm[1] - pm[0]).abs().mean() / torch.exp(0.5 * plv.mean()))
print(f"prior |mu_c-mu_n|/sqrt(mean var): {sep:.4f}")
print(f"prior mean logvar: crisis={plv[1].mean():.4f} normal={plv[0].mean():.4f}")

W = np.load("data/raw/windows.npy").astype(np.float32)
lab = np.load("data/raw/window_regimes_v2.npy")
A = np.load("data/raw/adj_matrices.npy").astype(np.float32)
NF = np.load("data/raw/window_node_features.npy").astype(np.float32)
sp = np.load("data/raw/split.npz")
va, te = np.asarray(sp["val_idx"]), np.asarray(sp["test_idx"])
sc = np.load("data/raw/scaler.npz")


def to_dev(a):
    return torch.nan_to_num(torch.from_numpy(a), nan=0.0, posinf=1.0, neginf=0.0).to(DEV)


# 2. real crisis/normal per-stock std ratio
def vol_ratio(X, l, tag):
    c, n = X[l == 1], X[l == 0]
    r = (c.std(axis=(0, 1)) / (n.std(axis=(0, 1)) + 1e-12))
    print(f"2. {tag}: n_crisis={int((l == 1).sum())} mean ratio={r.mean():.3f} min={r.min():.3f} max={r.max():.3f}")
    return r


vol_ratio(W, lab, "all windows")
vol_ratio(W[np.concatenate([va, te])], lab[np.concatenate([va, te])], "val+test")

# 3. posterior sigma per regime + correlation with realised std
with torch.no_grad():
    xa = to_dev(((W[va].astype(np.float64) - sc["mean"]) / sc["std"]).astype(np.float32))
    rh, _, _ = model(xa, to_dev(A[va]), to_dev(NF[va].mean(axis=1)))
    zs = model._last_zs
    sig = model.sde_model.emission_sigma(zs).cpu().numpy()  # (B,T,N) z-space
    m = model.sde_model.emi_m(zs).cpu().numpy()
    msig = sig.mean(axis=(1, 2))
    lv = lab[va]
    with torch.no_grad():
        mu_q, lv_q, _ = model.sde_model.encode_stats(xa)
        rr = torch.from_numpy(lv).to(DEV).long().clamp(0, 1)
        klv = model.gaussian_kl(mu_q, lv_q, model.prior_mu[rr], model.prior_logvar[rr])
    print(f"   post-hoc val KL: {float(klv):.4f}")
    print(f"3. posterior sigma: crisis={msig[lv == 1].mean():.3f} normal={msig[lv == 0].mean():.3f} "
          f"ratio={msig[lv == 1].mean() / msig[lv == 0].mean():.3f} (prior was 1.047v1.035=1.01x)")
    real_std = W[va].std(axis=(1, 2))
    print(f"   corr(post sigma, realised std)={np.corrcoef(msig, real_std)[0, 1]:.4f}")
    # 4. tail attribution (z-space posterior)
    xaz = xa.cpu().numpy()
    resid = (xaz - 0.5 * np.tanh(m)) / (sig + 1e-8)
    from eval.statistical import _per_stock_kurt_median

    print(f"4. median kurt: real={_per_stock_kurt_median(xaz):.2f} "
          f"resid={_per_stock_kurt_median(resid):.2f} gen=3.41(reported)")
    resid_flat = resid.ravel()
    print(f"   resid pooled kurt={stats.kurtosis(resid_flat, fisher=False):.2f}")

# 5. NLL baselines on val, SAME units as model (nats per window-day, summed over 47 stocks)
xv = ((W[va].astype(np.float64) - sc["mean"]) / sc["std"])
N_ST = int(sc["std"].shape[0])
nll_unit = float((0.5 * (np.log(2 * np.pi) + xv ** 2)).mean() * N_ST)
print(f"5. unit N(0,1) NLL: {nll_unit:.3f} (model val 62.66)")
tr = np.asarray(sp["train_idx"])
xtr = (W[tr].astype(np.float64) - sc["mean"]) / sc["std"]
m0, s0 = xtr.mean(axis=(0, 1)), xtr.std(axis=(0, 1)) + 1e-8
nll_g = float((0.5 * (np.log(2 * np.pi * s0 ** 2) + ((xv - m0) / s0) ** 2)).mean() * N_ST)
print(f"   per-stock Gaussian(train-fit) NLL: {nll_g:.3f}")
best = np.full(xtr.shape[2], 1e9)
for df_ in [3, 4, 5, 6, 8, 10, 15, 30]:
    ll = stats.t.logpdf((xv - m0) / s0, df_) - np.log(s0) + 0.5 * np.log(df_ / (df_ - 2))
    best = np.minimum(best, -ll.mean(axis=(0, 1)))
print(f"   per-stock Student-t(grid df, var-matched) NLL: {float(best.mean() * N_ST):.3f}")

# 6. disc forensics: importances, mean/std diffs, Gaussian under same 500/regime protocol
from sklearn.ensemble import GradientBoostingClassifier
nrm = np.where(lab[te] == 0)[0]
cri = np.where(lab[te] == 1)[0]
r500 = np.concatenate([np.tile(nrm, 500 // len(nrm) + 1)[:500], np.tile(cri, 500 // len(cri) + 1)[:500]])
Xr, lr500 = W[te][r500].reshape(1000, -1), lab[te][r500]
mu500, sd500 = Xr.mean(0), Xr.std(0) + 1e-12
Gg = np.random.default_rng(5).normal(mu500, sd500, size=Xr.shape)
X = np.concatenate([Xr, Gg])
y = np.array([1] * 1000 + [0] * 1000)
clf = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X, y)
imp = clf.feature_importances_.reshape(60, 47)
top = np.dstack(np.unravel_index(np.argsort(imp.ravel())[-5:], imp.shape))[0][::-1]
print("6. Gaussian-500/regime disc train-acc:", round(float(clf.score(X, y)), 4), "(optimistic, same data)")
print("   top-5 (day, stock):", [(int(t), int(s)) for t, s in top])
with torch.no_grad():
    g500m = model.sample_prior_scenarios(to_dev(A[te][r500]), to_dev(NF[te][r500].mean(axis=1)),
                                         torch.from_numpy(lr500).to(DEV)).cpu().numpy()
G73 = g500m * sc["std"] + sc["mean"]
print("   7.3a gen vs real per-stock mean maxabsdiff:",
      round(float(np.abs(G73.mean(axis=(0, 1)) - Xr.mean(axis=(0, 1))).max()), 5))
print("   7.3a gen vs real per-stock std maxabsdiff:",
      round(float(np.abs(G73.std(axis=(0, 1)) - Xr.std(axis=(0, 1))).max()), 5))
print("DONE")
