"""7.3b quick checks: conditioning audit, held-out V1/V3, corr spectrum, static t-PPCA, skew."""
import numpy as np
import torch
from scipy import stats
import sys
sys.path.insert(0, ".")
from training.vortex_model import VORTEXModel
from eval.discriminative import discriminative_score
from eval.contagion import evaluate_contagion, cvar_regime_ratio
from eval.statistical import evaluate_stylized_facts

DEV = "cuda"
CKPT = "models/checkpoints/vortex-epoch=08-val/total=61.1070.ckpt"
torch.manual_seed(7)
np.random.seed(7)
model = VORTEXModel.load_from_checkpoint(CKPT, strict=False).to(DEV).eval()
W = np.load("data/raw/windows.npy").astype(np.float64)
lab = np.load("data/raw/window_regimes_v2.npy")
A = np.load("data/raw/adj_matrices.npy").astype(np.float32)
NF = np.load("data/raw/window_node_features.npy").astype(np.float32)
sp = np.load("data/raw/split.npz")
tr, va, te = (np.asarray(sp[k]) for k in ["train_idx", "val_idx", "test_idx"])
sc = np.load("data/raw/scaler.npz")
tickers = list(__import__("pandas").read_csv("data/raw/nifty50_close.csv", index_col=0, nrows=0).columns)
print("a. diag4 V1-V4 grows = SAME-REGIME TRAIN rows (grows drawn from tr pools); "
      "7.3a-t prior numbers used VAL rows. gen/real column = mean over regimes of "
      "(gen per-stock std / same-regime real per-stock std).")


def to_dev(a):
    return torch.nan_to_num(torch.from_numpy(a), nan=0.0, posinf=1.0, neginf=0.0).to(DEV)


with torch.no_grad():
    Xs = ((W.astype(np.float64) - sc["mean"]) / sc["std"]).astype(np.float32)
    mu_all, lv_all, _ = model.sde_model.encode_stats(to_dev(Xs))
    MU, LV = mu_all.cpu().numpy(), lv_all.cpu().numpy()


def run_variant(z0, grows, tag):
    with torch.no_grad():
        ne, _ = model.gat(to_dev(NF[grows].mean(axis=1)), to_dev(A[grows]))
        _, zp = model.sde_model.decode_from_z0(ne.mean(dim=1), torch.from_numpy(z0.astype(np.float32)).to(DEV))
        sig = model.sde_model.emission_sigma(zp).cpu().numpy()
        gen = model.sde_model.emission_sample(zp).cpu().numpy() * sc["std"] + sc["mean"]
    return sig, gen


rng = np.random.default_rng(7)
vt = np.concatenate([va, te])
for tag, pool_idx in [("val+test", vt)]:
    r = lab[pool_idx]
    grows_c = rng.choice(pool_idx[r == 1], 500, replace=True)
    grows_n = rng.choice(pool_idx[r == 0], 500, replace=False)
    grows = np.concatenate([grows_c, grows_n])
    rr = np.array([1] * 500 + [0] * 500)
    mu_p = {1: MU[lab == 1], 0: MU[lab == 0]}
    pm, plv = model.prior_mu.detach().cpu().numpy(), model.prior_logvar.detach().cpu().numpy()
    z1 = np.concatenate([rng.normal(pm[1], np.exp(0.5 * plv[1]), size=(500, 64)),
                         rng.normal(pm[0], np.exp(0.5 * plv[0]), size=(500, 64))])
    s1, g1 = run_variant(z1, grows, "v1")
    idx = {rg: np.where(lab == rg)[0] for rg in (0, 1)}
    pk = {rg: rng.integers(0, len(mu_p[rg]), 500) for rg in (0, 1)}
    lv_p = {rg: LV[lab == rg] for rg in (0, 1)}
    z3 = np.concatenate([(mu_p[rg][pk[rg]] + rng.normal(0, 1, size=(500, 64)) *
                         np.exp(0.5 * lv_p[rg][pk[rg]])) for rg in (1, 0)])
    s3, g3 = run_variant(z3, grows, "v3")
    for nm, (s, g) in [("V1-heldout", (s1, g1)), ("V3-heldout", (s3, g3))]:
        ms = s.mean(axis=(1, 2))
        print(f"b. {nm}: emission-sigma ratio={ms[rr == 1].mean() / ms[rr == 0].mean():.3f}")
    R = W[pool_idx]
    print("   real val+test std contrast (1.17 ref):",
          round(float((R[lab[pool_idx] == 1].std(axis=(0, 1)) / (R[lab[pool_idx] == 0].std(axis=(0, 1)) + 1e-12)).mean()), 3))

# c. train correlation spectrum
Ctr = np.corrcoef(W[tr].reshape(-1, 47).T)
off = Ctr[~np.eye(47, dtype=bool)]
print("c. train corr: top5", np.sort(off)[-5:].round(3).tolist(), "bottom5", np.sort(off)[:5].round(3).tolist())
pairs = [(Ctr[i, j], tickers[i], tickers[j]) for i in range(47) for j in range(i + 1, 47)]
pairs.sort(reverse=True)
print("   top pairs:", [(t1, t2, round(float(v), 3)) for v, t1, t2 in pairs[:5]])
eig = np.linalg.eigvalsh(Ctr)
print("   smallest eigenvalue:", round(float(eig[0]), 4), "(STOP if <0.02); max pair:",
      round(pairs[0][0], 4), "(STOP if >0.95)")
print("   per-stock val var min/max:", round(float(W[va].var(axis=(0, 1)).min()), 6),
      round(float(W[va].var(axis=(0, 1)).max()), 6))

# d. static rank-8 PPCA + shared-mix t, 500 scenarios
from sklearn.decomposition import PCA
xtr = (W[tr].astype(np.float64) - sc["mean"]) / sc["std"]
F = xtr.reshape(-1, 47)
mu_f = F.mean(0)
pca = PCA(n_components=8).fit(F - mu_f)
B = pca.components_.T
sig2 = (np.linalg.eigvalsh(np.cov((F - mu_f).T))[::-1][8:]).mean()
L = np.linalg.cholesky(B @ np.diag(pca.explained_variance_) @ B.T + sig2 * np.eye(47))
nu = 7.6
w = rng.chisquare(nu, size=30000)
z = rng.normal(0, 1, size=(30000, 47))
S = (mu_f + np.sqrt((nu - 2) / nu) * (z @ L.T) / np.sqrt(w / nu)[:, None])
Sraw = S * sc["std"] + sc["mean"]
Swin = Sraw.reshape(500, 60, 47)
lab500 = np.array([1] * 250 + [0] * 250)
sq = Swin ** 2
vv = sq - sq.mean(axis=1, keepdims=True)
print("d. static t-PPCA500: acf=",
      round(float(((vv[:, :-1] * vv[:, 1:]).mean()) / (sq.var() + 1e-12)), 4))
Creal = np.corrcoef(W[te][:500].reshape(-1, 47).T)
Cgen = np.corrcoef(Sraw.T)
mask = ~np.eye(47, dtype=bool)
print("   styl(corr-focused): fro=", round(float(np.linalg.norm(Creal - Cgen)), 2),
      "mae=", round(float(np.abs(Creal[mask] - Cgen[mask]).mean()), 4))
dd = discriminative_score(W[te][:500], Swin, n_samples=500)
print("   disc:", round(dd["discriminative_score"], 4))
cc = evaluate_contagion(Swin, lab500)
cv = cvar_regime_ratio(Swin, lab500)
print("   boost:", round(cc["crisis_corr_boost"], 4), "cvar:", round(cv["cvar_ratio"], 3))

# e. skew + tail quantiles, 7.3a-t gen vs real
with torch.no_grad():
    g_al = model.sample_prior_scenarios(to_dev(A[te]), to_dev(NF[te].mean(axis=1)),
                                        torch.from_numpy(lab[te]).to(DEV)).cpu().numpy()
G_al = g_al * sc["std"] + sc["mean"]
R = W[te]
print("e. per-stock skew: gen mean=%.3f real mean=%.3f" % (stats.skew(G_al.ravel()), stats.skew(R.ravel())))
names = {28: tickers[28], 23: tickers[23], 16: tickers[16], 0: tickers[0], 31: tickers[31], 35: tickers[35]}
print("   stocks:", names)
for j in [28, 23, 16, 0, 31, 35]:
    print(f"   {tickers[j]}: skew gen={stats.skew(G_al[:, :, j].ravel()):.3f} real={stats.skew(R[:, :, j].ravel()):.3f} "
          f"p1 gen={np.percentile(G_al[:, :, j], 1):.4f} real={np.percentile(R[:, :, j], 1):.4f} "
          f"p99 gen={np.percentile(G_al[:, :, j], 99):.4f} real={np.percentile(R[:, :, j], 99):.4f}")
print("DONE-QUICK")
