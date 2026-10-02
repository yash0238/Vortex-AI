"""Step 0 diag4: z0 decomposition (5 seeds), disc controls, NN replay check, temporal PPCA."""
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
import sys
sys.path.insert(0, ".")
from training.vortex_model import VORTEXModel

DEV = "cuda"
CKPT = "models/checkpoints/vortex-epoch=08-val/total=61.1070.ckpt"
model = VORTEXModel.load_from_checkpoint(CKPT, strict=False).to(DEV).eval()
W = np.load("data/raw/windows.npy").astype(np.float64)
lab = np.load("data/raw/window_regimes_v2.npy")
A = np.load("data/raw/adj_matrices.npy").astype(np.float32)
NF = np.load("data/raw/window_node_features.npy").astype(np.float32)
sp = np.load("data/raw/split.npz")
tr, va, te = (np.asarray(sp[k]) for k in ["train_idx", "val_idx", "test_idx"])
sc = np.load("data/raw/scaler.npz")
print("a. conditioning rows are SAME-REGIME train rows (diag3: gpick drawn from same-regime "
      "grows; vortex_model sample_prior_scenarios takes caller rows). No mixed rerun needed.")
Xtr = ((W[tr].astype(np.float64) - sc["mean"]) / sc["std"]).astype(np.float32)


def to_dev(a):
    return torch.nan_to_num(torch.from_numpy(a), nan=0.0, posinf=1.0, neginf=0.0).to(DEV)


with torch.no_grad():
    mu_all, lv_all, _ = model.sde_model.encode_stats(to_dev(Xtr))
    MU, LV = mu_all.cpu().numpy(), lv_all.cpu().numpy()
print("posterior mu var: crisis=%.4f normal=%.4f" % (MU[lab[tr] == 1].var(), MU[lab[tr] == 0].var()))


def gen_from_z0(z0_np, grows_np):
    with torch.no_grad():
        ne, _ = model.gat(to_dev(NF[grows_np].mean(axis=1)), to_dev(A[grows_np]))
        _, zp = model.sde_model.decode_from_z0(ne.mean(dim=1),
                                               torch.from_numpy(z0_np.astype(np.float32)).to(DEV))
        sig = model.sde_model.emission_sigma(zp).cpu().numpy()
        gen = model.sde_model.emission_sample(zp).cpu().numpy() * sc["std"] + sc["mean"]
    return sig, gen


def ratios(sig, gen, r):
    ms = sig.mean(axis=(1, 2))
    out = {"sig_ratio": float(ms[r == 1].mean() / ms[r == 0].mean())}
    for rg, nm in [(1, "crisis"), (0, "normal")]:
        g, real = gen[r == rg], W[tr][lab[tr] == rg]
        out[f"genstd_{nm}"] = float(g.std(axis=(0, 1)).mean())
        out[f"genreal_{nm}"] = float((g.std(axis=(0, 1)) / (real.std(axis=(0, 1)) + 1e-12)).mean())
    return out


real_ratio = {}
for rg, nm in [(1, "crisis"), (0, "normal")]:
    real_ratio[nm] = W[tr][lab[tr] == rg].std(axis=(0, 1))
print("REAL per-stock std ratio crisis/normal: %.3f" % float((real_ratio["crisis"] / real_ratio["normal"]).mean()))

res = {}
with torch.no_grad():
    pm, plv = model.prior_mu.cpu().numpy(), model.prior_logvar.cpu().numpy()
for seed in range(5):
    rng = np.random.default_rng(100 + seed)
    # V1 parametric prior
    z1 = np.concatenate([rng.normal(pm[1], np.exp(0.5 * plv[1]), size=(500, 64)),
                         rng.normal(pm[0], np.exp(0.5 * plv[0]), size=(500, 64))])
    grows = np.concatenate([tr[np.tile(np.where(lab[tr] == 1)[0], 500 // int((lab[tr] == 1).sum()) + 1)[:500]],
                            tr[np.tile(np.where(lab[tr] == 0)[0], 500 // int((lab[tr] == 0).sum()) + 1)[:500]]])
    rr = np.array([1] * 500 + [0] * 500)
    s, g = gen_from_z0(z1, grows)
    res.setdefault("V1_prior", []).append(ratios(s, g, rr))
    # V2 mu deterministic
    mu_pool = {rg: MU[lab[tr] == rg] for rg in (0, 1)}
    z2 = np.concatenate([mu_pool[1][rng.integers(0, len(mu_pool[1]), 500)],
                         mu_pool[0][rng.integers(0, len(mu_pool[0]), 500)]])
    s, g = gen_from_z0(z2, grows)
    res.setdefault("V2_mu", []).append(ratios(s, g, rr))
    # V3 sampled posterior
    lv_pool = {rg: LV[lab[tr] == rg] for rg in (0, 1)}
    samp = {}
    for rg in (0, 1):
        idx = rng.integers(0, len(mu_pool[rg]), 500)
        samp[rg] = mu_pool[rg][idx] + rng.normal(0, 1, size=(500, 64)) * np.exp(0.5 * lv_pool[rg][idx])
    z3 = np.concatenate([samp[1], samp[0]])
    s, g = gen_from_z0(z3, grows)
    res.setdefault("V3_qsample", []).append(ratios(s, g, rr))
    # V4 shrunk Gaussian fit to V3 pool
    z4 = np.concatenate([
        rng.multivariate_normal(z3[:500].mean(0),
                                0.9 * np.cov(z3[:500].T) + 0.1 * np.trace(np.cov(z3[:500].T)) / 64 * np.eye(64),
                                size=500),
        rng.multivariate_normal(z3[500:].mean(0),
                                0.9 * np.cov(z3[500:].T) + 0.1 * np.trace(np.cov(z3[500:].T)) / 64 * np.eye(64),
                                size=500)])
    s, g = gen_from_z0(z4, grows)
    res.setdefault("V4_shrunk", []).append(ratios(s, g, rr))
    # V5 paired (z0_i with graph_i), train + val posteriors sampled
    for tag, idx in [("train", tr), ("val", va)]:
        Xs = ((W[idx].astype(np.float64) - sc["mean"]) / sc["std"]).astype(np.float32)
        with torch.no_grad():
            mq, lq, _ = model.sde_model.encode_stats(to_dev(Xs))
            z5 = (mq + torch.randn_like(mq) * torch.exp(0.5 * lq)).cpu().numpy()
        s, g = gen_from_z0(z5, idx)
        res.setdefault(f"V5_paired_{tag}", []).append(ratios(s, g, lab[idx]))
for k, v in res.items():
    sr = np.mean([d["sig_ratio"] for d in v])
    gr = np.mean([(d["genreal_crisis"] + d["genreal_normal"]) / 2 for d in v])
    print(f"b. {k}: sig_ratio={sr:.3f} +- {np.std([d['sig_ratio'] for d in v]):.3f} "
          f"genreal={gr:.3f}")

# c. disc on best of V3/V4 + time-shuffle control + NN replay
best = "V3_qsample" if np.mean([d["sig_ratio"] for d in res["V3_qsample"]]) >= np.mean(
    [d["sig_ratio"] for d in res["V4_shrunk"]]) else "V4_shrunk"
print("c. best variant:", best)
from eval.discriminative import discriminative_score
rng = np.random.default_rng(7)
Xr = W[te][:500]
G = {"V3_qsample": None, "V4_shrunk": None}
with torch.no_grad():
    for i, tag in enumerate(["V3_qsample", "V4_shrunk"]):
        mu_pool = {rg: MU[lab[tr] == rg] for rg in (0, 1)}
        if tag == "V3_qsample":
            lv_pool = {rg: LV[lab[tr] == rg] for rg in (0, 1)}
            idx = {rg: rng.integers(0, len(mu_pool[rg]), 500) for rg in (0, 1)}
            z = np.concatenate([(mu_pool[rg][idx[rg]] +
                                 rng.normal(0, 1, size=(500, 64)) * np.exp(0.5 * lv_pool[rg][idx[rg]])) for rg in (1, 0)])
        else:
            z = np.concatenate([rng.multivariate_normal(
                mu_pool[rg].mean(0), 0.9 * np.cov(mu_pool[rg].T) + 0.1 * np.trace(np.cov(mu_pool[rg].T)) / 64 * np.eye(64), size=500) for rg in (1, 0)])
        grows = np.tile(te[:500], 2)
        ne, _ = model.gat(to_dev(NF[grows].mean(axis=1)), to_dev(A[grows]))
        _, zp = model.sde_model.decode_from_z0(ne.mean(dim=1), torch.from_numpy(z.astype(np.float32)).to(DEV))
        G[tag] = model.sde_model.emission_sample(zp).cpu().numpy() * sc["std"] + sc["mean"]
r500c = rng.choice(np.where(lab == 1)[0], 500, replace=True)
r500n = rng.choice(np.where(lab == 0)[0], 500, replace=False)
Xr1000 = W[np.concatenate([r500c, r500n])]
for tag in ["V3_qsample", "V4_shrunk"]:
    d = discriminative_score(Xr1000, G[tag], n_samples=1000)
    print(f"   disc {tag} (500/regime, CV): {d['discriminative_score']:.4f}")
Xs = W[te][:500].copy()
for i in range(len(Xs)):
    Xs[i] = Xs[i][rng.permutation(60)]
d = discriminative_score(W[te][:500], Xs, n_samples=500)
print(f"   disc real-vs-shuffled: {d['discriminative_score']:.4f}")
r500c = rng.choice(np.where(lab == 1)[0], 500, replace=True)
r500n = rng.choice(np.where(lab == 0)[0], 500, replace=False)
Xa = np.concatenate([W[np.concatenate([r500c, r500n])].reshape(1000, -1), G[best].reshape(1000, -1)])
ya = np.array([1] * 1000 + [0] * 1000)
clf = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42).fit(Xa, ya)
imp = clf.feature_importances_.reshape(60, 47)
top = np.dstack(np.unravel_index(np.argsort(imp.ravel())[-5:], imp.shape))[0][::-1]
print("   top-5 (day,stock):", [(int(t), int(s)) for t, s in top])


def nn_mean(gen, ref):
    g = torch.from_numpy(gen.reshape(len(gen), -1)).to(DEV)
    out = []
    for i in range(0, len(ref), 500):
        r = torch.from_numpy(ref[i:i + 500].reshape(len(ref[i:i + 500]), -1)).to(DEV)
        out.append(torch.cdist(g, r).min(dim=1).values.cpu().numpy())
    return float(np.concatenate(out).mean())


print("   NN gen-to-train:", round(nn_mean(G[best], xtr), 2),
      "val-real-to-train:", round(nn_mean(xv, xtr), 2))

# d. temporal PPCA grid: fit PCA on pre-2017 train windows, score on val
import pandas as pd
px_idx = pd.read_csv("data/raw/nifty50_close.csv", index_col=0, parse_dates=True).index
px_idx = px_idx[px_idx >= "2010-11-08"][1:]
tr_dates = px_idx[[59 + j for j in tr]]
fit_mask = tr_dates < "2017-01-01"
Xfit = ((W[tr][fit_mask].astype(np.float64) - sc["mean"]) / sc["std"])
xtr = Xfit
xv = ((W[va].astype(np.float64) - sc["mean"]) / sc["std"])
F = Xfit.reshape(-1, 47)
mu_f = F.mean(0)
Fc = F - mu_f
eig = np.linalg.eigvalsh(np.cov(Fc.T))[::-1]
print("d. temporal fit: train rows before 2017, n_win=", int(fit_mask.sum()))
for K in [4, 8, 16]:
    pca = PCA(n_components=K).fit(Fc)
    B = pca.components_.T
    for s in [0.0, 0.1, 0.3]:
        sig2 = eig[K:].mean()
        C = (1 - s) * (B @ np.diag(pca.explained_variance_) @ B.T + sig2 * np.eye(47)) + s * np.trace(
            B @ np.diag(pca.explained_variance_) @ B.T + sig2 * np.eye(47)) / 47 * np.eye(47)
        sign, logdet = np.linalg.slogdet(C)
        Ci = np.linalg.inv(C)
        R = (xv.reshape(-1, 47) - mu_f)
        nll = float(0.5 * (47 * np.log(2 * np.pi) + logdet + np.einsum("ij,jk,ik->i", R, Ci, R).mean()))
        print(f"   K={K} shrink={s}: val NLL={nll:.2f} gain={62.67 - nll:.2f}")
print("DONE")
