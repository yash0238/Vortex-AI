"""Step 0: graph-input audit, ex-post prior, position profile, ACF(r2), PPCA ref (GPU, no training)."""
import numpy as np
import torch
from scipy import stats
from sklearn.decomposition import PCA
import sys
sys.path.insert(0, ".")
from training.vortex_model import VORTEXModel

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
print("a. sample_prior_scenarios: graph_emb from the REAL row's node feats/adj "
      "(training/vortex_model.py: sample_prior_scenarios encodes adj,node_feats of the "
      "conditioning window); only z0 comes from the regime prior. Posterior path uses the "
      "same real graph input. So graph input is regime-INFORMED in both paths.")


def to_dev(a):
    return torch.nan_to_num(torch.from_numpy(a), nan=0.0, posinf=1.0, neginf=0.0).to(DEV)


with torch.no_grad():
    Xs = ((W[tr].astype(np.float64) - sc["mean"]) / sc["std"]).astype(np.float32)
    mu_q, _, _ = model.sde_model.encode_stats(to_dev(Xs))
    MU = mu_q.cpu().numpy()
rng = np.random.default_rng(7)
out = {}
with torch.no_grad():
    for rg, nm in [(1, "crisis"), (0, "normal")]:
        pool = MU[lab[tr] == rg]
        picks = rng.integers(0, len(pool), size=500)
        z0 = torch.from_numpy(pool[picks].astype(np.float32)).to(DEV)
        grows = np.where(lab[tr] == rg)[0]
        gpick = tr[np.tile(grows, 500 // len(grows) + 1)[:500]]
        ne, _ = model.gat(to_dev(NF[gpick].mean(axis=1)), to_dev(A[gpick]))
        _, zp = model.sde_model.decode_from_z0(ne.mean(dim=1), z0)
        gen = model.sde_model.emission_sample(zp).cpu().numpy() * sc["std"] + sc["mean"]
        out[nm] = gen
        out[nm + "_sig"] = float(model.sde_model.emission_sigma(zp).cpu().numpy().mean())
        print(f"b. ex-post {nm}: sigma mean={out[nm + '_sig']:.3f}")
gc, gn = out["crisis"], out["normal"]
print(f"   ex-post sigma ratio: {out['crisis_sig'] / out['normal_sig']:.3f} (prior-sampled was 1.01x)")
# per-stock std ratio per regime set vs matched real
Rc = W[tr][lab[tr] == 1][:500] if (lab[tr] == 1).sum() >= 500 else W[tr][lab[tr] == 1]
print(f"   ex-post gen std ratio (vs same-regime real): crisis={(gc.std(axis=(0,1)) / (W[tr][lab[tr]==1].std(axis=(0,1))+1e-12)).mean():.3f} "
      f"normal={(gn.std(axis=(0,1)) / (W[tr][lab[tr]==0].std(axis=(0,1))+1e-12)).mean():.3f}")
# shrunk full-cov Gaussian per regime on posterior mus
for rg, nm in [(1, "crisis"), (0, "normal")]:
    pool = MU[lab[tr] == rg]
    C = np.cov(pool.T)
    sh = 0.1 * np.trace(C) / C.shape[0]
    Cs = (1 - 0.1) * C + sh * np.eye(C.shape[0])
    print(f"   {nm} posterior-mu cov: mean var={np.diag(C).mean():.3f}, shrinkage={sh:.4f}")

# c. per-position profile on val SAMPLED generations (not the mean path) vs real
with torch.no_grad():
    xa = to_dev(((W[va].astype(np.float64) - sc["mean"]) / sc["std"]).astype(np.float32))
    rh, _, _ = model(xa, to_dev(A[va]), to_dev(NF[va].mean(axis=1)))
    Gs = (model.sde_model.emission_sample(model._last_zs).cpu().numpy()
          * sc["std"] + sc["mean"])
R = W[va]
for name, arr in [("std", lambda a: a.std(axis=(0, 2))), ("mean", lambda a: a.mean(axis=(0, 2))),
                  ("kurt", lambda a: stats.kurtosis(a, axis=(0, 2), fisher=False))]:
    g, r = arr(Gs), arr(R)
    d = np.abs(g - r)
    print(f"c. {name}: maxdev={d.max():.4f} at pos={int(d.argmax())} (gen[{g[d.argmax()]:.4f}] real[{r[d.argmax()]:.4f}])")

# d. ACF(r2) posterior vs prior-sampled + PPCA rank-8 ref NLL
def acf2(a):
    s = a ** 2
    v = s - s.mean(axis=1, keepdims=True)
    return float(((v[:, :-1] * v[:, 1:]).mean()) / (v.var() + 1e-12))


with torch.no_grad():
    rp = model.sample_prior_scenarios(to_dev(A[va]), to_dev(NF[va].mean(axis=1)),
                                      torch.from_numpy(lab[va]).to(DEV)).cpu().numpy()
    Gp = rp * sc["std"] + sc["mean"]
print(f"d. ACF(r2): posterior={acf2(Gr):.4f} prior-sampled={acf2(Gp):.4f} real=0.1371")
xtr = (W[tr].astype(np.float64) - sc["mean"]) / sc["std"]
xv = (W[va].astype(np.float64) - sc["mean"]) / sc["std"]
flat_tr = xtr.reshape(-1, xtr.shape[-1])
mu8 = flat_tr.mean(0)
pca = PCA(n_components=8).fit(flat_tr - mu8)
Z = pca.transform((xv.reshape(-1, xv.shape[-1]) - mu8))
rec = Z @ pca.components_ + mu8
res = xv.reshape(-1, xv.shape[-1]) - rec
C8 = np.cov((flat_tr - mu8 - (flat_tr - mu8) @ pca.components_.T @ pca.components_).T) + 1e-6 * np.eye(47)
import numpy.linalg as la
sign, logdet = la.slogdet(C8)
Ci = la.inv(C8)
nll_ppca = float(0.5 * (47 * np.log(2 * np.pi) + logdet + np.einsum("ij,jk,ik->i", res, Ci, res).mean()))
print(f"   static rank-8 Gaussian NLL: {nll_ppca:.3f} (unit 62.67; 7.3a-t val ~61.1)")
print("DONE")
