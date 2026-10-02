"""Step 0: same-function NLL check, tail share, Gaussian CV disc, sigma ratios by split."""
import numpy as np
import torch
from scipy import stats
import sys
sys.path.insert(0, ".")
from training.vortex_model import VORTEXModel
from eval.discriminative import discriminative_score

DEV = "cuda"
CKPT = "models/checkpoints/vortex-epoch=05-val/total=62.6602.ckpt"
torch.manual_seed(7)
np.random.seed(7)
W = np.load("data/raw/windows.npy").astype(np.float64)
lab = np.load("data/raw/window_regimes_v2.npy")
A = np.load("data/raw/adj_matrices.npy").astype(np.float32)
NF = np.load("data/raw/window_node_features.npy").astype(np.float32)
sp = np.load("data/raw/split.npz")
tr, va, te = (np.asarray(sp[k]) for k in ["train_idx", "val_idx", "test_idx"])
sc = np.load("data/raw/scaler.npz")
X = (W - sc["mean"]) / sc["std"]  # z-space val inputs
xv, xtr = X[va], X[tr]
N = xv.shape[-1]


def nll_elem(mean, var, x, df=None):
    """One function for all baselines: mean per-element NLL. var = variance."""
    if df is None:
        return float((0.5 * (np.log(2 * np.pi * var) + (x - mean) ** 2 / var)).mean())
    sd = np.sqrt(var * (df - 2) / df)
    return float((-stats.t.logpdf((x - mean) / sd, df) + np.log(sd)).mean())


m0, v0 = xtr.mean(axis=(0, 1)), xtr.var(axis=(0, 1)) + 1e-8
a_unit = nll_elem(0.0, 1.0, xv)
a_gauss = nll_elem(m0, v0, xv)
NS = xtr.shape[2]
dfs = np.zeros(NS, dtype=int)
bnll = np.full(NS, 1e9)
for df_ in [3, 4, 5, 6, 8, 10, 15, 30]:
    nll = np.array([nll_elem(m0[j], v0[j], xv[:, :, j], df_) for j in range(NS)])
    better = nll < bnll
    bnll[better] = nll[better]
    dfs[better] = df_
print(f"a. per-element NLL: unit={a_unit:.4f} gauss={a_gauss:.4f} t={bnll.mean():.4f} "
      f"(x47: {a_unit * N:.2f} {a_gauss * N:.2f} {bnll.mean() * N:.2f})")
print(f"   fitted df: median={int(np.median(dfs))} min={int(dfs.min())} max={int(dfs.max())}")
model = VORTEXModel.load_from_checkpoint(CKPT, strict=False).to(DEV).eval()
print(f"   model per-element NLL: {62.679 / N:.4f} (joint 62.679/47)")


def tail_share(x, mean, var, df=None, q=0.995):
    if df is None:
        el = 0.5 * (np.log(2 * np.pi * var) + (x - mean) ** 2 / var)
    else:
        sd = np.sqrt(var * (df - 2) / df)
        el = -stats.t.logpdf((x - mean) / sd, df) + np.log(sd)
    thr = np.quantile(el, q)
    m = el > thr
    return float(el[m].sum() / el.sum()), float(el[~m].mean() * N)


ts, tm_ = m0, v0
share_g, excl_g = tail_share(xv, m0, v0)
print(f"b. top-0.5pct NLL share: gauss={share_g:.3f}, excl NLL={excl_g:.3f}")
print("   (t tail share needs per-stock df; using median df)")
share_t, excl_t = tail_share(xv, m0, v0, df=float(np.median(dfs)))
print(f"   top-0.5pct NLL share: t={share_t:.3f}, excl NLL={excl_t:.3f}")

rng = np.random.default_rng(5)
mu500, sd500 = xv.mean(axis=(0, 1)), xv.std(axis=(0, 1)) + 1e-12
G = rng.normal(mu500, sd500, size=(500, xv.shape[1], xv.shape[2]))
d = discriminative_score(xv[:500], G, n_samples=500)
print(f"c. Gaussian CV disc (exact protocol): {d['discriminative_score']:.4f} pass={d['discriminative_pass']}")


def to_dev(a):
    return torch.nan_to_num(torch.from_numpy(a), nan=0.0, posinf=1.0, neginf=0.0).to(DEV)


NF = np.load("data/raw/window_node_features.npy").astype(np.float32)
with torch.no_grad():
    for name, idx in [("train", tr), ("val+test", np.concatenate([va, te]))]:
        Xs = ((W[idx].astype(np.float64) - sc["mean"]) / sc["std"]).astype(np.float32)
        r = lab[idx]
        mu_q, _, _ = model.sde_model.encode_stats(to_dev(Xs))
        z0 = mu_q  # posterior mean state
        ne, _ = model.gat(to_dev(NF[idx].mean(axis=1)), to_dev(
            np.nan_to_num(A[idx], nan=0.0, posinf=1.0, neginf=0.0)))
        _, zs = model.sde_model.decode_from_z0(ne.mean(dim=1), z0)
        sig = model.sde_model.emission_sigma(zs).cpu().numpy()
        ms = sig.mean(axis=(1, 2))
        print(f"d. {name} (n_crisis={int((r == 1).sum())}): posterior sigma crisis={ms[r == 1].mean():.3f} "
              f"normal={ms[r == 0].mean():.3f}")
print("DONE")
