"""7.3b eval: V3 + V1 priors, headline val+test and train, profile, plot, median kurt (GPU)."""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import sys
sys.path.insert(0, ".")
from training.vortex_model import VORTEXModel
from eval.statistical import evaluate_stylized_facts, _per_stock_kurt_median
from eval.discriminative import discriminative_score
from eval.contagion import evaluate_contagion, cvar_regime_ratio

DEV = "cuda"
CKPT = "models/checkpoints/vortex-epoch=07-val/total=57.4987.ckpt"
torch.manual_seed(7)
np.random.seed(7)
model = VORTEXModel.load_from_checkpoint(CKPT, strict=False).to(DEV).eval()
print("ckpt:", CKPT, "| nu:", round(float(model.sde_model._nu(torch.zeros(1).to(DEV))), 2))
W = np.load("data/raw/windows.npy").astype(np.float64)
lab = np.load("data/raw/window_regimes_v2.npy")
A = np.load("data/raw/adj_matrices.npy").astype(np.float32)
NF = np.load("data/raw/window_node_features.npy").astype(np.float32)
sp = np.load("data/raw/split.npz")
tr, va, te = (np.asarray(sp[k]) for k in ["train_idx", "val_idx", "test_idx"])
sc = np.load("data/raw/scaler.npz")
vt = np.concatenate([va, te])


def to_dev(a):
    return torch.nan_to_num(torch.from_numpy(a), nan=0.0, posinf=1.0, neginf=0.0).to(DEV)


with torch.no_grad():
    mu_all, lv_all, _ = model.sde_model.encode_stats(
        to_dev(((W[tr].astype(np.float64) - sc["mean"]) / sc["std"]).astype(np.float32)))
    MU, LV = mu_all.cpu().numpy(), lv_all.cpu().numpy()
rng = np.random.default_rng(7)


def gen(rows, z0):
    with torch.no_grad():
        ne, _ = model.gat(to_dev(NF[rows].mean(axis=1)), to_dev(A[rows]))
        _, zp = model.sde_model.decode_from_z0(ne.mean(dim=1), torch.from_numpy(z0.astype(np.float32)).to(DEV))
        return (model.sde_model.emission_sample(zp).cpu().numpy() * sc["std"] + sc["mean"])


def v3_z0(n_each, seed):
    r = np.random.default_rng(seed)
    out = []
    for rg in (1, 0):
        p = MU[lab[tr] == rg]
        lv = LV[lab[tr] == rg]
        ix = r.integers(0, len(p), n_each)
        out.append(p[ix] + r.normal(0, 1, size=(n_each, 64)) * np.exp(0.5 * lv[ix]))
    return np.concatenate(out)


def v1_z0(n_each, seed):
    r = np.random.default_rng(seed)
    pm, plv = (model.prior_mu.detach().cpu().numpy(), model.prior_logvar.detach().cpu().numpy())
    return np.concatenate([r.normal(pm[rg], np.exp(0.5 * plv[rg]), size=(n_each, 64)) for rg in (1, 0)])


def grows_for(rows, n_each, seed):
    r = np.random.default_rng(seed)
    out = []
    for rg in (1, 0):
        pool = rows[lab[rows] == rg]
        out.append(r.choice(pool, n_each, replace=True))
    return np.concatenate(out)


import json
rep = {}
for tag, rows, zfn in [("headline-vt", vt, v3_z0), ("train", tr, v3_z0)]:
    for prior, z in [("V3", zfn(500, 11)), ("V1", v1_z0(500, 12))]:
        gr = grows_for(rows, 500, 13)
        G = gen(gr, z)
        R = W[gr]
        ratio = (G.std(axis=(0, 1)) / (R.std(axis=(0, 1)) + 1e-12)).mean()
        lg = np.array([np.corrcoef(G[:, :-1, j].ravel(), G[:, 1:, j].ravel())[0, 1] for j in range(47)])
        lr = np.array([np.corrcoef(R[:, :-1, j].ravel(), R[:, 1:, j].ravel())[0, 1] for j in range(47)])
        st = evaluate_stylized_facts(R, G)
        dc = discriminative_score(R, G, n_samples=1000)
        lb = np.array([1] * 500 + [0] * 500)
        ct = evaluate_contagion(G, lb)
        cv = cvar_regime_ratio(G, lb)
        rep[f"{tag}-{prior}"] = {"ratio": ratio, "lag1": (np.nanmean(lg), np.nanmean(lr)),
                                 "kurtmed": (_per_stock_kurt_median(G), _per_stock_kurt_median(R)),
                                 "corr": st["corr_error"], "mae": st["corr_mae"],
                                 "disc": dc["discriminative_score"],
                                 "boost": ct["crisis_corr_boost"], "cvar": cv["cvar_ratio"]}
        print(f"{tag}-{prior}: ratio={ratio:.3f} lag1={np.nanmean(lg):.4f}/{np.nanmean(lr):.4f} "
              f"kurtmed={_per_stock_kurt_median(G):.2f}/{_per_stock_kurt_median(R):.2f} "
              f"corr={st['corr_error']:.2f} mae={st['corr_mae']:.4f} disc={dc['discriminative_score']:.4f} "
              f"boost={ct['crisis_corr_boost']:.4f} cvar={cv['cvar_ratio']:.3f}")
json.dump(rep, open("results/stage73b_eval.json", "w"), indent=2, default=float)
# position profile + plot on headline V3
G = gen(grows_for(vt, 500, 13), v3_z0(500, 11))
R = W[grows_for(vt, 500, 13)]
d = np.abs(G.std(axis=(0, 2)) - R.std(axis=(0, 2)))
print("position std maxdev:", round(float(d.max()), 4), "at", int(d.argmax()))
fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
for s_ in range(3):
    for i in range(5):
        axes[s_].plot(R[i * 97, :, s_ * 15], alpha=0.5)
    for i in range(5):
        axes[s_].plot(G[i * 97, :, s_ * 15], alpha=0.5, linestyle="--")
fig.suptitle("Stage 7.3b: 5 real (solid) vs 5 generated (dashed) per stock")
fig.savefig("results/figures/stage73b_samples.png", dpi=100)
print("saved results/stage73b_eval.json + results/figures/stage73b_samples.png")
