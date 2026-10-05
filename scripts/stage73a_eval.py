"""Stage 7.3a eval: ACF, ratio, median kurt, sigma probe, plot, full evaluator (GPU, 1 seed)."""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, ".")
from training.vortex_model import VORTEXModel
from eval.statistical import evaluate_stylized_facts, _per_stock_kurt_median
from eval.discriminative import discriminative_score
from eval.contagion import evaluate_contagion, cvar_regime_ratio

DEV = "cuda"
import argparse as _ap
_p = _ap.ArgumentParser()
_p.add_argument("--ckpt", type=str, default="models/checkpoints/vortex-epoch=05-val/total=62.6602.ckpt")
CKPT, _ = _p.parse_known_args()
CKPT = CKPT.ckpt
torch.manual_seed(7)
np.random.seed(7)
model = VORTEXModel.load_from_checkpoint(CKPT, strict=False).to(DEV).eval()
print("ckpt:", CKPT, "| emission:", model.emission, "K:", model.rank_K,
      "sigma_max:", model.sde_model.sde.sigma_max)
W = np.load("data/raw/windows.npy").astype(np.float32)
A = np.load("data/raw/adj_matrices.npy").astype(np.float32)
NF = np.load("data/raw/window_node_features.npy").astype(np.float32)
lab = np.load("data/raw/window_regimes_v2.npy")
sp = np.load("data/raw/split.npz")
va, te = np.asarray(sp["val_idx"]), np.asarray(sp["test_idx"])
sc = np.load("data/raw/scaler.npz")


def to_dev(a):
    return torch.nan_to_num(torch.from_numpy(a), nan=0.0, posinf=1.0, neginf=0.0).to(DEV)


def raw(g):
    return g * sc["std"] + sc["mean"]


with torch.no_grad():
    xa = to_dev(((W[va].astype(np.float64) - sc["mean"]) / sc["std"]).astype(np.float32))
    rh, _, _ = model(xa, to_dev(A[va]), to_dev(NF[va].mean(axis=1)))
    zs = model._last_zs
    nll = model.sde_model.emission_nll(zs, xa)
    print("val NLL:", round(float(nll), 3))
    sig = model.sde_model.emission_sigma(zs).cpu().numpy()
    print(f"emission sigma: mean={sig.mean():.3f} min={sig.min():.3f} max={sig.max():.3f} std={sig.std():.3f}")
    # crisis vs normal prior sigma
    for rg, nm in [(1, "crisis"), (0, "normal")]:
        m = model.prior_mu[rg].unsqueeze(0).expand(64, -1)
        ge = torch.zeros(64, 64).to(DEV)
        z0 = m + torch.randn_like(m) * torch.exp(0.5 * model.prior_logvar[rg])
        _, zp = model.sde_model.decode_from_z0(ge, z0)
        s = model.sde_model.emission_sigma(zp).cpu().numpy()
        print(f"prior {nm} sigma mean: {s.mean():.3f}")
    Gr = raw(model.sample_prior_scenarios(to_dev(A[va]), to_dev(NF[va].mean(axis=1)),
                                          torch.from_numpy(lab[va]).to(DEV)).cpu().numpy())
    R = W[va]
    r = Gr.std(axis=(0, 1)) / (R.std(axis=(0, 1)) + 1e-12)
    print(f"val prior ratio: mean={r.mean():.3f} min={r.min():.3f} max={r.max():.3f}")
    lg = np.array([np.corrcoef(Gr[:, :-1, j].ravel(), Gr[:, 1:, j].ravel())[0, 1] for j in range(47)])
    lr = np.array([np.corrcoef(R[:, :-1, j].ravel(), R[:, 1:, j].ravel())[0, 1] for j in range(47)])
    print(f"lag1 ACF: gen={np.nanmean(lg):.4f} real={np.nanmean(lr):.4f}")
    print(f"median kurt: gen={_per_stock_kurt_median(Gr):.2f} real={_per_stock_kurt_median(R):.2f}")
    xt = to_dev(((W[te].astype(np.float64) - sc["mean"]) / sc["std"]).astype(np.float32))
    g_al = model.sample_prior_scenarios(to_dev(A[te]), to_dev(NF[te].mean(axis=1)),
                                        torch.from_numpy(lab[te]).to(DEV)).cpu().numpy()
    G_al = raw(g_al)
    nrm = np.where(lab[te] == 0)[0]
    cri = np.where(lab[te] == 1)[0]
    picks = np.concatenate([np.tile(nrm, 500 // len(nrm) + 1)[:500],
                            np.tile(cri, 500 // len(cri) + 1)[:500]])
    g500 = model.sample_prior_scenarios(to_dev(A[te][picks]), to_dev(NF[te][picks].mean(axis=1)),
                                        torch.from_numpy(lab[te][picks]).to(DEV)).cpu().numpy()
    G500, l500 = raw(g500), lab[te][picks]

styl = evaluate_stylized_facts(W[te], G_al)
disc = discriminative_score(W[te], G_al, n_samples=len(te))
cont = evaluate_contagion(G500, l500)
cvar = cvar_regime_ratio(G500, l500)
print("styl corr:", round(styl["corr_error"], 2), "mae:", round(styl["corr_mae"], 4),
      "| kurt gen:", round(styl["kurtosis_gen"], 2), "| acf:", round(styl["acf_sq_gen"], 4))
print("disc:", round(disc["discriminative_score"], 4), "| boost:", round(cont["crisis_corr_boost"], 4),
      "| cvar:", round(cvar["cvar_ratio"], 3))
import json
json.dump({"styl": styl, "disc": disc, "contagion": cont, "cvar": cvar},
          open("results/stage73a_eval.json", "w"), indent=2, default=float)
fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
for s_ in range(3):
    for i in range(5):
        axes[s_].plot(R[i * 97, :, s_ * 15], alpha=0.5)
    for i in range(5):
        axes[s_].plot(Gr[i * 97, :, s_ * 15], alpha=0.5, linestyle="--")
fig.suptitle("Stage 7.3a: 5 real (solid) vs 5 generated (dashed) per stock")
fig.savefig("results/figures/stage73a_samples.png", dpi=100)
print("saved results/figures/stage73a_samples.png + results/stage73a_eval.json")
