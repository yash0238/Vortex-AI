"""Stage 7.2 eval: ratios, day-to-day structure, full evaluator, sample plot (GPU, 1 seed)."""
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
CKPT = "models/checkpoints/last-v2.ckpt"  # end of 7.2 run (no new best; val never beat 0.8349)
torch.manual_seed(7)
np.random.seed(7)
model = VORTEXModel.load_from_checkpoint(CKPT, strict=False).to(DEV).eval()
print("ckpt:", CKPT)
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


def lag1(r):
    v = r - r.mean(axis=1, keepdims=True)
    return float((v[:, :-1] * v[:, 1:]).mean() / ((v ** 2).mean() + 1e-12))


with torch.no_grad():
    xa = to_dev(((W[va].astype(np.float64) - sc["mean"]) / sc["std"]).astype(np.float32))
    rh, _, _ = model(xa, to_dev(A[va]), to_dev(NF[va].mean(axis=1)))
    print("val posterior z-MSE:", round(float(((rh - xa) ** 2).mean()), 4))
    Gr = raw(rh.cpu().numpy())
    r = Gr.std(axis=(0, 1)) / (W[va].std(axis=(0, 1)) + 1e-12)
    print(f"val posterior ratio: mean={r.mean():.3f} min={r.min():.3f} max={r.max():.3f}")
    rp = model.sample_prior_scenarios(to_dev(A[va]), to_dev(NF[va].mean(axis=1)),
                                      torch.from_numpy(lab[va]).to(DEV)).cpu().numpy()
    Rp = raw(rp)
    r2 = Rp.std(axis=(0, 1)) / (W[va].std(axis=(0, 1)) + 1e-12)
    print(f"val prior ratio: mean={r2.mean():.3f} min={r2.min():.3f} max={r2.max():.3f}")
    # day-to-day structure on val posterior
    R = W[va]
    print(f"lag1 ACF r: gen={lag1(Gr.mean(axis=0).reshape(1, -1, 1)):.4f} "
          f"(per-stock mean below)")
    lg = np.array([np.corrcoef(Gr[:, :-1, j].ravel(), Gr[:, 1:, j].ravel())[0, 1] for j in range(47)])
    lr = np.array([np.corrcoef(R[:, :-1, j].ravel(), R[:, 1:, j].ravel())[0, 1] for j in range(47)])
    print(f"lag1 ACF mean over stocks: gen={np.nanmean(lg):.4f} real={np.nanmean(lr):.4f}")
    fg = np.diff(Gr, axis=1).std()
    fr = np.diff(R, axis=1).std()
    print(f"first-diff std: gen={fg:.5f} real={fr:.5f} ratio={fg / fr:.3f}")
    print(f"median kurt: gen={_per_stock_kurt_median(Gr):.2f} real={_per_stock_kurt_median(R):.2f}")
    # prior-sampled test sets for evaluator
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
      "pass:", styl["corr_error_pass"], "| kurt gen:", round(styl["kurtosis_gen"], 2),
      "| acf:", round(styl["acf_sq_gen"], 4), styl["acf_sq_pass"])
print("disc:", round(disc["discriminative_score"], 4), disc["discriminative_pass"])
print("boost:", round(cont["crisis_corr_boost"], 4), "| cvar:", round(cvar["cvar_ratio"], 3))
import json
json.dump({"styl": styl, "disc": disc, "contagion": cont, "cvar": cvar},
          open("results/stage72_eval.json", "w"), indent=2, default=float)

fig, axes = plt.subplots(5, 1, figsize=(10, 8), sharex=True)
j = 3
for i in range(5):
    axes[i].plot(R[i * 100, :, j], label="real", alpha=0.8)
    axes[i].plot(Gr[i * 100, :, j], label="gen", alpha=0.8)
    axes[i].legend(fontsize=8)
fig.suptitle("Stage 7.2: 5 real vs 5 generated windows (one stock)")
fig.savefig("results/figures/stage72_samples.png", dpi=100)
print("saved results/figures/stage72_samples.png + results/stage72_eval.json")
