"""Stage 7.1 eval: posterior vs prior ratios + evaluator on prior-sampled test scenarios (GPU)."""
import glob
import json
import os
import re
import numpy as np
import torch
import sys
sys.path.insert(0, ".")
from training.vortex_model import VORTEXModel
from eval.statistical import evaluate_stylized_facts
from eval.discriminative import discriminative_score
from eval.contagion import evaluate_contagion, cvar_regime_ratio

DEV = "cuda"
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", type=str, default=None, help="explicit checkpoint (default: newest 7.1 epoch-18)")
args = ap.parse_args()
ckpts = sorted(glob.glob("models/checkpoints/**/*.ckpt", recursive=True), key=os.path.getmtime)
if args.ckpt:
    best = args.ckpt
else:
    cands = [c for c in ckpts if "epoch=18" in c]
    best = cands[-1] if cands else max(ckpts, key=os.path.getmtime)
print("ckpt:", best)
model = VORTEXModel.load_from_checkpoint(best, strict=False).to(DEV).eval()
W = np.load("data/raw/windows.npy").astype(np.float32)
A = np.load("data/raw/adj_matrices.npy").astype(np.float32)
NF = np.load("data/raw/window_node_features.npy").astype(np.float32)
lab = np.load("data/raw/window_regimes_v2.npy")
sp = np.load("data/raw/split.npz")
va, te = np.asarray(sp["val_idx"]), np.asarray(sp["test_idx"])
sc = np.load("data/raw/scaler.npz")
mu_t = torch.from_numpy(sc["mean"]).to(DEV)
sd_t = torch.from_numpy(sc["std"]).to(DEV)


def to_dev(a):
    return torch.nan_to_num(torch.from_numpy(a), nan=0.0, posinf=1.0, neginf=0.0).to(DEV)


def ratio(gen_raw, real_raw, tag):
    r = gen_raw.std(axis=(0, 1)) / (real_raw.std(axis=(0, 1)) + 1e-12)
    print(f"{tag} std ratio: mean={r.mean():.3f} min={r.min():.3f} max={r.max():.3f}")
    return r


def raw(gen_z):
    return (gen_z * sc["std"] + sc["mean"])


with torch.no_grad():
    # posterior recon on val
    xa = to_dev(((W[va].astype(np.float64) - sc["mean"]) / sc["std"]).astype(np.float32))
    rh, _, _ = model(xa, to_dev(A[va]), to_dev(NF[va].mean(axis=1)))
    print("val posterior z-MSE:", round(float(((rh - xa) ** 2).mean()), 4))
    ratio(raw(rh.cpu().numpy()), W[va], "val posterior")
    # prior-sampled on val rows (regime of each row)
    rp = model.sample_prior_scenarios(to_dev(A[va]), to_dev(NF[va].mean(axis=1)),
                                      torch.from_numpy(lab[va]).to(DEV)).cpu().numpy()
    ratio(raw(rp), W[va], "val prior-sampled")
    # aligned 738 prior scenarios on test (1 per real window, same order)
    xt = to_dev(((W[te].astype(np.float64) - sc["mean"]) / sc["std"]).astype(np.float32))
    g_al = model.sample_prior_scenarios(to_dev(A[te]), to_dev(NF[te].mean(axis=1)),
                                        torch.from_numpy(lab[te]).to(DEV)).cpu().numpy()
    G_al = raw(g_al)
    # 500 per regime by cycling
    nrm = np.where(lab[te] == 0)[0]
    cri = np.where(lab[te] == 1)[0]
    picks = np.concatenate([np.tile(nrm, 500 // len(nrm) + 1)[:500],
                            np.tile(cri, 500 // len(cri) + 1)[:500]])
    g500 = model.sample_prior_scenarios(to_dev(A[te][picks]), to_dev(NF[te][picks].mean(axis=1)),
                                        torch.from_numpy(lab[te][picks]).to(DEV)).cpu().numpy()
    G500 = raw(g500)
    l500 = lab[te][picks]

styl = evaluate_stylized_facts(W[te], G_al)
disc = discriminative_score(W[te], G_al, n_samples=len(te))
cont = evaluate_contagion(G500, l500)
cvar = cvar_regime_ratio(G500, l500)
print("styl:", {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in styl.items()},
      "interval:", [round(float(x), 2) for x in styl["kurtosis_interval"]])
print("disc:", {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in disc.items()})
print("contagion:", {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in cont.items()})
print("cvar:", {k: (round(float(v), 6) if isinstance(v, float) else v) for k, v in cvar.items()})
json.dump({"styl": styl, "disc": disc, "contagion": cont, "cvar": cvar},
          open("results/stage71_eval.json", "w"), indent=2, default=float)
print("saved results/stage71_eval.json")
