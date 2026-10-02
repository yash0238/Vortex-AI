"""Diag4 remainder 1: NN replay check (GPU)."""
import numpy as np
import torch
import sys
sys.path.insert(0, ".")
from training.vortex_model import VORTEXModel

DEV = "cuda"
model = VORTEXModel.load_from_checkpoint(
    "models/checkpoints/vortex-epoch=08-val/total=61.1070.ckpt", strict=False).to(DEV).eval()
W = np.load("data/raw/windows.npy").astype(np.float64)
lab = np.load("data/raw/window_regimes_v2.npy")
A = np.load("data/raw/adj_matrices.npy").astype(np.float32)
NF = np.load("data/raw/window_node_features.npy").astype(np.float32)
sp = np.load("data/raw/split.npz")
tr, va = np.asarray(sp["train_idx"]), np.asarray(sp["val_idx"])
sc = np.load("data/raw/scaler.npz")
X = (W - sc["mean"]) / sc["std"]
rng = np.random.default_rng(7)


def to_dev(a):
    return torch.nan_to_num(torch.from_numpy(a), nan=0.0, posinf=1.0, neginf=0.0).to(DEV)


with torch.no_grad():
    mu_all, lv_all, _ = model.sde_model.encode_stats(to_dev(X[tr].astype(np.float32)))
    MU, LV = mu_all.cpu().numpy(), lv_all.cpu().numpy()
    mu_pool = {rg: MU[lab[tr] == rg] for rg in (0, 1)}
    lv_pool = {rg: LV[lab[tr] == rg] for rg in (0, 1)}
    idx = {rg: rng.integers(0, len(mu_pool[rg]), 500) for rg in (0, 1)}
    z = np.concatenate([(mu_pool[rg][idx[rg]] + rng.normal(0, 1, size=(500, 64)) *
                         np.exp(0.5 * lv_pool[rg][idx[rg]])) for rg in (1, 0)])
    grows = np.tile(np.asarray(sp["test_idx"])[:500], 2)
    ne, _ = model.gat(to_dev(NF[grows].mean(axis=1)), to_dev(A[grows]))
    _, zp = model.sde_model.decode_from_z0(ne.mean(dim=1), torch.from_numpy(z.astype(np.float32)).to(DEV))
    G = model.sde_model.emission_sample(zp).cpu().numpy()


def nn_mean(gen, ref):
    g = torch.from_numpy(np.asarray(gen, dtype=np.float32).reshape(len(gen), -1)).to(DEV)
    ref = np.asarray(ref, dtype=np.float32)
    out = []
    for i in range(0, len(ref), 500):
        r = torch.from_numpy(ref[i:i + 500].reshape(len(ref[i:i + 500]), -1)).to(DEV)
        out.append(torch.cdist(g, r).min(dim=1).values.cpu().numpy())
    return float(np.concatenate(out).mean())


print("NN gen-to-train:", round(nn_mean(G, X[tr]), 2),
      "val-real-to-train:", round(nn_mean(X[va], X[tr]), 2))
print("DONE-NN")
