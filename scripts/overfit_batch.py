"""Stage 5: overfit one fixed batch, recon only. Fixed Brownian seed via torch.manual_seed each step."""
import time
import numpy as np
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, ".")
from models.gat_encoder import DynamicGATEncoder, pool_graph_embedding
from models.neural_sde import LatentSDEModel

SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = "cpu"
sc = np.load("data/raw/scaler.npz")
mu, sd = sc["mean"], sc["std"]
W = (((np.load("data/raw/windows.npy").astype(np.float64) - mu) / sd)).astype(np.float32)
A = np.load("data/raw/adj_matrices.npy").astype(np.float32)
NF = np.load("data/raw/window_node_features.npy").astype(np.float32)
sp = np.load("data/raw/split.npz")
tr = np.asarray(sp["train_idx"])
lab = np.load("data/raw/window_regimes_v2.npy")
rng = np.random.default_rng(0)
cr = rng.choice(np.where(lab[tr] == 1)[0], 8, replace=False)
no = rng.choice(np.where(lab[tr] == 0)[0], 8, replace=False)
sel = np.concatenate([cr, no])
x = torch.from_numpy(W[tr][sel]).to(DEVICE)
adj = torch.nan_to_num(torch.from_numpy(A[tr][sel]), nan=0.0, posinf=1.0, neginf=0.0).to(DEVICE)
nf = torch.nan_to_num(torch.from_numpy(NF[tr][sel].mean(axis=1)), nan=0.0).to(DEVICE)
print("init z-MSE (predict-zero):", round(float((x ** 2).mean()), 4))

gat = DynamicGATEncoder().to(DEVICE)
sde = LatentSDEModel(n_stocks=x.shape[-1]).to(DEVICE)
opt = torch.optim.Adam(list(gat.parameters()) + list(sde.parameters()), lr=1e-3)
t0 = time.time()
for step in range(1, 301):
    torch.manual_seed(SEED)  # fixed Brownian increments: deterministic SDE solve
    opt.zero_grad()
    node_embs, _ = gat(nf, adj)
    r_hat, _ = sde(x, pool_graph_embedding(node_embs))
    loss = F.mse_loss(r_hat, x)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(gat.parameters()) + list(sde.parameters()), 1.0)
    opt.step()
    if step % 20 == 0:
        ok = bool(torch.isfinite(loss)) and bool(torch.isfinite(r_hat).all())
        gs = []
        for m in [gat, sde.encoder, sde.sde, sde.decoder]:
            g = torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None])
            gs.append(round(float(g.norm()), 4))
        print(f"step {step}: loss={float(loss):.5f} finite={ok} grads(gat,enc,sde,dec)={gs}")
dt = time.time() - t0
print(f"fixed-seed final MSE: {float(loss):.5f} | {dt / 300:.3f}s per step")
with torch.no_grad():
    torch.manual_seed(999)  # fresh Brownian seed
    node_embs, _ = gat(nf, adj)
    r2, _ = sde(x, pool_graph_embedding(node_embs))
    print("fresh-seed final MSE:", round(float(F.mse_loss(r2, x)), 5))
    node_embs2, _ = gat(nf, torch.rand_like(adj))
    r3, _ = sde(x, pool_graph_embedding(node_embs2))
    print("adj-perturbation changes output:", bool((r3 - r2).abs().max() > 1e-6),
          "max abs diff:", round(float((r3 - r2).abs().max()), 6))
print(f"projected s/epoch bs16: {dt / 300 * (2021 / 16):.0f}s")
