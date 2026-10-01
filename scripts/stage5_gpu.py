"""Stage 5 diagnosis on GPU: sanity rerun, GRU-AE control, eval mode, sigma=0, profile."""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, ".")
from models.gat_encoder import DynamicGATEncoder, pool_graph_embedding
from models.neural_sde import LatentSDEModel

SEED = 123
DEV = "cuda"
torch.manual_seed(SEED)
np.random.seed(SEED)
sc = np.load("data/raw/scaler.npz")
mu, sd = sc["mean"], sc["std"]
W = (((np.load("data/raw/windows.npy").astype(np.float64) - mu) / sd)).astype(np.float32)
A = np.load("data/raw/adj_matrices.npy").astype(np.float32)
NF = np.load("data/raw/window_node_features.npy").astype(np.float32)
sp = np.load("data/raw/split.npz")
tr = np.asarray(sp["train_idx"])
lab = np.load("data/raw/window_regimes_v2.npy")
rng = np.random.default_rng(0)
sel = np.concatenate([rng.choice(np.where(lab[tr] == 1)[0], 8, replace=False),
                      rng.choice(np.where(lab[tr] == 0)[0], 8, replace=False)])
x = torch.from_numpy(W[tr][sel]).to(DEV)
adj = torch.nan_to_num(torch.from_numpy(A[tr][sel]), nan=0.0, posinf=1.0, neginf=0.0).to(DEV)
nf = torch.nan_to_num(torch.from_numpy(NF[tr][sel].mean(axis=1)), nan=0.0).to(DEV)
print("devices:", "batch", x.device, "| T=60 N=47 | batch 8 crisis + 8 normal, seed 123/123")


def show(tag, params, training):
    n_opt = sum(p.numel() for p in params)
    print(f"{tag}: z=64 gru_hidden=64 decoder=Linear(64->47)x1 mode={'train' if training else 'eval'} "
          f"params in optimizer={n_opt} clip=1.0")


def run_loop(tag, fwd, params, steps=300, lr=1e-3):
    opt = torch.optim.Adam(params, lr=lr)
    t0 = time.time()
    torch.cuda.synchronize()
    for step in range(1, steps + 1):
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        opt.zero_grad()
        r_hat = fwd()
        loss = F.mse_loss(r_hat, x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 20 == 0 or step == 1:
            torch.cuda.synchronize()
            print(f"  {tag} step {step}: loss={loss.item():.5f} finite={bool(torch.isfinite(loss))}")
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"  {tag} final={loss.item():.5f} {dt / steps:.4f}s/step")
    return float(loss.item()), dt / steps


# B0 sanity: original real model on GPU
gat = DynamicGATEncoder().to(DEV)
sde = LatentSDEModel(n_stocks=47).to(DEV)
gat.train()
sde.train()
print("model device:", next(gat.parameters()).device, next(sde.parameters()).device)
with torch.no_grad():
    r0, _ = sde(x, pool_graph_embedding(gat(nf, adj)[0]))
    print(f"step0: r_hat std={r0.std().item():.4f} target std={x.std().item():.4f}")
opt = torch.optim.Adam(list(gat.parameters()) + list(sde.parameters()), lr=1e-3)
show("sanity", list(gat.parameters()) + list(sde.parameters()), True)
l1, s1 = run_loop("sanity", lambda: sde(x, pool_graph_embedding(gat(nf, adj)[0]))[0],
                  list(gat.parameters()) + list(sde.parameters()))
print(f"sanity vs CPU 1.14906: diff={abs(l1 - 1.14906):.5f} ({'OK' if abs(l1 - 1.14906) < 0.02 else 'DEVICE BUG'})")
with torch.no_grad():
    rE, _ = sde(x, pool_graph_embedding(gat(nf, adj)[0]))
    print(f"step300: r_hat std={rE.std().item():.4f} target std={x.std().item():.4f}")


# B2 control: plain GRU autoencoder, no SDE/GAT
class GRUAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.GRU(47, 64, batch_first=True)
        self.dec = nn.Linear(64, 47)

    def forward(self, x):
        _, h = self.enc(x)
        z = h.squeeze(0)
        return self.dec(z.unsqueeze(1).expand(-1, 60, -1))


gru = GRUAE().to(DEV)
gru.train()
show("control-gruAE", list(gru.parameters()), True)
l2, s2 = run_loop("control", lambda: gru(x), list(gru.parameters()))

# B3 eval mode (dropout off)
gat2 = DynamicGATEncoder().to(DEV)
sde2 = LatentSDEModel(n_stocks=47).to(DEV)
# note: fresh inits differ from sanity run; this isolates train-vs-eval dynamics, not init
gat2.eval()
sde2.eval()
show("evalmode", list(gat2.parameters()) + list(sde2.parameters()), False)
l3, s3 = run_loop("evalmode", lambda: sde2(x, pool_graph_embedding(gat2(nf, adj)[0]))[0],
                  list(gat2.parameters()) + list(sde2.parameters()))

# B4 sigma = 0 (deterministic ODE)
gat4 = DynamicGATEncoder().to(DEV)
sde4 = LatentSDEModel(n_stocks=47).to(DEV)
gat4.train()
sde4.train()
sde4.sde.g = lambda t, y: torch.zeros_like(y)
show("sigma0", list(gat4.parameters()) + list(sde4.parameters()), True)
l4, s4 = run_loop("sigma0", lambda: sde4(x, pool_graph_embedding(gat4(nf, adj)[0]))[0],
                  list(gat4.parameters()) + list(sde4.parameters()))

# B5 profile of sanity final output
with torch.no_grad():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    rF, _ = sde(x, pool_graph_embedding(gat(nf, adj)[0]))
mse_t = ((rF - x) ** 2).mean(dim=(0, 2)).cpu().numpy()
print("per-timestep MSE buckets:", [round(float(mse_t[i:i + 10].mean()), 4) for i in range(0, 60, 10)])
tm = x.mean(dim=2, keepdim=True)
sm = x.mean(dim=1, keepdim=True)
print("var explained by time-means:", round(float(1 - ((x - tm) ** 2).mean() / x.var()), 4))
print("var explained by stock-means:", round(float(1 - ((x - sm) ** 2).mean() / x.var()), 4))
print(f"SUMMARY sanity={l1:.4f} control={l2:.4f} evalmode={l3:.4f} sigma0={l4:.4f}")
print(f"TIMING s/step sanity={s1:.4f} (cpu 0.128) control={s2:.4f}; proj epoch bs16 gpu={s1 * (2021 / 16):.0f}s")
