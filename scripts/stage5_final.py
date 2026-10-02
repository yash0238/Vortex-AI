"""Stage 5 final: fair GRU-AE control (300k+ params) vs real model lr 3e-3, 1000 steps, GPU."""
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
print("devices: batch on", x.device, "| T=60 N=47")


class FairGRUAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.GRU(47, 128, batch_first=True)
        self.dec = nn.Sequential(nn.Linear(128, 512), nn.ReLU(), nn.Linear(512, 60 * 47))

    def forward(self, x):
        _, h = self.enc(x)
        return self.dec(h.squeeze(0)).view(-1, 60, 47)


def run_loop(tag, fwd, params, steps=1000, lr=1e-3):
    opt = torch.optim.Adam(params, lr=lr)
    t0 = time.time()
    for step in range(1, steps + 1):
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        opt.zero_grad()
        loss = F.mse_loss(fwd(), x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 100 == 0 or step == 1:
            torch.cuda.synchronize()
            print(f"  {tag} step {step}: loss={loss.item():.5f} finite={bool(torch.isfinite(loss))}", flush=True)
    torch.cuda.synchronize()
    return float(loss.item())


gru = FairGRUAE().to(DEV)
gru.train()
nctl = sum(p.numel() for p in gru.parameters())
print(f"control params: {nctl} (expect >300k)")
l_ctl = run_loop("control", lambda: gru(x), list(gru.parameters()))

gat = DynamicGATEncoder().to(DEV)
sde = LatentSDEModel(n_stocks=47).to(DEV)
gat.train()
sde.train()
l_real = run_loop("real-lr3e-3", lambda: sde(x, pool_graph_embedding(gat(nf, adj)[0]))[0],
                  list(gat.parameters()) + list(sde.parameters()), lr=3e-3)
with torch.no_grad():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    rF, _ = sde(x, pool_graph_embedding(gat(nf, adj)[0]))
    print(f"real r_hat std={rF.std().item():.4f} target std={x.std().item():.4f}")
print(f"SUMMARY control={l_ctl:.4f} real={l_real:.4f}")
