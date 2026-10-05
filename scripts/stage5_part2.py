"""Stage 5 part 2: B3 gat-dropout-off, B4 sigma=0, B5 profile (GPU, fixed seeds)."""
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


def run_loop(tag, fwd, params, steps=300):
    opt = torch.optim.Adam(params, lr=1e-3)
    for step in range(1, steps + 1):
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        opt.zero_grad()
        loss = F.mse_loss(fwd(), x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 20 == 0 or step == 1:
            print(f"  {tag} step {step}: loss={loss.item():.5f}", flush=True)
    return float(loss.item())


# B3: GAT convs in eval (dropout off), GRU stays train so cudnn backward works
gat3 = DynamicGATEncoder().to(DEV)
sde3 = LatentSDEModel(n_stocks=47).to(DEV)
gat3.train()
sde3.train()
gat3.gat1.eval()
gat3.gat2.eval()
l3 = run_loop("dropout-off", lambda: sde3(x, pool_graph_embedding(gat3(nf, adj)[0]))[0],
              list(gat3.parameters()) + list(sde3.parameters()))
print("B3 dropout-off final:", round(l3, 5))

# B4: sigma = 0
gat4 = DynamicGATEncoder().to(DEV)
sde4 = LatentSDEModel(n_stocks=47).to(DEV)
gat4.train()
sde4.train()
sde4.sde.g = lambda t, y: torch.zeros_like(y)
l4 = run_loop("sigma0", lambda: sde4(x, pool_graph_embedding(gat4(nf, adj)[0]))[0],
              list(gat4.parameters()) + list(sde4.parameters()))
print("B4 sigma0 final:", round(l4, 5))

# B5: fresh sanity rerun for profile
gat = DynamicGATEncoder().to(DEV)
sde = LatentSDEModel(n_stocks=47).to(DEV)
gat.train()
sde.train()
run_loop("profile-sanity", lambda: sde(x, pool_graph_embedding(gat(nf, adj)[0]))[0],
         list(gat.parameters()) + list(sde.parameters()))
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
print(f"SUMMARY dropout-off={l3:.4f} sigma0={l4:.4f}")
