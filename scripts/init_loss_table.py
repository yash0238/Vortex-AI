"""Stage 3e: init loss/grad table on one z-space batch + predict-zero baseline."""
import numpy as np
import torch
import sys
sys.path.insert(0, ".")
from models.gat_encoder import DynamicGATEncoder, pool_graph_embedding
from models.neural_sde import LatentSDEModel
from models.contrastive import SupConLoss, ProjectionHead
from training.losses import reconstruction_loss

torch.manual_seed(0)
np.random.seed(0)
sc = np.load("data/raw/scaler.npz")
mu, sd = sc["mean"], sc["std"]
W = np.load("data/raw/windows.npy").astype(np.float64)
Z = ((W - mu) / sd).astype(np.float32)
A = np.load("data/raw/adj_matrices.npy").astype(np.float32)
F = np.load("data/raw/window_node_features.npy").astype(np.float32)
sp = np.load("data/raw/split.npz")
tr = sp["train_idx"]
print("z-space global mean/std:", round(float(Z.mean()), 4), round(float(Z.std()), 4))
print("predict-zero z-MSE (train):", round(float((Z[tr] ** 2).mean()), 4))

lab = np.load("data/raw/window_regimes_v2.npy")
cr = np.where(lab[tr] == 1)[0]
no = np.where(lab[tr] == 0)[0]
rng = np.random.default_rng(0)
sel = np.concatenate([rng.choice(cr, 8, replace=False), rng.choice(no, 8, replace=False)])
x = torch.from_numpy(Z[tr][sel])
adj = torch.nan_to_num(torch.from_numpy(A[tr][sel]), nan=0.0, posinf=1.0, neginf=0.0)
nf = torch.nan_to_num(torch.from_numpy(F[tr][sel].mean(axis=1)), nan=0.0)
reg = torch.from_numpy(lab[tr][sel]).long()
print("batch regimes:", reg.tolist())

model_gat = DynamicGATEncoder()
model_sde = LatentSDEModel(n_stocks=x.shape[-1])
model_proj = ProjectionHead()
supcon = SupConLoss(temperature=0.07)
for m in [model_gat, model_sde, model_proj]:
    m.train()
node_embs, A_learned = model_gat(nf, adj)
graph_emb = pool_graph_embedding(node_embs)
r_hat, zs = model_sde(x, graph_emb)
z_proj = model_proj(zs)
from training.losses import total_loss
eff_con = 0.1 * (0 + 1) / 20  # warmup epoch 0 of 20
loss, comp = total_loss(x, r_hat, A_learned, adj, z_proj, reg, supcon,
                        lam_rec=1.0, lam_graph=0.1, lam_con=eff_con,
                        lam_na=0.05, lambda_styl=0.1)
print("eff lam_con at epoch 0:", round(eff_con, 4))
for k, v in comp.items():
    print(f"  {k}: {float(v):.6g}")
print("total:", round(float(loss), 6))
loss.backward()
for name, mod in [("gat", model_gat), ("encoder", model_sde.encoder),
                  ("sde", model_sde.sde), ("decoder", model_sde.decoder),
                  ("proj", model_proj)]:
    g = torch.cat([p.grad.flatten() for p in mod.parameters() if p.grad is not None]) if any(
        p.grad is not None for p in mod.parameters()) else None
    print(f"  grad {name}: {'none' if g is None else round(float(g.norm()), 6)} (finite: {bool(torch.isfinite(g).all()) if g is not None else '-'})")
print("loss finite:", bool(torch.isfinite(loss)))
print("r_hat std (z-space):", round(float(r_hat.std()), 4))
