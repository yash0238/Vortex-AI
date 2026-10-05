"""Stage 6 post-hoc: val MSE + per-stock gen/real std ratio from best ckpt (GPU)."""
import glob
import numpy as np
import torch
import os
import sys
sys.path.insert(0, ".")
from training.vortex_model import VORTEXModel

DEV = "cuda"
ckpts = sorted(glob.glob("models/checkpoints/**/*.ckpt", recursive=True), key=os.path.getmtime)
def _val(f):
    import re
    m = re.search(r"total=([0-9]+\.[0-9]+)", f)
    return float(m.group(1)) if m else 9e9
best = min([c for c in ckpts if "last" not in os.path.basename(c)], key=_val)
print("ckpts:", len(ckpts), "latest:", os.path.basename(ckpts[-1]))
model = VORTEXModel.load_from_checkpoint(best, strict=False).to(DEV).eval()
print("evaluating:", best)
W = np.load("data/raw/windows.npy").astype(np.float32)
A = np.load("data/raw/adj_matrices.npy").astype(np.float32)
NF = np.load("data/raw/window_node_features.npy").astype(np.float32)
sp = np.load("data/raw/split.npz")
tr, va = np.asarray(sp["train_idx"]), np.asarray(sp["val_idx"])
sc = np.load("data/raw/scaler.npz")
mu = torch.from_numpy(sc["mean"]).to(DEV)
sd = torch.from_numpy(sc["std"]).to(DEV)
xa = torch.nan_to_num(torch.from_numpy(
    ((W[va].astype(np.float64) - sc["mean"]) / sc["std"]).astype(np.float32)), nan=0.0).to(DEV)
ad = torch.nan_to_num(torch.from_numpy(A[va]), nan=0.0, posinf=1.0, neginf=0.0).to(DEV)
nf = torch.nan_to_num(torch.from_numpy(NF[va].mean(axis=1)), nan=0.0).to(DEV)
with torch.no_grad():
    r_hat, _, _ = model(xa, ad, nf)
mse = float(((r_hat - xa) ** 2).mean())
gr = (r_hat * sd + mu).cpu().numpy()
rr = W[va]
ratio = gr.std(axis=(0, 1)) / (rr.std(axis=(0, 1)) + 1e-12)
print(f"val z-MSE: {mse:.4f} (predict-zero 1.0)")
print(f"std ratio gen/real per stock: mean={ratio.mean():.3f} min={ratio.min():.3f} max={ratio.max():.3f}")
print("MSE per epoch from logs:")
import pandas as pd
logs = sorted(glob.glob("runs/*/*/metrics.csv"), key=os.path.getmtime)
df = pd.read_csv(logs[-1])
print("log:", logs[-1])
cols = [c for c in ["epoch", "train/total_epoch", "val/total"] if c in df.columns]
print(df[cols].dropna(how="all").to_string(index=False))
