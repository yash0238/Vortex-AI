"""Lane 1 preflight: repro check (flags off) + lam_corr scale probe."""
import numpy as np
import torch
import sys
sys.path.insert(0, ".")
from training.vortex_model import VORTEXModel

DEV = "cuda"
torch.manual_seed(0)
m = VORTEXModel.load_from_checkpoint(
    "models/checkpoints/vortex-epoch=07-val/total=57.4987.ckpt", strict=False).to(DEV).eval()
print("flags:", m.lam_corr, m.vol_dyn, m.emission, m.rank_K)
W = np.load("data/raw/windows.npy").astype(np.float32)
A = np.load("data/raw/adj_matrices.npy").astype(np.float32)
NF = np.load("data/raw/window_node_features.npy").astype(np.float32)
lab = np.load("data/raw/window_regimes_v2.npy")
sp = np.load("data/raw/split.npz")
va = np.asarray(sp["val_idx"])
sc = np.load("data/raw/scaler.npz")


def to_dev(a):
    return torch.nan_to_num(torch.from_numpy(a), nan=0.0, posinf=1.0, neginf=0.0).to(DEV)


with torch.no_grad():
    xa = to_dev(((W[va].astype(np.float64) - sc["mean"]) / sc["std"]).astype(np.float32))
    rh, _, _ = m(xa, to_dev(A[va]), to_dev(NF[va].mean(axis=1)))
    print("repro val NLL:", round(float(m.sde_model.emission_nll(m._last_zs, xa)), 3), "(expect ~57.50)")
m.train()
xa = to_dev(((W[va][:64].astype(np.float64) - sc["mean"]) / sc["std"]).astype(np.float32))
rg = torch.from_numpy(lab[va][:64]).to(DEV)
rh, Al, zp = m(xa, to_dev(A[va][:64]), to_dev(NF[va][:64].mean(axis=1)))
nll = m.sde_model.emission_nll(m._last_zs, xa)
rs = m.sde_model.emission_sample_corr(m._last_zs)
lc = m.correlation_loss(rs, rg)
print(f"epoch-0 probe: NLL={float(nll):.2f} NLL/60={float(nll) / 60:.3f} Lcorr_raw={float(lc):.5f} "
      f"weighted@1.0={float(lc):.5f}")
print("DONE-PREFLIGHT")
