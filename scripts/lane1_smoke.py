import torch, sys
sys.path.insert(0, ".")
from training.vortex_model import VORTEXModel
m = VORTEXModel(n_stocks=47, emission="mt", rank_K=8, sde_sigma_max=0.75, lam_corr=100.0, vol_dyn=True)
m.train()
x = torch.randn(16, 60, 47)
a = torch.rand(16, 47, 47)
nf = torch.randn(16, 47, 6)
r = torch.tensor([1] * 8 + [0] * 8)
out = m.training_step({"returns": x, "adj_matrix": a, "node_features": nf,
                       "adj_empirical": a, "regimes": r}, 0)
print("train step ok, loss:", round(float(out), 3))
m.eval()
print("eval ok")
g = m.sample_prior_scenarios(a, nf, r)
print("seq sample ok:", tuple(g.shape), bool(__import__("torch").isfinite(g).all()))
print("DONE-SMOKE")
