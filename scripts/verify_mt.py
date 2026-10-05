import sys
sys.path.insert(0, ".")
import torch
import numpy as np
from models.neural_sde import LatentSDEModel, _SharedMixMultivariateT

torch.manual_seed(0)
np.random.seed(0)
m = LatentSDEModel(n_stocks=47, latent_dim=64, emission="mt", rank_K=8)
x = torch.randn(4, 60, 47)
ge = torch.randn(4, 64)
rh, zs = m(x, ge)
print("fwd ok", tuple(rh.shape))
print("nll:", round(float(m.emission_nll(zs, x)), 3))
s = m.emission_sample(zs)
print("sample ok", tuple(s.shape), bool(torch.isfinite(s).all()))

# verify vs scipy on small random case
from scipy import stats as sps
torch.manual_seed(1)
N, K = 5, 2
loc = torch.randn(3, N)
B = torch.randn(3, N, K) * 0.5
d = torch.rand(3, N) + 0.5
nu = torch.tensor(7.0)
mine = _SharedMixMultivariateT(loc, B, d, nu)
xx = torch.randn(3, N)
lp_mine = mine.log_prob(xx).detach().numpy()
lp_ref = np.array([sps.multivariate_t.logpdf(xx[i].numpy(), loc[i].numpy(),
                  (B[i] @ B[i].T + torch.diag(d[i])).numpy(), 7.0) for i in range(3)])
print("log_prob max abs err vs scipy:", float(np.abs(lp_mine - lp_ref).max()))
# covariance check: E[(r-m)(r-m)^T] should equal Sigma
big = torch.cat([mine.rsample().unsqueeze(0) for _ in range(4000)], dim=0)
emp = torch.cov((big - loc.mean(0)).reshape(-1, N).T)
tgt = (B @ B.transpose(-1, -2) + torch.diag_embed(d)).mean(0)
print("cov max abs err:", float((emp - tgt).abs().max()))
print("DONE-VERIFY")
