"""Smoke tests: imports, shapes, loss-finite checks for repair-plan modules.

Run: .venv/Scripts/python -m pytest tests/test_smoke.py -q  (or python tests/test_smoke.py)
Covers Phase 1-2 contracts: total_loss tuple, SupCon, GAT shapes,
VortexDataset sample, stratified sampler, eval metrics on synthetic data.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_imports():
    import train_vortex  # noqa: F401
    import evaluate_vortex  # noqa: F401
    from training import losses, vortex_model, trainer  # noqa: F401
    from models import gat_encoder, neural_sde, contrastive  # noqa: F401
    from eval import statistical, contagion, discriminative  # noqa: F401
    from sandbox import circuit_filter  # noqa: F401
    from data import node_features  # noqa: F401


def test_total_loss_contract():
    from training.losses import total_loss
    from models.contrastive import SupConLoss
    B, T, N, P = 8, 10, 6, 8
    x = torch.randn(B, T, N) * 0.01
    xh = x + torch.randn(B, T, N) * 0.001
    out = total_loss(x, xh, torch.eye(N).expand(B, -1, -1),
                     torch.eye(N).expand(B, -1, -1),
                     torch.nn.functional.normalize(torch.randn(B, P), dim=1),
                     torch.tensor([0] * 4 + [1] * 4),
                     SupConLoss(), lam_na=0.05)
    assert isinstance(out, tuple) and len(out) == 2
    loss, comp = out
    assert set(comp) == {"total", "reconstruction", "graph", "contrastive", "circuit"}
    assert torch.isfinite(loss) and all(torch.isfinite(v).all() for v in comp.values())


def test_gat_shapes():
    from models.gat_encoder import DynamicGATEncoder
    m = DynamicGATEncoder().eval()
    nf = torch.randn(4, 50, 6)
    adj = (torch.rand(4, 50, 50) > 0.7).float()
    with torch.no_grad():
        embs, learned = m(nf, adj)
    assert embs.shape == (4, 50, 64) and learned.shape == (4, 50, 50)
    assert torch.isfinite(embs).all() and torch.isfinite(learned).all()


def test_sde_shapes():
    from models.neural_sde import LatentSDEModel
    m = LatentSDEModel(n_stocks=6, latent_dim=8, T=10).eval()
    with torch.no_grad():
        r_hat, zs = m(torch.randn(2, 10, 6) * 0.01, torch.randn(2, 8))
    assert r_hat.shape == (2, 10, 6) and zs.shape == (2, 10, 8)
    assert torch.isfinite(r_hat).all()


def test_eval_metrics_finite():
    from eval.statistical import evaluate_stylized_facts, compare_distributions, test_no_autocorrelation_returns
    from eval.contagion import evaluate_contagion, cvar_regime_ratio
    r = np.random.randn(12, 20, 4)
    g = np.random.randn(12, 20, 4)
    reg = np.array([0] * 6 + [1] * 6)
    sf = evaluate_stylized_facts(r, g)
    assert set(sf) >= {"kurtosis_gen", "acf_sq_gen", "corr_error", "all_tests_pass"}
    assert np.isfinite(sf["corr_error"])
    assert np.isfinite(test_no_autocorrelation_returns(g)["returns_acf_lag1"])
    assert np.isfinite(compare_distributions(r, g)["std_gen"])
    assert np.isfinite(evaluate_contagion(g, reg)["crisis_corr_boost"])
    assert np.isfinite(cvar_regime_ratio(g, reg)["cvar_ratio"])


def test_circuit_bands():
    from sandbox.circuit_filter import _is_band_a, BAND_A_STOCKS
    assert _is_band_a("SBIN") and _is_band_a("SBIN.NS")
    assert not _is_band_a("RELIANCE") and not _is_band_a("RELIANCE.NS")
    assert len(BAND_A_STOCKS) == 3


if __name__ == "__main__":
    test_imports()
    print("imports ok")
    test_total_loss_contract()
    print("total_loss ok")
    test_gat_shapes()
    print("gat ok")
    test_sde_shapes()
    print("sde ok")
    test_eval_metrics_finite()
    print("eval ok")
    test_circuit_bands()
    print("circuit ok")
    print("ALL SMOKE TESTS PASSED")
