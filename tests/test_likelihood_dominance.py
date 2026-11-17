"""Likelihood dominance verification (Phase 2).

Verifies the three fixes that balance the loss so fidelity terms are not
decoration: per-asset NLL scaling, beta_max=0.05, and kappa floor at 0.6.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import PropertyMock, patch

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _make_batch(B=2, T=4, N=4, F=6):
    return {
        "returns": torch.randn(B, T, N) * 0.01,
        "adj_matrix": torch.rand(B, N, N),
        "node_features": torch.randn(B, N, F),
        "adj_empirical": torch.rand(B, N, N),
        "regimes": torch.tensor([0, 1]),
    }


# ======================================================================
# 1. _nll_per_asset divides by n_stocks with edge-case guard
# ======================================================================
def test_nll_per_asset_divides_by_n_stocks():
    from training.vortex_model import VORTEXModel

    # n_stocks=5 -> NLL 50 -> 10.0
    model5 = VORTEXModel(n_stocks=5, latent_dim=8, T=4, emission="smt", rank_K=3)
    assert model5._nll_per_asset(torch.tensor(50.0)).item() == 10.0
    assert model5._nll_per_asset(torch.tensor(65.0)).item() == 13.0

    # n_stocks=50 -> NLL 65 -> 1.3 (the v18 observation)
    model50 = VORTEXModel(n_stocks=50, latent_dim=8, T=4, emission="smt", rank_K=3)
    assert model50._nll_per_asset(torch.tensor(65.0)).item() == pytest.approx(1.3, abs=1e-6)

    # Edge-case guard: max(1, n_stocks) prevents division by zero if n_stocks somehow 0
    model0 = VORTEXModel.__new__(VORTEXModel)
    model0.sde_model = type("obj", (), {"n_stocks": 0})()
    assert model0._nll_per_asset(torch.tensor(10.0)).item() == 10.0


def test_training_step_uses_nll_per_asset():
    """training_step must call _nll_per_asset at runtime, not use raw NLL directly."""
    import inspect

    from training.vortex_model import VORTEXModel

    source = inspect.getsource(VORTEXModel.training_step)
    assert "_nll_per_asset" in source, "training_step must use _nll_per_asset"
    assert "self._nll_per_asset(nll)" in source.replace(" ", ""), (
        "training_step must compute loss via _nll_per_asset(nll) + beta*kl"
    )

    # Runtime check: loss returned by training_step equals raw_nll / n_stocks
    # (with beta_max=0 the beta*kl term vanishes, so loss == _nll_per_asset(nll)).
    model = VORTEXModel(
        n_stocks=4,
        latent_dim=4,
        T=4,
        emission="smt",
        rank_K=2,
        beta_max=0.0,
        lam_graph=0.0,
        lam_corr=0.0,
        lam_emission_corr=0.0,
        lambda_styl=0.0,
        vol_dyn=False,
        warmup_epochs=1,
    )
    model.log = lambda *a, **k: None
    model.train()

    batch = _make_batch(B=2, T=4, N=4, F=6)
    loss = model.training_step(batch, 0)

    # Recompute raw NLL the same way training_step does.
    with torch.no_grad():
        lv = model.sde_model.log_vol(batch["returns"])
        raw_nll = model.sde_model.emission_nll(model._last_zs, batch["returns"], batch["regimes"], lv)

    expected = raw_nll / model.sde_model.n_stocks
    assert loss.item() == pytest.approx(expected.item(), abs=1e-6)
    # And it must NOT equal the raw joint NLL (~65 vs ~1.3 at 50 stocks).
    assert loss.item() != pytest.approx(raw_nll.item(), abs=1e-6)


# ======================================================================
# 2. beta_max = 0.05 in config.yaml and propagated through merge_config
# ======================================================================
def test_config_yaml_has_beta_max_005():
    config_path = Path(__file__).resolve().parent.parent / "training" / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    assert cfg["cgnsde"]["beta_max"] == 0.05, f"config.yaml cgnsde.beta_max is {cfg['cgnsde']['beta_max']}, expected 0.05"


def test_merge_config_propagates_beta_max():
    from train_vortex import merge_config

    config = {"cgnsde": {"beta_max": 0.05, "loss_weights": {}}}
    flat = merge_config(config)
    assert flat["beta_max"] == 0.05


def test_eff_beta_reaches_beta_max_at_warmup_end():
    from training.vortex_model import VORTEXModel

    model = VORTEXModel(
        n_stocks=10,
        latent_dim=8,
        T=4,
        emission="smt",
        rank_K=3,
        beta_max=0.05,
        prior_warmup=5,
    )
    # prior_warmup=5, so at epoch 4 (0-indexed) progress = 5/5 = 1.0
    with patch.object(type(model), "current_epoch", new_callable=PropertyMock, return_value=4):
        assert model._eff_beta() == pytest.approx(0.05, abs=1e-9)

    # Before warmup: epoch 0 -> 0.05 * 1/5 = 0.01
    with patch.object(type(model), "current_epoch", new_callable=PropertyMock, return_value=0):
        assert model._eff_beta() == pytest.approx(0.01, abs=1e-9)


# ======================================================================
# 3. kappa is clamped at 0.6 in models/neural_sde.py
# ======================================================================
def test_kappa_clamped_at_06():
    from models.neural_sde import LatentSDEModel

    # Negative kappa_raw -> sigmoid < 0.5 -> clamped to 0.6
    model_neg = LatentSDEModel(n_stocks=6, latent_dim=8, T=4, emission="smt", rank_K=3)
    with torch.no_grad():
        model_neg.kappa_raw.fill_(-10.0)  # sigmoid(-10) ~ 0.000045
    rho, kappa = model_neg._rho_kappa(torch.zeros(1))
    assert kappa.item() == pytest.approx(0.6, abs=1e-6)

    model_neg2 = LatentSDEModel(n_stocks=6, latent_dim=8, T=4, emission="smt", rank_K=3)
    with torch.no_grad():
        model_neg2.kappa_raw.fill_(-1.0)  # sigmoid(-1) ~ 0.269
    rho, kappa = model_neg2._rho_kappa(torch.zeros(1))
    assert kappa.item() == pytest.approx(0.6, abs=1e-6)

    # kappa_raw = 0.0 -> sigmoid(0) = 0.5 -> clamped to 0.6 (default init)
    model_zero = LatentSDEModel(n_stocks=6, latent_dim=8, T=4, emission="smt", rank_K=3)
    with torch.no_grad():
        model_zero.kappa_raw.fill_(0.0)
    rho, kappa = model_zero._rho_kappa(torch.zeros(1))
    assert kappa.item() == pytest.approx(0.6, abs=1e-6), "default init kappa_raw=0 must clamp to 0.6"

    # Positive kappa_raw -> sigmoid > 0.6 -> unclamped
    model_pos = LatentSDEModel(n_stocks=6, latent_dim=8, T=4, emission="smt", rank_K=3)
    with torch.no_grad():
        model_pos.kappa_raw.fill_(2.0)  # sigmoid(2) ~ 0.88 > 0.6
    rho, kappa = model_pos._rho_kappa(torch.zeros(1))
    expected = torch.sigmoid(torch.tensor(2.0)).item()
    assert kappa.item() == pytest.approx(expected, abs=1e-6)
    assert kappa.item() > 0.6


def test_emission_nll_vol_receives_kappa_ge_06():
    """Runtime emission_nll_vol must receive kappa >= 0.6."""
    from models.neural_sde import LatentSDEModel

    model = LatentSDEModel(n_stocks=6, latent_dim=8, T=12, emission="smt", rank_K=3)
    model.eval()
    with torch.no_grad():
        model.kappa_raw.fill_(-10.0)  # would be ~0 without clamp
    zs = torch.randn(2, 12, 8) * 0.1
    regime = torch.tensor([0, 1])
    log_vol = torch.zeros(2)
    x = torch.randn(2, 12, 6) * 0.01

    # This internally calls _rho_kappa; if clamp worked, no error / sensible NLL
    nll = model.emission_nll_vol(zs, x, regime, log_vol)
    assert torch.isfinite(nll)

    # Also verify by spying on _rho_kappa
    calls = []

    original_rho_kappa = model._rho_kappa

    def spy_rho_kappa(like):
        rho, kappa = original_rho_kappa(like)
        calls.append(kappa.item())
        return rho, kappa

    model._rho_kappa = spy_rho_kappa
    _ = model.emission_nll_vol(zs, x, regime, log_vol)
    assert all(k >= 0.6 - 1e-6 for k in calls), f"kappa values {calls} not all >= 0.6"


def test_emission_sample_seq_receives_kappa_ge_06():
    """Runtime emission_sample_seq must receive kappa >= 0.6."""
    from models.neural_sde import LatentSDEModel

    model = LatentSDEModel(n_stocks=6, latent_dim=8, T=12, emission="smt", rank_K=3)
    model.eval()
    with torch.no_grad():
        model.kappa_raw.fill_(-10.0)
    zs = torch.randn(2, 12, 8) * 0.1
    regime = torch.tensor([0, 1])
    log_vol = torch.zeros(2)

    calls = []

    original_rho_kappa = model._rho_kappa

    def spy_rho_kappa(like):
        rho, kappa = original_rho_kappa(like)
        calls.append(kappa.item())
        return rho, kappa

    model._rho_kappa = spy_rho_kappa
    out = model.emission_sample_seq(zs, regime, log_vol)
    assert out.shape == (2, 12, 6)
    assert all(k >= 0.6 - 1e-6 for k in calls), f"kappa values {calls} not all >= 0.6"


# ======================================================================
# Integration sanity: combined effect brings loss to same order
# ======================================================================
def test_combined_loss_scale_is_balanced():
    """With the fixes, nll_per_asset ~1.3 and fidelity weights ~1-20 are comparable."""
    from training.vortex_model import VORTEXModel
    from unittest.mock import PropertyMock, patch

    model = VORTEXModel(
        n_stocks=50,
        latent_dim=64,
        T=60,
        emission="smt",
        rank_K=8,
        beta_max=0.05,
        prior_warmup=20,
        lambda_styl=1.0,
        lam_emission_corr=20.0,
        warmup_epochs=20,
        vol_dyn=True,
        stylized_sampler="seq",
    )
    # Simulate: raw NLL ~65, per-asset ~1.3; fidelity terms O(0.01-0.5) * weights 1-20 -> same order
    raw_nll = torch.tensor(65.0)
    nll_pa = model._nll_per_asset(raw_nll)
    assert 1.0 <= nll_pa.item() <= 2.0, f"per-asset NLL {nll_pa.item():.3f} not in expected range"

    # beta_max=0.05 * KL (typically ~0.5-2) -> 0.025-0.1, small but non-zero
    with patch.object(type(model), "current_epoch", new_callable=PropertyMock, return_value=19):
        beta = model._eff_beta()
        assert beta == pytest.approx(0.05, abs=1e-6)

    # kappa floor ensures vol clustering can be learned
    model.sde_model.kappa_raw.data.fill_(-10.0)
    rho, kappa = model.sde_model._rho_kappa(torch.zeros(1))
    assert kappa.item() == pytest.approx(0.6, abs=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
