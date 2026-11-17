"""Defect 2: training/validation/generation must use the same emission sampler.

The stylized-facts objective used to optimise ``emission_sample`` (one joint draw
over all 60 steps, no volatility feedback) while every generative entry point
uses ``emission_sample_seq`` (step-by-step with the EGARCH-style vol recursion).
The temporal recursion that produces ACF(r^2) and the tail statistics therefore
received no gradient from the fidelity objective.

These tests pin the alignment in three ways:

1. source inspection -- the stylized-loss branches must consult the
   ``stylized_sampler`` switch and be able to reach ``emission_sample_seq``;
2. behavioural -- ``training_step`` / ``validation_step`` must actually call the
   sampler selected by the switch (not merely mention it);
3. contract -- ``emission_sample_seq`` must keep the signature and the
   sequential recursion generation relies on.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_stylized_loss_uses_same_sampler_as_generation():
    from training.vortex_model import VORTEXModel

    # training_step uses emission_sample_seq for stylized loss when configured
    source = inspect.getsource(VORTEXModel.training_step)
    assert "emission_sample_seq" in source, (
        "training_step must use emission_sample_seq for stylized loss "
        "when stylized_sampler='seq'"
    )
    assert "stylized_sampler" in source, (
        "training_step must check stylized_sampler config"
    )

    # validation_step uses emission_sample_seq when configured
    source = inspect.getsource(VORTEXModel.validation_step)
    assert "emission_sample_seq" in source, (
        "validation_step must use emission_sample_seq when stylized_sampler='seq'"
    )
    assert "stylized_sampler" in source, (
        "validation_step must check stylized_sampler config"
    )

    # generation uses emission_sample_seq
    source = inspect.getsource(VORTEXModel.sample_regime_scenarios)
    assert "emission_sample_seq" in source, (
        "generation must use emission_sample_seq"
    )

    # verify emission_sample_seq exists and is callable
    from models.neural_sde import LatentSDEModel

    assert hasattr(LatentSDEModel, "emission_sample_seq")
    sig = inspect.signature(LatentSDEModel.emission_sample_seq)
    assert "zs" in sig.parameters
    assert "regime" in sig.parameters
    assert "log_vol" in sig.parameters


def test_emission_sample_seq_is_sequential_with_vol_feedback():
    """The sequential sampler must feed the sampled return back into vol."""
    from models.neural_sde import LatentSDEModel

    source = inspect.getsource(LatentSDEModel.emission_sample_seq)
    # The recursion is what distinguishes it from emission_sample.
    assert "for t in range(T)" in source, "expected a sequential loop over time"
    assert "torch.stack" in source, "expected per-step draws stacked to (B, T, N)"
    # Vol state must be updated from the realised draw, not the target.
    assert "v = (rho * v + (1 - rho)" in source, "expected the vol recursion update"


def _tiny_model(stylized_sampler: str):
    """4 assets / 5 steps is enough to exercise the loss wiring."""
    from training.vortex_model import VORTEXModel

    torch.manual_seed(0)
    model = VORTEXModel(
        n_stocks=4,
        T=5,
        in_feats=6,
        latent_dim=8,
        proj_dim=4,
        hidden_dim=8,
        gat_heads=2,
        tau=0.07,
        lr=1e-3,
        lam_rec=1.0,
        lam_graph=0.0,
        lam_con=0.0,
        lam_na=0.0,
        beta_max=0.0,
        prior_warmup=1,
        lam_var=0.0,
        lam_corr=0.0,
        corr_ramp=1,
        lam_emission_corr=0.0,
        vol_dyn=True,
        emission="smt",
        rank_K=0,
        sde_sigma_max=1.0,
        warmup_epochs=1,
        lambda_styl=1.0,
        gat_dropout=0.1,
        stylized_sampler=stylized_sampler,
    )
    # `self.log` needs a Trainer; the sampler wiring does not.
    model.log = lambda *a, **k: None
    return model


def _batch():
    B, T, N, F = 2, 5, 4, 6
    torch.manual_seed(1)
    return {
        "returns": torch.randn(B, T, N) * 0.01,
        "adj_matrix": torch.eye(N).expand(B, -1, -1).clone(),
        "node_features": torch.randn(B, N, F),
        "adj_empirical": torch.eye(N).expand(B, -1, -1).clone(),
        "regimes": torch.tensor([0, 1]),
    }


@pytest.mark.parametrize(
    "sampler, expected, forbidden",
    [("seq", "emission_sample_seq", "emission_sample"),
     ("single", "emission_sample", "emission_sample_seq")],
)
def test_training_step_calls_configured_sampler(sampler, expected, forbidden):
    """With lambda_styl > 0 the stylized branch must use the configured sampler."""
    model = _tiny_model(sampler)
    calls = []

    def record(name):
        def _fn(zs, regime=None, log_vol=None):
            calls.append(name)
            return torch.zeros_like(_batch()["returns"][: zs.shape[0]])
        return _fn

    model.sde_model.emission_sample = record("emission_sample")
    model.sde_model.emission_sample_seq = record("emission_sample_seq")

    # `_last_zs` is set by forward; the patched samplers ignore its content.
    model.training_step(_batch(), 0)

    assert expected in calls, f"{sampler!r} arm must call {expected}, calls={calls}"
    assert forbidden not in calls, (
        f"{sampler!r} arm must NOT call {forbidden} for the stylized loss, "
        f"calls={calls}"
    )


@pytest.mark.parametrize(
    "sampler, expected, forbidden",
    [("seq", "emission_sample_seq", "emission_sample"),
     ("single", "emission_sample", "emission_sample_seq")],
)
def test_validation_step_calls_configured_sampler(sampler, expected, forbidden):
    model = _tiny_model(sampler)
    calls = []

    def record(name):
        def _fn(zs, regime=None, log_vol=None):
            calls.append(name)
            return torch.zeros_like(_batch()["returns"][: zs.shape[0]])
        return _fn

    model.sde_model.emission_sample = record("emission_sample")
    model.sde_model.emission_sample_seq = record("emission_sample_seq")

    # batch_idx=1 skips the every-5-epoch gate probe.
    model.validation_step(_batch(), 1)

    assert expected in calls, f"{sampler!r} arm must call {expected}, calls={calls}"
    assert forbidden not in calls, (
        f"{sampler!r} arm must NOT call {forbidden} for the stylized loss, "
        f"calls={calls}"
    )


def test_emission_sample_seq_returns_btN_and_reacts_to_volatility():
    """Shape contract plus a sanity check that log_vol actually moves the draws."""
    from models.neural_sde import LatentSDEModel

    torch.manual_seed(0)
    sde = LatentSDEModel(n_stocks=6, latent_dim=8, T=12, emission="smt", rank_K=2)
    zs = torch.randn(3, 12, 8) * 0.1
    regime = torch.tensor([0, 1, 0])

    torch.manual_seed(2)
    low = sde.emission_sample_seq(zs, regime, torch.full((3,), -1.0))
    torch.manual_seed(2)
    high = sde.emission_sample_seq(zs, regime, torch.full((3,), +1.0))

    assert low.shape == (3, 12, 6)
    assert high.shape == (3, 12, 6)
    assert torch.isfinite(low).all() and torch.isfinite(high).all()
    # Volatility conditioning is exp(c * log_vol) on the loadings, exp(2 c) on
    # the variance: a higher vol state must widen the emitted window.
    assert high.std() > low.std(), "emission scale must increase with log_vol"


def test_emission_sample_seq_receives_gradients_when_seq_sampler():
    """Verify emission_sample_seq receives non-zero gradients when stylized_sampler='seq'."""
    model_seq = _tiny_model("seq")
    batch = _batch()
    captured = {}
    orig_seq = model_seq.sde_model.emission_sample_seq

    def spy_seq(*args, **kwargs):
        out = orig_seq(*args, **kwargs)
        out.retain_grad()
        captured["output"] = out
        return out

    model_seq.sde_model.emission_sample_seq = spy_seq
    loss_seq = model_seq.training_step(batch, 0)
    loss_seq.backward()

    assert "output" in captured, (
        "emission_sample_seq was not called in training_step when stylized_sampler='seq'"
    )
    assert captured["output"].grad is not None, (
        "emission_sample_seq output did not receive gradient"
    )
    grad_norm = captured["output"].grad.norm().item()
    assert grad_norm > 0.0, (
        f"expected non-zero gradient on emission_sample_seq output, got {grad_norm}"
    )

    # Also verify that when stylized_sampler='single', emission_sample_seq is NOT called
    model_single = _tiny_model("single")
    captured_single = {}
    orig_seq_single = model_single.sde_model.emission_sample_seq

    def spy_single(*args, **kwargs):
        out = orig_seq_single(*args, **kwargs)
        captured_single["output"] = out
        return out

    model_single.sde_model.emission_sample_seq = spy_single
    loss_single = model_single.training_step(batch, 0)
    loss_single.backward()
    assert "output" not in captured_single, (
        "emission_sample_seq must not be called when stylized_sampler='single'"
    )



def test_config_stylized_sampler_is_seq():
    """Verify training/config.yaml sets cgnsde.stylized_sampler to 'seq'."""
    import yaml
    from pathlib import Path

    config_path = Path(__file__).resolve().parent.parent / "training" / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    assert "cgnsde" in cfg, "config.yaml must have a 'cgnsde' section"
    assert "stylized_sampler" in cfg["cgnsde"], "cgnsde.stylized_sampler must be set"
    assert cfg["cgnsde"]["stylized_sampler"] == "seq", (
        f"cgnsde.stylized_sampler must be 'seq', got {cfg['cgnsde']['stylized_sampler']!r}"
    )


def _run_training_step_and_collect(model, batch):
    """Run a single training_step and return loss, grads, and called sampler.

    The REAL (unpatched) ``emission_sample`` / ``emission_sample_seq``
    implementations execute; we only spy on them to record which one was
    invoked, without altering their return values.
    """
    # Capture which sampler was called
    calls = []

    orig_sample = model.sde_model.emission_sample
    orig_sample_seq = model.sde_model.emission_sample_seq

    def _make_spy(name, orig):
        def _fn(zs, regime=None, log_vol=None):
            calls.append(name)
            return orig(zs, regime=regime, log_vol=log_vol)
        return _fn

    # Spy without changing return values
    model.sde_model.emission_sample = _make_spy("emission_sample", orig_sample)
    model.sde_model.emission_sample_seq = _make_spy("emission_sample_seq", orig_sample_seq)

    # Fix RNG right before training_step so the difference between runs comes
    # strictly from the sampler, not from RNG drift in the forward pass.
    torch.manual_seed(123)
    loss = model.training_step(batch, 0)

    # Collect gradients after backward
    loss.backward()

    grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.clone().detach()

    return loss.detach(), grads, calls


def test_stylized_sampler_seq_changes_training_behavior():
    """Verify that stylized_sampler='seq' produces different loss/gradients than 'single'.

    Same seed and batch must yield different training_step loss and/or gradients
    because the sequential sampler's vol-recursion path differs from the one-shot joint draw.
    """
    from training.vortex_model import VORTEXModel

    torch.manual_seed(42)
    batch = _batch()

    # Model with 'single' sampler
    torch.manual_seed(100)
    model_single = _tiny_model("single")
    loss_single, grads_single, calls_single = _run_training_step_and_collect(model_single, batch)

    # Model with 'seq' sampler
    torch.manual_seed(100)
    model_seq = _tiny_model("seq")
    loss_seq, grads_seq, calls_seq = _run_training_step_and_collect(model_seq, batch)

    # Verify the correct sampler was called in each case
    assert "emission_sample" in calls_single and "emission_sample_seq" not in calls_single
    assert "emission_sample_seq" in calls_seq and "emission_sample" not in calls_seq

    # Loss values must differ (different computational graphs)
    assert not torch.allclose(loss_single, loss_seq), (
        f"Loss must differ between samplers: single={loss_single.item():.6f}, seq={loss_seq.item():.6f}"
    )

    # At least one gradient must differ (different backprop paths through vol recursion)
    grad_diff = False
    for name in grads_single:
        if name in grads_seq:
            if not torch.allclose(grads_single[name], grads_seq[name]):
                grad_diff = True
                break
    assert grad_diff, "Gradients must differ between samplers (vol recursion vs joint draw)"


def test_emission_sample_seq_receives_gradients_for_dynamics_and_emission_heads():
    """Verify emission_sample_seq backprops to rho_raw, kappa_raw, raw_nu, and emission heads.

    When stylized_sampler='seq', the stylized_facts_loss -> training_step backward
    must produce non-None gradients for:
      - sde_model.rho_raw
      - sde_model.kappa_raw (note: at init kappa_raw=0.0 -> sigmoid=0.5, clamped to 0.6,
        so grad may be zero at init due to clamp; the test checks the path exists)
      - sde_model.raw_nu (when emission in t/mt/smt)
      - sde_model.emi_m.weight/bias, emi_d.weight/bias, emi_B.weight/bias (if rank_K > 0)
    """
    from training.vortex_model import VORTEXModel
    import torch

    torch.manual_seed(200)
    batch = _batch()

    # Need emission='smt' to have raw_nu and emi_B (rank_K > 0)
    model = VORTEXModel(
        n_stocks=4,
        T=5,
        in_feats=6,
        latent_dim=8,
        proj_dim=4,
        hidden_dim=8,
        gat_heads=2,
        tau=0.07,
        lr=1e-3,
        lam_rec=1.0,
        lam_graph=0.0,
        lam_con=0.0,
        lam_na=0.0,
        beta_max=0.0,
        prior_warmup=1,
        lam_var=0.0,
        lam_corr=0.0,
        corr_ramp=1,
        lam_emission_corr=0.0,
        vol_dyn=True,
        emission="smt",
        rank_K=2,
        sde_sigma_max=1.0,
        warmup_epochs=1,
        lambda_styl=1.0,
        gat_dropout=0.1,
        stylized_sampler="seq",
    )
    model.log = lambda *a, **k: None

    # Set kappa_raw and raw_nu to values where clamp doesn't block gradients:
    # - kappa_raw=0.5 -> sigmoid(0.5) ≈ 0.62 > 0.6 (unblocked from min=0.6 clamp)
    # - raw_nu=1.0 -> softplus(1.0) ≈ 1.31 < 3.5 (unblocked from max=3.5 clamp)
    with torch.no_grad():
        model.sde_model.kappa_raw.fill_(0.5)
        model.sde_model.raw_nu.fill_(1.0)

    # Run training_step with lambda_styl > 0 so stylized_facts_loss is active
    loss = model.training_step(batch, 0)
    loss.backward()

    # Check dynamics parameters receive gradients (non-None and finite)
    assert model.sde_model.rho_raw.grad is not None, "rho_raw must receive gradient"
    assert model.sde_model.kappa_raw.grad is not None, "kappa_raw must receive gradient"
    assert torch.isfinite(model.sde_model.rho_raw.grad).all(), "rho_raw gradient must be finite"
    assert torch.isfinite(model.sde_model.kappa_raw.grad).all(), "kappa_raw gradient must be finite"

    # Check raw_nu (degrees of freedom for t/mt/smt) receives gradient
    assert model.sde_model.raw_nu is not None, "raw_nu must exist for emission='smt'"
    assert model.sde_model.raw_nu.grad is not None, "raw_nu must receive gradient"
    assert torch.isfinite(model.sde_model.raw_nu.grad).all(), "raw_nu gradient must be finite"

    # Check emission head parameters receive gradients
    assert model.sde_model.emi_m.weight.grad is not None, "emi_m.weight must receive gradient"
    assert model.sde_model.emi_m.bias.grad is not None, "emi_m.bias must receive gradient"
    assert model.sde_model.emi_d.weight.grad is not None, "emi_d.weight must receive gradient"
    assert model.sde_model.emi_d.bias.grad is not None, "emi_d.bias must receive gradient"
    assert model.sde_model.emi_B is not None, "emi_B must exist for rank_K > 0"
    assert model.sde_model.emi_B.weight.grad is not None, "emi_B.weight must receive gradient"
    assert model.sde_model.emi_B.bias.grad is not None, "emi_B.bias must receive gradient"

    # Sanity: gradients should be non-zero (at least some elements) for params
    # where the computational path is not blocked by clamps at initialization.
    for name, param in [
        ("rho_raw", model.sde_model.rho_raw),
        ("kappa_raw", model.sde_model.kappa_raw),  # now unblocked by kappa_raw=0.5 init
        ("raw_nu", model.sde_model.raw_nu),
        ("emi_m.weight", model.sde_model.emi_m.weight),
        ("emi_d.weight", model.sde_model.emi_d.weight),
        ("emi_B.weight", model.sde_model.emi_B.weight),
    ]:
        assert param.grad.abs().max() > 0, f"{name} gradient must have non-zero elements"
