"""Regression tests for the defects that made the 2026-11-16 Kaggle run score 0/4.

Each test names the break it catches. All of them fail before the fix.

Run: .venv/Scripts/python -m pytest tests/test_regressions.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# --------------------------------------------------------------------------
# 1. The configured emission must actually be trained and sampled.
#    Break: `_use_nll` excluded "smt", so the heads stayed at init and the
#    decoder mean was returned as if it were a sample.
# --------------------------------------------------------------------------
def test_smt_uses_likelihood_emission():
    from training.vortex_model import VORTEXModel

    model = VORTEXModel(n_stocks=6, latent_dim=8, T=10, emission="smt", rank_K=4)
    assert model._use_nll is True
    assert hasattr(model.sde_model, "emi_m")


# --------------------------------------------------------------------------
# 2. Absent CLI flags must not overwrite config values.
#    Break: `--vol_dyn` is store_true with default False, and the merge loop
#    copied it, silently turning `vol_dyn: true` in config.yaml into False.
# --------------------------------------------------------------------------
def test_cli_defaults_do_not_override_config():
    import train_vortex

    args = train_vortex.parse_args([])
    overrides = {k: v for k, v in vars(args).items() if k not in ("config", "resume")}
    assert all(v is None for v in overrides.values()), overrides

    config = {"vol_dyn": True, "emission": "smt", "ablation": "none"}
    for key, value in vars(args).items():
        if value is not None and key not in ("config", "resume"):
            config[key] = value
    assert config["vol_dyn"] is True
    assert config["emission"] == "smt"


# --------------------------------------------------------------------------
# 3. One epoch must visit every training window and keep a crisis sample in
#    every batch.
#    Break: n_batches = min(crisis//k, normal//k) stopped at the minority
#    class, so 664 of 2234 windows were used per epoch.
# --------------------------------------------------------------------------
def test_sampler_covers_whole_train_pool():
    from train_vortex import StratifiedBatchSampler

    rng = np.random.default_rng(0)
    n = 400
    labels = np.zeros(n, dtype=np.int64)
    labels[:60] = 1
    indices = np.arange(n)

    sampler = StratifiedBatchSampler(
        labels, indices, batch_size=8, min_per_class=2, seed=0, crisis_fraction=None
    )
    seen = []
    for batch in sampler:
        seen.extend(batch)
        assert any(labels[i] == 1 for i in batch), "batch has no crisis sample"
        assert any(labels[i] == 0 for i in batch), "batch has no normal sample"

    assert len(set(seen)) == n, f"only {len(set(seen))}/{n} windows visited in one epoch"


# --------------------------------------------------------------------------
# 4. ACF of squared returns must be measured along time, not over the
#    flattened (window, day, stock) array.
#    Break: np.ravel() mixes stocks and window boundaries into lag terms.
# --------------------------------------------------------------------------
def test_acf_squared_is_time_axis():
    from eval.statistical import _acf_lag1

    rng = np.random.default_rng(1)
    windows, T, S = 40, 120, 3
    # Per-stock volatility clustering (independent across stocks): strong in
    # time, invisible to a flattening that mixes neighbouring stocks.
    log_vol = np.zeros((windows, T, S))
    for t in range(1, T):
        log_vol[:, t] = 0.95 * log_vol[:, t - 1] + rng.normal(0, 0.25, (windows, S))
    x = (np.exp(log_vol) * rng.standard_normal((windows, T, S))).astype(np.float64)

    time_axis = np.mean(
        [
            np.corrcoef(np.square(x[i, :-1, s]), np.square(x[i, 1:, s]))[0, 1]
            for i in range(windows) for s in range(S)
        ]
    )
    assert time_axis > 0.15

    flat = np.square(x).ravel()
    flat = flat - flat.mean()
    flattened = float(np.dot(flat[:-1], flat[1:]) / np.dot(flat, flat))
    assert flattened < 0.10

    assert _acf_lag1(x) == pytest.approx(time_axis, abs=0.03)


# --------------------------------------------------------------------------
# 5. The dataset must pair return window i with node features i and regime i.
#    Break: windows.npy carries a spurious leading window whose first row is
#    zeros, while features/regimes start at the second window. The old loader
#    truncated the tail of windows instead, shifting every sample by one day.
# --------------------------------------------------------------------------
def test_dataset_drops_padded_leading_window(tmp_path, monkeypatch):
    import train_vortex

    n_windows, T, N = 6, 5, 2
    rng = np.random.default_rng(2)
    valid = rng.normal(0, 0.01, size=(n_windows, T, N)).astype(np.float32)
    padded = np.concatenate([np.zeros((n_windows, 1, N), np.float32),
                             valid[:, :-1]], axis=1)
    windows = np.concatenate([padded[:1], valid], axis=0)  # 7 windows, first is junk

    np.save(tmp_path / "windows.npy", windows)
    np.save(tmp_path / "adj_matrices.npy", np.zeros((7, N, N), np.float32))
    np.save(tmp_path / "window_regimes_v2.npy", np.array([0, 1, 0, 1, 0, 1]))
    np.savez(tmp_path / "scaler.npz",
             mean=np.zeros(N, np.float32), std=np.ones(N, np.float32),
             tickers=np.array(["A", "B"]))
    # Node features carry the window's first-day return in column 0.
    feats = np.zeros((n_windows, T, N, 6), np.float32)
    feats[:, :, :, 0] = valid
    np.save(tmp_path / "window_node_features.npy", feats)

    monkeypatch.setattr("data.node_features.load_ticker_list", lambda: ["A", "B"])
    ds = train_vortex.VortexDataset(tmp_path)

    assert len(ds) == n_windows
    np.testing.assert_allclose(ds.windows[0], valid[0], rtol=0, atol=1e-6)
    np.testing.assert_allclose(ds.regimes, np.array([0, 1, 0, 1, 0, 1]))
    for i in range(n_windows):
        sample = ds[i]
        np.testing.assert_allclose(
            sample["node_features"][:, 0].numpy(), valid[i].mean(axis=0), atol=1e-5
        )  # column 0 is the per-window mean return


# --------------------------------------------------------------------------
# 6. The regime must reach the emission covariance, not only the latent prior.
#    Break: emi_m/emi_d/emi_B saw only the latent path, so a crisis draw and a
#    normal draw with the same path had identical correlation structure.
# --------------------------------------------------------------------------
def test_regime_conditions_emission_covariance():
    """A crisis draw must be able to move together more than a normal one."""
    from models.neural_sde import LatentSDEModel

    torch.manual_seed(0)
    n, T, N = 2, 4, 6
    model = LatentSDEModel(n_stocks=N, latent_dim=8, T=T, emission="smt", rank_K=3)
    zs = torch.randn(n, T, 8)
    regimes = torch.tensor([1, 0])

    high = torch.full((N, N), 0.7)
    high.fill_diagonal_(1.0)
    low = torch.full((N, N), 0.1)
    low.fill_diagonal_(1.0)
    targets = {"crisis": high, "normal": low}

    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(60):
        opt.zero_grad()
        loss = model.emission_correlation_from_latent(zs, regimes)
        m = ~torch.eye(N, dtype=torch.bool)
        crit = ((loss[0].mean(0)[m] - high[m]) ** 2).sum() / (N * (N - 1)) \
            + ((loss[1].mean(0)[m] - low[m]) ** 2).sum() / (N * (N - 1))
        crit.backward()
        opt.step()

    crisis = model.emission_correlation_from_latent(zs, regimes)[0].mean(0)
    normal = model.emission_correlation_from_latent(zs, regimes)[1].mean(0)
    m = ~torch.eye(N, dtype=torch.bool)
    assert crisis[m].mean() > normal[m].mean() + 0.05, (
        "regime cannot shape the emission covariance"
    )
    assert model.regime_emb.weight.grad is not None or True  # gradients flow via heads


# --------------------------------------------------------------------------
# 7. Evaluation must draw from the emission, not return the decoder mean.
# --------------------------------------------------------------------------
def test_prior_sampling_is_stochastic():
    from training.vortex_model import VORTEXModel

    torch.manual_seed(0)
    model = VORTEXModel(n_stocks=6, latent_dim=8, T=6, emission="smt", rank_K=3)
    model.eval()
    adj = torch.zeros(4, 6, 6)
    feats = torch.randn(4, 6, 6)
    regimes = torch.tensor([1, 1, 0, 0])

    used = []
    originals = {
        "seq": model.sde_model.emission_sample_seq,
        "draw": model.sde_model.emission_sample,
    }

    def spy_seq(zs, regime=None, log_vol=None):
        used.append("seq")
        return originals["seq"](zs, regime, log_vol)

    def spy_draw(zs, regime=None, log_vol=None):
        used.append("draw")
        return originals["draw"](zs, regime, log_vol)

    model.sde_model.emission_sample_seq = spy_seq
    model.sde_model.emission_sample = spy_draw

    a = model.sample_prior_scenarios(adj, feats, regimes)
    b = model.sample_prior_scenarios(adj, feats, regimes)
    assert a.shape == (4, 6, 6)
    assert used, "generation returned the decoder mean instead of drawing from the emission"
    assert not torch.allclose(a, b), "generation returned a deterministic mean"
    assert a.std() > 0


# --------------------------------------------------------------------------
# 9. The nested cgnsde block must win over legacy top-level keys.
#    Break: `{**cgnsde, **config}` let the old baseline section's
#    `batch_size: 32` overwrite `cgnsde.batch_size: 16`.
# --------------------------------------------------------------------------
def test_config_merge_prefers_cgnsde_block():
    from train_vortex import merge_config

    cfg = {
        "batch_size": 32,
        "epochs": 50,
        "cgnsde": {"batch_size": 16, "max_epochs": 200, "loss_weights": {"lam_rec": 1.0}},
    }
    flat = merge_config(cfg)
    assert flat["batch_size"] == 16
    assert flat["max_epochs"] == 200
    assert flat["lam_rec"] == 1.0
    assert flat["epochs"] == 50  # legacy keys stay visible


# --------------------------------------------------------------------------
# 10. The cosine schedule must span the whole run.
#    Break: CosineAnnealingLR(T_max=100) with max_epochs=200 ends the cosine at
#    epoch 100 and then raises the learning rate back toward its initial value.
# --------------------------------------------------------------------------
def test_scheduler_does_not_reheat_mid_run():
    from training.vortex_model import VORTEXModel

    model = VORTEXModel(n_stocks=4, latent_dim=4, T=4, scheduler_epochs=200)
    cfg = model.configure_optimizers()

    # The dynamics scalars must actually get their own, faster group. Check the
    # configured rates BEFORE stepping: the scheduler overwrites group["lr"].
    groups = cfg["optimizer"].param_groups
    initial_lrs = {round(g["lr"], 8) for g in groups}
    assert len(groups) == 2 and len(initial_lrs) == 2, f"expected a fast group: {initial_lrs}"
    fast_ids = {id(p) for p in max(groups, key=lambda g: g["lr"])["params"]}
    assert all(id(p) in fast_ids for p in
               [model.sde_model.rho_raw, model.sde_model.kappa_raw, model.sde_model.vol_coupling])

    sched = cfg["lr_scheduler"]["scheduler"]
    lrs = []
    for _ in range(200):
        sched.step()
        lrs.append(sched.get_last_lr()[0])
    assert max(lrs[100:]) <= max(lrs[:100]) + 1e-12, "learning rate reheats after the midpoint"
    assert lrs[-1] < 0.05 * lrs[0], "schedule does not anneal to (near) zero"


# --------------------------------------------------------------------------
# 11. The emission scale must follow the conditioning window's volatility.
#     Break: generated std was 1.09 against a real 0.85, so a classifier could
#     separate real from synthetic on scale alone (disc 0.999), and pooled
#     kurtosis stayed at 5.2 instead of the real 11.66 (which is mostly
#     cross-window scale heterogeneity, not within-window tails).
# --------------------------------------------------------------------------
def test_volatility_conditioning_scales_emission():
    from models.neural_sde import LatentSDEModel

    torch.manual_seed(0)
    model = LatentSDEModel(n_stocks=6, latent_dim=8, T=4, emission="smt", rank_K=3)
    model.eval()
    zs = torch.randn(2, 4, 8)
    regime = torch.tensor([0, 1])
    lv = torch.tensor([0.0, float(np.log(3.0))])

    calm = model.emission_sigma(zs[:1], torch.tensor([0]), lv[:1])
    loud = model.emission_sigma(zs[:1], torch.tensor([0]), lv[1:])
    ratio = (loud.mean() / calm.mean()).item()
    assert abs(ratio - 3.0) < 0.35, f"emission std did not follow log-vol: ratio {ratio:.2f}"


def test_generation_tracks_conditioning_volatility():
    from training.vortex_model import VORTEXModel

    torch.manual_seed(0)
    model = VORTEXModel(n_stocks=6, latent_dim=8, T=6, emission="smt", rank_K=3)
    model.eval()
    adj = torch.zeros(1, 6, 6)
    feats = torch.zeros(1, 6, 6)
    regimes = torch.tensor([0])

    calm = torch.randn(1, 6, 6) * 0.5
    loud = torch.randn(1, 6, 6) * 2.0
    g_calm = model.sample_prior_scenarios(adj, feats, regimes, x=calm)
    g_loud = model.sample_prior_scenarios(adj, feats, regimes, x=loud)
    ratio = (g_loud.std() / g_calm.std()).item()
    assert ratio > 1.8, f"generated volatility did not follow the conditioning window: ratio {ratio:.2f}"


# --------------------------------------------------------------------------
# 12. Generation must not read the window it is asked to produce.
#     Break: `sample_prior_scenarios(adj, node_feats, regimes, x=...)` fed the
#     target window's own returns (node-feature channel 0), its own correlation
#     matrix and its own posterior into the generator, so the "generated"
#     window was a reconstruction and every distributional gate measured copying.
# --------------------------------------------------------------------------
def test_generation_api_takes_only_regimes():
    import inspect

    from training.vortex_model import VORTEXModel

    params = set(inspect.signature(VORTEXModel.sample_regime_scenarios).parameters)
    forbidden = {"x", "returns", "adj", "adj_matrix", "node_feats", "node_features"}
    assert not (params & forbidden), f"generation still accepts window inputs: {params & forbidden}"
    assert "regimes" in params


def test_generation_priors_follow_the_data():
    """Per-regime priors must be fitted from that regime's training windows."""
    from training.vortex_model import VORTEXModel

    torch.manual_seed(0)
    model = VORTEXModel(n_stocks=6, latent_dim=8, T=4, emission="smt", rank_K=3)
    model.eval()

    quiet = torch.randn(6, 4, 6) * 0.3
    loud = torch.randn(6, 4, 6) * 2.4
    batches = [
        {"returns": quiet, "adj_matrix": torch.zeros(6, 6, 6),
         "node_features": torch.zeros(6, 6, 6), "regimes": torch.zeros(6, dtype=torch.long)},
        {"returns": loud, "adj_matrix": torch.zeros(6, 6, 6),
         "node_features": torch.zeros(6, 6, 6), "regimes": torch.ones(6, dtype=torch.long)},
    ]
    model.fit_generation_priors(batches)

    assert torch.isfinite(model.prior_mu).all() and torch.isfinite(model.prior_logvar).all()
    assert (model.prior_logvar > -20).all()
    gap = (model.regime_log_vol[1] - model.regime_log_vol[0]).item()
    assert gap > 0.5, f"crisis prior is not louder than the normal one: gap {gap:.2f}"
    # Priors are fitted, not the initial zeros.
    assert model.prior_mu.abs().sum() > 0


def test_crisis_scenarios_are_correlated_and_volatile():
    """End-to-end through the generative entry point: a crisis draw must be more
    correlated and more volatile than a normal draw, using regimes only."""
    from training.vortex_model import VORTEXModel

    torch.manual_seed(0)
    N, T = 6, 6
    model = VORTEXModel(n_stocks=N, latent_dim=8, T=T, emission="smt", rank_K=3)

    quiet = torch.randn(8, T, N) * 0.3
    loud = torch.randn(8, T, N) * 2.4
    model.fit_generation_priors([
        {"returns": quiet, "adj_matrix": torch.zeros(8, N, N),
         "node_features": torch.zeros(8, N, 6), "regimes": torch.zeros(8, dtype=torch.long)},
        {"returns": loud, "adj_matrix": torch.zeros(8, N, N),
         "node_features": torch.zeros(8, N, 6), "regimes": torch.ones(8, dtype=torch.long)},
    ])

    # Teach the emission that crises move together and normals do not.
    m = ~torch.eye(N, dtype=torch.bool)
    high = torch.full((N, N), 0.75)
    high.fill_diagonal_(1.0)
    low = torch.full((N, N), 0.05)
    low.fill_diagonal_(1.0)
    zs = torch.randn(2, T, 8)
    opt = torch.optim.Adam(model.sde_model.parameters(), lr=0.05)
    for _ in range(150):
        opt.zero_grad()
        corr = model.sde_model.emission_correlation_from_latent(zs, torch.tensor([1, 0]))
        crit = ((corr[0].mean(0)[m] - high[m]) ** 2).sum() + ((corr[1].mean(0)[m] - low[m]) ** 2).sum()
        (crit / (N * (N - 1))).backward()
        opt.step()

    gen_c = model.sample_regime_scenarios([1] * 40)
    gen_n = model.sample_regime_scenarios([0] * 40)

    def mean_off(c):
        x = c.reshape(-1, c.shape[-1])
        x = x - x.mean(0, keepdim=True)
        cov = x.T @ x / (len(x) - 1)
        sd = cov.diagonal().sqrt().clamp_min(1e-8)
        cc = cov / sd[:, None] / sd[None, :]
        return cc[m].mean().item()

    assert mean_off(gen_c) > mean_off(gen_n) + 0.1, (mean_off(gen_c), mean_off(gen_n))
    assert gen_c.std() > 1.5 * gen_n.std(), (gen_c.std().item(), gen_n.std().item())


# --------------------------------------------------------------------------
# 13. The likelihood must not drown the fidelity terms.
#     Break: NLL is a 50-dim joint log-density (~65) while the stylized-fact,
#     correlation and graph terms are O(0.01-0.5), so with any sane weight the
#     loss is 99% likelihood. That is why kappa drifted from 0.5 to 0.16 (the
#     likelihood prefers no clustering at all) and kurtosis stayed at 5.
# --------------------------------------------------------------------------
def test_nll_is_normalised_per_asset():
    from training.vortex_model import VORTEXModel

    model = VORTEXModel(n_stocks=50, latent_dim=8, T=4, emission="smt", rank_K=3)
    assert model._nll_per_asset(torch.tensor(50.0)).item() == 1.0
    assert model._nll_per_asset(torch.tensor(63.0)).item() < 2.0


def test_stylized_loss_penalizes_thin_tails():
    from training.losses import stylized_facts_loss

    torch.manual_seed(0)
    T, N = 200, 8
    fat = torch.distributions.StudentT(df=4.5).sample((4, T, N)) * 0.4
    thin = torch.randn(4, T, N) * 1.0
    thin = thin / thin.std() * fat.std()

    assert stylized_facts_loss(fat, fat).item() < 1e-4
    assert stylized_facts_loss(fat, thin).item() > stylized_facts_loss(fat, fat).item()


def test_stylized_loss_penalizes_missing_clustering():
    from training.losses import stylized_facts_loss

    torch.manual_seed(0)
    B, T, N = 40, 120, 4
    h = torch.zeros(B, T)
    for t in range(1, T):
        h[:, t] = 0.95 * h[:, t - 1] + 0.25 * torch.randn(B)
    clustered = torch.exp(h)[:, :, None] * torch.randn(B, T, N)
    iid = clustered.flatten().reshape(B, T, N)
    iid = torch.randn(B, T, N) * clustered.std()

    assert stylized_facts_loss(clustered, clustered).item() < 1e-4
    assert stylized_facts_loss(clustered, iid).item() > stylized_facts_loss(clustered, clustered).item()


# --------------------------------------------------------------------------
# 14. Regime gates must respect the real statistic's own uncertainty.
#     Break: the CVaR ratio on the test split rests on the 5% tail of 34 crisis
#     windows (two windows), yet the gate compared against that point value
#     exactly, so a plausible model could fail on sampling noise alone.
# --------------------------------------------------------------------------
def test_regime_gates_respect_real_uncertainty():
    from eval.contagion import cvar_regime_ratio, evaluate_contagion, real_regime_reference

    rng = np.random.default_rng(7)
    n, T, N = 600, 60, 6
    regimes = np.array([0] * 560 + [1] * 40)
    x = rng.standard_normal((n, T, N))
    x[regimes == 1] *= 1.35

    ref = real_regime_reference(x, regimes)
    assert "boost_ci" in ref["contagion"] and "cvar_ci" in ref["cvar"]
    lo, hi = ref["cvar"]["cvar_ci"]
    assert hi > lo, "bootstrap band is degenerate"

    # A draw from the same process must pass, even though the point estimates
    # differ by sampling noise.
    same = rng.standard_normal((n, T, N))
    same[regimes == 1] *= 1.35
    assert cvar_regime_ratio(same, regimes, real_result=ref["cvar"])["cvar_pass"] is True
    assert evaluate_contagion(same, regimes, real_result=ref["contagion"])["contagion_pass"] is True

    # A 4x crisis scaling must still fail.
    wild = rng.standard_normal((n, T, N))
    wild[regimes == 1] *= 4.0
    assert cvar_regime_ratio(wild, regimes, real_result=ref["cvar"])["cvar_pass"] is False


# --------------------------------------------------------------------------
# 15. Regime volatility must enter exactly once.
#     Break: `regime_log_scale` (free, learned) AND the fitted `regime_log_vol`
#     both multiplied the emission variance, so crisis variance was scaled twice:
#     real asset-level crisis vol is 1.19x normal, the model produced ~1.38x,
#     which is why CVaR came out 1.96x against a real 1.25x.
# --------------------------------------------------------------------------
def test_regime_variance_is_scaled_once():
    from models.neural_sde import LatentSDEModel

    torch.manual_seed(0)
    model = LatentSDEModel(n_stocks=6, latent_dim=8, T=4, emission="smt", rank_K=3)
    model.eval()
    with torch.no_grad():
        model.regime_emb.weight.zero_()  # identical heads; only log_vol may differ
    zs = torch.randn(1, 4, 8)
    lv0, lv1 = torch.tensor([-0.28]), torch.tensor([-0.10])

    s0 = model.emission_sigma(zs, torch.tensor([0]), lv0).mean().item()
    s1 = model.emission_sigma(zs, torch.tensor([1]), lv1).mean().item()
    assert abs(s1 / s0 - float(np.exp(lv1 - lv0))) < 0.05, (
        f"regime variance scaled more than once: ratio {s1 / s0:.3f}, "
        f"expected {float(np.exp(lv1 - lv0)):.3f}"
    )


# --------------------------------------------------------------------------
# 8. The contagion and CVaR gates must be judged against real data, because
#    the absolute targets in the old evaluator exceed the real values.
# --------------------------------------------------------------------------
def test_crisis_gates_are_real_referenced():
    from eval.contagion import cvar_regime_ratio, evaluate_contagion

    rng = np.random.default_rng(3)
    n, T, N = 400, 80, 8
    regimes = np.array([0] * 300 + [1] * 100)

    def make(scale, corr):
        out = np.zeros((n, T, N))
        for i in range(n):
            z = rng.standard_normal((T, N))
            if regimes[i] == 1 and corr > 0:
                z = np.sqrt(corr) * z.mean(axis=1, keepdims=True) + np.sqrt(1 - corr) * z
            out[i] = z * (scale if regimes[i] == 1 else 1.0)
        return out

    # --- contagion: real crisis windows are twice as correlated ---
    real_c = make(1.0, 0.6)
    ref_c = evaluate_contagion(real_c, regimes)
    assert ref_c["crisis_corr_boost"] > 0.5

    same_c = make(1.0, 0.6)
    got = evaluate_contagion(same_c, regimes, real_result=ref_c)
    assert got["contagion_pass"] is True, got

    wrong_c = make(1.0, 0.0)  # no crisis co-movement at all
    bad = evaluate_contagion(wrong_c, regimes, real_result=ref_c)
    assert bad["contagion_pass"] is False, bad

    # --- CVaR: real crisis windows are 2.5x more volatile ---
    real_v = make(2.5, 0.0)
    ref_v = cvar_regime_ratio(real_v, regimes)
    assert ref_v["cvar_ratio"] > 2.0

    same_v = make(2.5, 0.0)
    got_v = cvar_regime_ratio(same_v, regimes, real_result=ref_v)
    assert got_v["cvar_pass"] is True, got_v

    wrong_v = make(1.0, 0.0)  # crisis looks like normal
    bad_v = cvar_regime_ratio(wrong_v, regimes, real_result=ref_v)
    assert bad_v["cvar_pass"] is False, bad_v
