"""Tests for the CG-NSDE generator.  Run: .venv/Scripts/python.exe tests/test_cgnsde.py

Small synthetic panel, tiny budgets.  Every test asserts something that a broken
implementation would violate (see the comment at the top of each test).
"""
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crisisbench import protocol as P                                   # noqa: E402
from crisisbench.generators import get                                  # noqa: E402
from crisisbench.generators import cgnsde as C                          # noqa: E402
from crisisbench.generators.base import FitData                         # noqa: E402
from crisisbench.labels import fit_rule                                 # noqa: E402

torch.set_num_threads(4)
SMALL = dict(latent_dim=8, graph_dim=8, gat_hidden=8, sde_hidden=16, proj_dim=8, rank_K=3, batch=16, min_crisis=6,
             gmm_components=(1, 3))


def make_data(D=460, N=8, seed=0, volume=True) -> FitData:
    """One-factor panel with a high-vol, high-correlation episode on days 100..170; validation = days 280..389."""
    rng = np.random.default_rng(seed)
    vol = np.full(D, 0.01)
    vol[100:170] = 0.035
    load = np.where((np.arange(D) >= 100) & (np.arange(D) < 170), 0.9, 0.4)[:, None]
    f = rng.standard_normal((D, 1))
    R = vol[:, None] * (load * f + np.sqrt(1 - load ** 2) * rng.standard_normal((D, N))) + 0.0003
    val = np.zeros(D, bool)
    val[280:390] = True
    V = np.exp(rng.normal(10, 1, (D, N))) if volume else None
    fd = FitData("synthetic", R, pd.bdate_range("2015-01-01", periods=D), [f"A{i}" for i in range(N)], val, V)
    fd.rule = fit_rule(fd.windows())
    return fd


def make_batch(net, B=12, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((B, net.T, net.N))
    Z[: B // 2] *= 2.0
    adj, tgt = C.window_graphs(Z, 3)
    feats = C.node_features(Z, Z * 0.01, None, np.zeros(net.N, np.float32))
    y = torch.tensor([1] * (B // 2) + [0] * (B - B // 2))
    tgt_corr = torch.stack([torch.eye(net.N) * 0.5 + 0.5, torch.eye(net.N) * 0.8 + 0.2])
    return dict(x=torch.from_numpy(Z.astype(np.float32)), adj=torch.from_numpy(adj), tgt=torch.from_numpy(tgt),
                feats=torch.from_numpy(feats), y=y), tgt_corr


def test_emission_density_and_sampler():
    """log_prob equals independent reference densities, and draws have covariance mult * (BB' + diag d)."""
    from scipy import stats
    torch.manual_seed(0)
    N, K = 5, 2
    x, m = torch.randn(4, N, dtype=torch.float64), torch.randn(4, N, dtype=torch.float64) * 0.1
    d = torch.rand(4, N, dtype=torch.float64) + 0.2
    B = torch.randn(4, N, K, dtype=torch.float64) * 0.5
    mult = torch.tensor([0.5, 1.0, 2.0, 3.0], dtype=torch.float64)
    nu = torch.tensor(6.0, dtype=torch.float64)
    cov = mult[:, None, None] * (B @ B.transpose(-1, -2) + torch.diag_embed(d))
    ref = torch.distributions.MultivariateNormal(m, cov).log_prob(x)
    assert torch.allclose(C.emission_log_prob(x, m, d, B, mult, None), ref, atol=1e-9)
    ref = torch.distributions.Normal(m, torch.sqrt(mult[:, None] * d)).log_prob(x).sum(-1)
    assert torch.allclose(C.emission_log_prob(x, m, d, None, mult, None), ref, atol=1e-9)
    lp = C.emission_log_prob(x, m, d, B, mult, nu)
    for i in range(4):                    # scipy's shape matrix = covariance * (nu-2)/nu
        r = stats.multivariate_t(m[i].numpy(), cov[i].numpy() * 4.0 / 6.0, df=6.0).logpdf(x[i].numpy())
        assert abs(float(lp[i]) - r) < 1e-8, (float(lp[i]), r)
    ref = torch.distributions.StudentT(nu, m, torch.sqrt(mult[:, None] * d * 4.0 / 6.0)).log_prob(x).sum(-1)
    assert torch.allclose(C.emission_log_prob(x, m, d, None, mult, nu), ref, atol=1e-9)

    n = 400_000
    g = torch.Generator().manual_seed(1)
    for shared in (True, False):
        Bi = B[2] if shared else None
        w = torch.distributions.Chi2(nu).sample((n, 1 if shared else N))
        r = C.emission_draw(m[2], d[2], Bi, mult[2].expand(n), nu, torch.randn(n, N, generator=g, dtype=torch.float64),
                            torch.randn(n, K, generator=g, dtype=torch.float64), w)
        want = cov[2] if shared else torch.diag(mult[2] * d[2])
        got = torch.cov(r.T)
        assert (got - want).abs().max() < 0.05 * want.abs().max(), (got, want)
        assert (r.mean(0) - m[2]).abs().max() < 0.02


def test_sde_solver_and_diffusion_floor():
    """torchsde path == hand-written Euler-Maruyama on the same increments; diffusion spans (0, sigma_max)."""
    torch.manual_seed(0)
    net = C.CGNSDENet(6, {**C.DEFAULTS, **SMALL})
    z0, g = torch.randn(5, net.L), torch.randn(5, net.G)
    dW = torch.randn(net.T, 5, net.L)
    with torch.no_grad():
        zs = net.path(z0, g, dW)
        f = C._SDEFunc(net, g)
        z, ref = z0, []
        for s in range(net.T):
            inp = torch.cat([z, g, torch.full((5, 1), s / net.T)], 1)
            z = z + torch.tanh(net.drift(inp)) / net.T \
                + net.sigma_max * torch.sigmoid(net.diff(inp)) * dW[s] / math.sqrt(net.T)
            ref.append(z)
        assert zs.shape == (5, net.T, net.L)
        assert torch.allclose(zs, torch.stack(ref, 1), atol=1e-5)
        assert not torch.allclose(zs, net.path(z0, g, torch.zeros_like(dW)), atol=1e-3)   # noise does enter
        t = torch.tensor(3.0)
        net.diff[-1].weight.zero_()
        net.diff[-1].bias.fill_(-20.0)
        assert f.g(t, z0).max() < 1e-7, "diffusion cannot approach 0"
        net.diff[-1].bias.fill_(20.0)
        assert torch.allclose(f.g(t, z0) * math.sqrt(net.T), torch.full((5, net.L), net.sigma_max), atol=1e-5)
    # legacy double sigmoid for contrast: its minimum is sigma_max * sigmoid(0) = 0.5 * sigma_max
    assert net.sigma_max * torch.sigmoid(torch.sigmoid(torch.tensor(-20.0))) > 0.49 * net.sigma_max


def test_batched_gat_equals_loop():
    """One offset-index GAT call == the same network applied graph by graph; attention rows are distributions."""
    torch.manual_seed(1)
    net = C.CGNSDENet(7, {**C.DEFAULTS, **SMALL}).eval()
    b, _ = make_batch(net, B=6)
    with torch.no_grad():
        g, att = net.graph(b["feats"], b["adj"])
        for i in range(6):
            gi, ai = net.graph(b["feats"][i:i + 1], b["adj"][i:i + 1])
            assert torch.allclose(g[i], gi[0], atol=1e-5) and torch.allclose(att[i], ai[0], atol=1e-5)
    assert g.shape == (6, net.G) and att.shape == (6, 7, 7)
    assert torch.allclose(att.sum(-1), torch.ones(6, 7), atol=1e-5)
    assert (att[~b["adj"]] == 0).all() and (att[b["adj"]] > 0).all()
    assert not torch.allclose(g[0], g[1], atol=1e-4)            # graphs do not leak into each other trivially
    assert torch.allclose(b["tgt"].sum(-1), torch.ones(6, 7), atol=1e-5) and (b["tgt"][~b["adj"]] == 0).all()
    assert (b["adj"] == b["adj"].transpose(1, 2)).all() and b["adj"].sum(-1).min() >= 4   # symmetric, k=3 + self


def test_each_loss_term_contributes_gradient():
    """For every term: its own gradient is non-zero on parameters shared with the NLL path, and switching its
    weight from 0 to its value changes the gradient of the total loss."""
    hp = {**C.DEFAULTS, **SMALL}
    torch.manual_seed(2)
    net = C.CGNSDENet(7, hp)
    batch, tgt_corr = make_batch(net)
    w_full = C.epoch_weights(hp, 100)
    assert w_full == dict(kl=0.01, corr=100.0, supcon=0.1, graph=0.1), w_full
    assert abs(C.epoch_weights(hp, 0)["supcon"] - 0.1 / 5) < 1e-12 and abs(C.epoch_weights(hp, 0)["kl"] - 0.01 / 10) < 1e-12

    def grads(weights, only=None):
        torch.manual_seed(3)
        terms = C.loss_terms(net, batch, tgt_corr, hp["tau"], torch.Generator().manual_seed(3))
        assert set(terms) == {"nll", "kl", "corr", "supcon", "graph"}
        assert all(torch.isfinite(v) for v in terms.values())
        loss = terms[only] if only else C.total_loss(terms, weights)
        net.zero_grad()
        loss.backward()
        return {n: (p.grad.clone() if p.grad is not None else torch.zeros_like(p)) for n, p in net.named_parameters()}

    g_full = grads(w_full)
    assert torch.allclose(g_full["enc.weight_ih_l0"], grads(w_full)["enc.weight_ih_l0"])     # reproducible
    shared = {"nll": "enc.weight_ih_l0", "kl": "enc.weight_ih_l0", "corr": "enc.weight_ih_l0",
              "supcon": "enc.weight_ih_l0", "graph": "gat1.lin.weight"}
    norms = {}
    for term, pname in shared.items():
        own = grads(None, only=term)
        norms[term] = float(torch.sqrt(sum((v ** 2).sum() for v in own.values())))
        assert own[pname].abs().max() > 0, f"{term}: no gradient reaches {pname}"
        if term == "nll":
            continue
        g_off = grads({**w_full, term: 0.0})
        assert not torch.allclose(g_full[pname], g_off[pname], rtol=1e-4, atol=0), f"{term} does not change the gradient"
    assert g_full["gat1.lin.weight"].abs().max() > 0 and grads(None, only="nll")["gat1.lin.weight"].abs().max() > 0
    assert grads(None, only="supcon")["proj.2.weight"].abs().max() > 0
    assert grads(None, only="supcon")["drift.0.weight"].abs().max() > 0       # SupCon reaches the SDE
    assert grads(None, only="kl")["prior_mu"].abs().max() > 0
    print("   weighted gradient norms at init:",
          {k: f"{norms[k] * (1.0 if k == 'nll' else w_full[k]):.3g}" for k in norms})


def test_supcon_matches_definition():
    """Vectorised SupCon equals the literal double loop of Khosla et al. eq. 2."""
    torch.manual_seed(4)
    f, y, tau = torch.randn(9, 5), torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1]), 0.07
    u = torch.nn.functional.normalize(f, dim=1)
    tot = 0.0
    for i in range(9):
        den = sum(math.exp(float(u[i] @ u[a]) / tau) for a in range(9) if a != i)
        pos = [p for p in range(9) if p != i and y[p] == y[i]]
        tot += -sum(math.log(math.exp(float(u[i] @ u[p]) / tau) / den) for p in pos) / len(pos)
    assert abs(float(C.supcon_loss(f, y, tau)) - tot / 9) < 1e-4


def test_vol_recursion_teacher_equals_sequential():
    """Feeding sample_seq's own output back through teacher_mult reproduces the multipliers it used
    (training and sampling run the same recursion), and the multiplier is bounded."""
    torch.manual_seed(5)
    net = C.CGNSDENet(6, {**C.DEFAULTS, **SMALL}).eval()
    with torch.no_grad():
        net.kappa_raw.fill_(0.5)
        zs = torch.randn(4, net.T, net.L)
        sN, sK, sw = net.noise_shapes(4)
        eN, eK = torch.randn(sN), torch.randn(sK)
        w = torch.distributions.Chi2(net.nu()).sample(sw)
        r = net.sample_seq(zs, eN, eK, w)
        m, d, B = net.heads(zs)
        mult = net.teacher_mult(r, m, d, B)
        assert torch.allclose(r, C.emission_draw(m, d, B, mult, net.nu(), eN, eK, w), atol=1e-4)
        assert mult.std() > 0.01 and (mult[:, 0] == 1).all()
        assert mult.max() <= math.exp(4.0) and mult.min() >= math.exp(-2.0)
        assert torch.isfinite(net.sample_seq(zs * 50, eN * 50, eK, w)).all()         # extreme inputs stay finite


def test_stratified_batches():
    y = np.array([1] * 9 + [0] * 100)
    rng = np.random.default_rng(0)
    bs = list(C.stratified_batches(y, 16, 6, rng))
    assert len(bs) == 10 and all(len(b) == 16 and y[b].sum() == 6 and len(set(b)) == 16 for b in bs)
    assert len(set(np.concatenate(bs)[y[np.concatenate(bs)] == 0])) == 100       # every normal window once


def test_fit_sample_save_load():
    """End to end on the synthetic panel: scaler from training days only, shapes, finiteness, determinism,
    regime handle, no stored training windows, save/load identity, encode/attention."""
    fd = make_data()
    gen = get("cgnsde", **SMALL, max_epochs=3)
    info = gen.fit(fd, seed=0)
    assert np.allclose(gen.mu, fd.returns[~fd.val_mask].mean(0)) and np.allclose(gen.sd, fd.returns[~fd.val_mask].std(0))
    assert not np.allclose(gen.mu, fd.returns.mean(0))
    assert info["epochs_run"] == 3 and 1 <= info["best_epoch"] <= 3 and len(info["history"]) == 3
    need = {"train_nll", "train_kl", "train_corr", "train_supcon", "train_graph", "train_total", "val_nll"}
    assert need <= set(info["history"][0]) and all(math.isfinite(h[k]) for h in info["history"] for k in need)
    assert info["nu"] > 4.5 and set(info["mixture_components"]) == {"normal", "crisis"}
    assert info["history"][-1]["train_nll"] < info["history"][0]["train_nll"]            # it learns
    tr, _ = fd.split_windows()
    assert abs(gen.p_crisis - fd.window_labels()[tr].mean()) < 1e-12
    # nothing of window-count size is kept on the object: sampling cannot replay training windows
    for k, v in vars(gen).items():
        if isinstance(v, (np.ndarray, torch.Tensor)):
            assert v.shape[0] != len(tr) and v.size < 1000, k

    G = gen.sample(64, np.random.default_rng(5))
    assert G.shape == (64, P.T, fd.n_assets) and np.isfinite(G).all()
    assert np.array_equal(G, gen.sample(64, np.random.default_rng(5)))
    assert not np.array_equal(G, gen.sample(64, np.random.default_rng(6)))
    assert 0.2 < G.std() / fd.returns.std() < 5.0, G.std() / fd.returns.std()             # raw units
    Gc, Gn = gen.sample_regime(40, 1, np.random.default_rng(7)), gen.sample_regime(40, 0, np.random.default_rng(7))
    assert Gc.shape == Gn.shape == (40, P.T, fd.n_assets) and not np.array_equal(Gc, Gn)
    assert np.array_equal(Gc, gen.sample_regime(40, 1, np.random.default_rng(7)))

    gen2 = get("cgnsde", **SMALL, max_epochs=3)                    # same seed -> same fit
    gen2.fit(fd, seed=0)
    assert np.allclose(G, gen2.sample(64, np.random.default_rng(5)), atol=1e-6)

    with tempfile.TemporaryDirectory() as tmp:
        gen.save(Path(tmp) / "sub" / "seed0")
        assert [p.name for p in (Path(tmp) / "sub").iterdir()] == ["seed0.pt"]
        back = C.CGNSDE.load(Path(tmp) / "sub" / "seed0")
    assert np.array_equal(G, back.sample(64, np.random.default_rng(5)))
    assert np.array_equal(Gc, back.sample_regime(40, 1, np.random.default_rng(7)))
    assert back.tickers == fd.tickers and back.hparams == gen.hparams

    W = fd.windows()[:10]
    e = gen.encode(W)
    assert e["z0"].shape == (10, 8) and e["g"].shape == (10, 8) and e["proj"].shape == (10, 8)
    assert all(np.isfinite(v).all() for v in e.values()) and np.allclose(np.linalg.norm(e["proj"], axis=1), 1, atol=1e-5)
    assert all(np.array_equal(e[k], back.encode(W)[k]) for k in e)
    a = gen.attention(W[0])
    assert a.shape == (8, 8) and np.allclose(a.sum(1), 1, atol=1e-5) and (a >= 0).all()
    for bad in ("sample", "encode"):
        try:
            getattr(get("cgnsde"), bad)(1, np.random.default_rng(0))
            raise AssertionError("unfitted generator did not raise")
        except RuntimeError:
            pass


def test_variants_run_one_epoch():
    """Every registered variant builds what it claims, trains for one epoch and samples finite windows."""
    fd = make_data(volume=False)
    expect = dict(no_gat=lambda n, h: not n.use_gat and not hasattr(n, "gat1"),
                  no_supcon=lambda n, h: h["lam_con"] == 0, no_graph_loss=lambda n, h: h["lam_graph"] == 0,
                  no_corr=lambda n, h: h["lam_corr"] == 0, no_voldyn=lambda n, h: not hasattr(n, "rho_raw"),
                  no_sde=lambda n, h: not hasattr(n, "drift"), gauss=lambda n, h: n.nu() is None,
                  K0=lambda n, h: n.K == 0 and not hasattr(n, "emi_B") and n.nu() is not None,
                  K2=lambda n, h: n.K == 2, K4=lambda n, h: n.K == 4, K16=lambda n, h: n.K == 16,
                  lam_graph03=lambda n, h: h["lam_graph"] == 0.3)
    assert set(expect) == set(C.CGNSDE.VARIANTS)
    skip = {"no_supcon": "train_supcon", "no_corr": "train_corr", "no_graph_loss": "train_graph", "no_gat": "train_graph"}
    for name, check in expect.items():
        small = {k: v for k, v in SMALL.items() if k not in C.CGNSDE.VARIANTS[name]}
        gen = get(f"cgnsde@{name}", **small, max_epochs=1)
        assert gen.name == f"cgnsde@{name}"
        info = gen.fit(fd, seed=1)
        assert check(gen.net, gen.hparams), name
        h = info["history"][0]
        assert math.isfinite(h["train_total"]) and math.isfinite(h["val_nll"])
        for v, key in skip.items():
            assert (key in h) == (v != name and name != "no_gat" or key != "train_graph" and v != name), (name, key)
        G = gen.sample(20, np.random.default_rng(0))
        assert G.shape == (20, P.T, fd.n_assets) and np.isfinite(G).all(), name
        if name == "no_sde":                                   # latent path is constant: z_t = z0
            z0 = torch.randn(3, gen.net.L)
            zs = gen.net.path(z0, torch.randn(3, gen.net.G), torch.randn(P.T, 3, gen.net.L))
            assert (zs == z0[:, None, :]).all()
        if name == "no_gat":
            try:
                gen.attention(fd.windows()[0])
                raise AssertionError("no_gat returned an attention matrix")
            except NotImplementedError:
                pass
    try:
        get("cgnsde", not_a_param=1)
        raise AssertionError("unknown hyperparameter accepted")
    except ValueError:
        pass


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("OK")
