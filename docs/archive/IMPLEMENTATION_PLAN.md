# CG-NSDE Phase 3: Neural SDE Integration — Implementation Plan

## Project Context

**VORTEX-AI (CG-NSDE)** — Contrastive Graph-Neural Stochastic Differential Equations for NSE Stress Testing.

**Current status (Phases 0-2 complete):**
- Phase 0: Data pipeline — `data/download.py`, `data/preprocess.py`, `data/node_features.py`, `data/graph_builder.py`
- Phase 1: GAT Encoder — `DynamicGATEncoder` in `models/gat_encoder.py`
- Phase 2: Baseline LSTM + GAT training — `training/trainer.py`, `training/gat_trainer.py`

**Next target:** Phase 3 — Neural SDE using `torchsde` with graph-conditioned drift/diffusion.

**Masterplan reference:** `VORTEX-AI_CG-NSDE_Masterplan.pdf`, Part C.3, Part D.3

---

## 1. Objective

Implement a `GraphConditionedSDE` (subclass of `torchsde.SDEIto`) and a `LatentSDEModel` wrapper that:
1. Takes latent embeddings from the GAT encoder as graph context
2. Integrates SDE paths via Euler-Maruyama
3. Decodes latent paths back to per-stock return space
4. Produces `(B, T, N)` return sequences for downstream reconstruction + circuit filter loss

---

## 2. Technical Specifications

### 2.1 Mathematical Formulation

The Itô SDE to implement:

```
dX_t = mu_theta(X_t, G_t, z_t, t) dt + sigma_theta(X_t, G_t, z_t, t) dW_t
```

Where:
- `X_t` — latent state at time t, shape `(B, latent_dim)`
- `G_t` — graph embedding context from GAT, shape `(B, graph_emb_dim)`
- `z_t` — optional encoder hidden state, shape `(B, latent_dim)`
- `t` — scalar or vector time
- `W_t` — Wiener process (Brownian motion)

### 2.2 Environment & Dependencies

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.13 | (via `.venv`) |
| torch | 2.13.0+cu126 | CUDA build available |
| torchsde | 0.2.6 | Provides `SDEIto` base class, `sdeint` integrator |
| torch-geometric | 2.8.0.post1 | GATConv layers |

### 2.3 Key Design Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `latent_dim` | 64 | Matches masterplan spec; GAT hidden output dim |
| `graph_emb_dim` | 64 | From `pool_graph_embedding` (mean pooling of GAT node embeddings) |
| `sde_hidden_dim` | 128 | Masterplan: "Increase for more expressivity" |
| `n_stocks` | 50 | NIFTY-50 fixed count |
| SDE solver | `euler` (Euler-Maruyama) | Masterplan: "do NOT use adaptive solvers early on" |
| `dt` | 1/T = 1/60 ≈ 0.0167 | Fixed step per window timestep |
| `noise_type` | `diagonal` | Per-dimension independent noise |
| `sde_type` | `ito` | Itô interpretation, standard for finance |
| Gradient clip | `max_norm=1.0` | Non-negotiable per masterplan Part K |
| Device | `cuda` if available, else `cpu` | CUDA currently unavailable in env; will use CPU |

---

## 3. Implementation Components

### 3.1 `GraphConditionedSDE` (models/neural_sde.py)

**New class** — replaces the simplified `NeuralSDE` currently in the file.

```python
class GraphConditionedSDE(torchsde.SDEIto):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, latent_dim=64, graph_emb_dim=64, hidden_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.graph_emb_dim = graph_emb_dim

        # Input to nets: [X_t (latent_dim), G_t (graph_emb_dim), t (1)]
        input_dim = latent_dim + graph_emb_dim + 1

        self.drift_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.diffusion_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Softplus(),  # Ensures non-negative diffusion
        )

        self.graph_context = None  # Set per-forward via set_graph_context()

    def set_graph_context(self, graph_emb):
        """Inject graph embedding from GAT pooling: (B, H) -> (B, graph_emb_dim)."""
        self.graph_context = graph_emb

    def _get_input(self, t, y):
        """Concatenate state, graph context, and scalar time.

        Args:
            t: scalar tensor (0-dim) or (B,) vector
            y: (B, latent_dim)
        Returns:
            (B, latent_dim + graph_emb_dim + 1)
        """
        B = y.shape[0]
        t_vec = t.expand(B, 1) if t.dim() == 0 else t.unsqueeze(-1)
        return torch.cat([y, self.graph_context, t_vec], dim=-1)

    def f(self, t, y):
        """Drift: mu_theta(X_t, G_t, z_t, t)."""
        inp = self._get_input(t, y)
        return self.drift_net(inp)

    def g(self, t, y):
        """Diffusion: sigma_theta(X_t, G_t, z_t, t)."""
        inp = self._get_input(t, y)
        return self.diffusion_net(inp)
```

### 3.2 `LatentSDEModel` (models/neural_sde.py)

**New class** — encapsulates encoder, SDE integration, and decoder.

```python
class LatentSDEModel(nn.Module):
    def __init__(self, n_stocks=50, latent_dim=64, T=60, rnn_hidden=64):
        super().__init__()
        self.T = T
        self.latent_dim = latent_dim
        self.n_stocks = n_stocks

        # Block 2: Latent Encoder (per masterplan D.3 — GRU)
        self.encoder = nn.GRU(n_stocks, rnn_hidden, batch_first=True)

        # Block 3: Neural SDE
        self.sde = GraphConditionedSDE(
            latent_dim=rnn_hidden,
            graph_emb_dim=latent_dim,  # GAT pooled output dim
            hidden_dim=128,
        )

        # Block 5: Decoder
        self.decoder = nn.Linear(rnn_hidden, n_stocks)

    def forward(self, x, graph_emb, ts=None):
        """Full forward pass.

        Args:
            x: (B, T, N) — windowed return sequences
            graph_emb: (B, H) — pooled GAT node embeddings
            ts: (T,) — time grid (default: linspace(0, 1, T))
        Returns:
            r_hat: (B, T, N) — decoded return estimates
            zs: (B, T, latent_dim) — latent paths
        """
        B, T, N = x.shape
        if ts is None:
            ts = torch.linspace(0, 1, T, device=x.device)

        # Encode: (B, T, N) -> (B, latent_dim)
        _, h_n = self.encoder(x)
        z0 = h_n.squeeze(0)

        # Inject graph context into SDE
        self.sde.set_graph_context(graph_emb)

        # Integrate: (T, B, latent_dim) -> permute to (B, T, latent_dim)
        zs = torchsde.sdeint(
            self.sde, z0, ts,
            method="euler",
            dt=1.0 / T,
            names={"drift": "f", "diffusion": "g"},
        )
        zs = zs.permute(1, 0, 2)  # (B, T, latent_dim)

        # Decode: (B, T, latent_dim) -> (B, T, N)
        r_hat = self.decoder(zs)

        return r_hat, zs
```

### 3.3 GAT Integration (models/gat_encoder.py — minimal change)

The existing `DynamicGATEncoder` already produces `node_embs` (B, N, H). The coupling step is mean pooling:

```python
# In VORTEXModel.forward (next phase):
node_embs, learned_adj = self.gat(node_feats, adj_matrix)  # (B, N, H), (B, N, N)
graph_emb = node_embs.mean(dim=1)  # (B, H) -> passed as graph_emb to LatentSDEModel
```

### 3.4 Updated `src/model.py`

Add `LatentSDEModel` to the exported API:

```python
from models.gat_encoder import DynamicGATEncoder
from models.neural_sde import GraphConditionedSDE, LatentSDEModel

__all__ = ["DynamicGATEncoder", "LatentSDEModel", "GraphConditionedSDE"]
```

---

## 4. Validation Methodology

### 4.1 Single-Stock Sanity Test (Pre-Scaling)

Before running on all 50 stocks, verify the SDE produces GBM-like behavior:

```python
# Test: single stock, known parameters
x = torch.randn(4, 60, 1) * 0.01  # (B=4, T=60, N=1)
graph_emb = torch.zeros(4, 64)    # neutral graph context
model = LatentSDEModel(n_stocks=1, latent_dim=64, T=60, rnn_hidden=8)
r_hat, zs = model(x, graph_emb)
assert r_hat.shape == (4, 60, 1)
assert zs.shape == (4, 60, 8)
assert torch.isfinite(r_hat).all()
# Generated paths should look approximately Gaussian
```

**Acceptance criteria:**
- No NaN/Inf in output
- Output shape matches `(B, T, N)`
- Returns distribution approximates Gaussian (mean ≈ 0, volatility ≈ input scale)

### 4.2 Full 50-Stock Forward Pass

```python
# Test: full NIFTY-50 window
x = torch.randn(8, 60, 50)        # (B=8, T=60, N=50)
graph_emb = torch.randn(8, 64)    # realistic GAT output
model = LatentSDEModel(n_stocks=50, latent_dim=64, T=60, rnn_hidden=64)
r_hat, zs = model(x, graph_emb)
assert r_hat.shape == (8, 60, 50)
assert zs.shape == (8, 60, 64)
```

**Acceptance criteria:**
- Output shape `(B, T, 50)`
- All finite values
- Computation completes in <30s on CPU (batch=8)

### 4.3 NaN Prevention Protocol (from masterplan Part E)

| Failure Mode | Detection | Fix |
|-------------|-----------|-----|
| SDE loss NaN | `torch.isnan(loss)` check in training loop | Use `method='euler'`, `dt=1/120`, `clip_grad_norm_(1.0)` |
| Exploding paths | `torch.abs(zs).max() > 100` check | Reduce learning rate by 10x, add gradient clipping |
| Constant output | `zs.std() < 1e-6` check | Verify graph_emb is non-zero, add input noise |

---

## 5. Optimized Workflow

### 5.1 Iterative Development Sequence

```
Step 1: Implement GraphConditionedSDE + LatentSDEModel
Step 2: Run single-stock sanity test
Step 3: Fix NaN/divergence issues (apply gradient clipping)
Step 4: Run 50-stock forward pass
Step 5: Wire into end-to-end model (Phase 4 prep)
Step 6: Document integration points for trainer
```

### 5.2 Performance Optimizations

| Optimization | Detail |
|-------------|--------|
| **CPU threading** | Set `torch.set_num_threads()` to match core count |
| **Avoid recompilation** | Use `torchsde.sdeint` with `method='euler'` (no adaptive overhead) |
| **Batch the SDE** | Process all `(B, N)` samples in one `sdeint` call, not per-sample loop |
| **Gradient checkpointing** | For memory-constrained GPU: use `torch.utils.checkpoint` on encoder/GRU |
| **Mixed precision** | Phase 3 does not use AMP (SDEs are sensitive); defer to later phases |

### 5.3 Debugging & Logging

- Log `zs.std()` and `r_hat.std()` per epoch to detect mode collapse
- Log gradient norms for `drift_net`, `diffusion_net`, `encoder`, `decoder`
- Use `torchsde`'s `logger` for solver diagnostics if NaN occurs

---

## 6. File Changes Required

| File | Action | Details |
|------|--------|---------|
| `models/neural_sde.py` | **Rewrite** | Replace `NeuralSDE` with `GraphConditionedSDE` + `LatentSDEModel` |
| `src/model.py` | **Update** | Export `LatentSDEModel`, `GraphConditionedSDE` |
| `models/gat_encoder.py` | **No change** | `pool_graph_embedding()` already available |
| `training/losses.py` | **Update** (Phase 5) | Add `reconstruction_loss`, `stylized_facts_loss`, `total_loss` per masterplan D.6 |
| `training/trainer.py` | **Update** (Phase 7) | Wire into PyTorch Lightning `VORTEXModel` per masterplan D.7 |

### 6.1 `models/neural_sde.py` — Complete Rewrite

```python
"""Neural SDE with graph-conditioned drift and diffusion.

Implements Block 3 of the CG-NSDE architecture:
- Input: latent z0 (B, latent_dim), graph_emb (B, H), ts (T,)
- Output: latent paths (B, T, latent_dim), decoded returns (B, T, N)
- Solver: Euler-Maruyama (fixed step)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchsde


class GraphConditionedSDE(torchsde.SDEIto):
    """Itô SDE with graph-conditional drift and diffusion.

    dX_t = mu_theta(X_t, G_t, t) dt + sigma_theta(X_t, G_t, t) dW_t
    """

    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, latent_dim=64, graph_emb_dim=64, hidden_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.graph_emb_dim = graph_emb_dim
        input_dim = latent_dim + graph_emb_dim + 1

        self.drift_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.diffusion_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Softplus(),
        )

        self.graph_context = None

    def set_graph_context(self, graph_emb):
        self.graph_context = graph_emb

    def _get_input(self, t, y):
        B = y.shape[0]
        t_vec = t.expand(B, 1) if t.dim() == 0 else t.unsqueeze(-1)
        return torch.cat([y, self.graph_context, t_vec], dim=-1)

    def f(self, t, y):
        return self.drift_net(self._get_input(t, y))

    def g(self, t, y):
        return self.diffusion_net(self._get_input(t, y))


class LatentSDEModel(nn.Module):
    """Encoder + SDE + Decoder for latent path generation."""

    def __init__(self, n_stocks=50, latent_dim=64, T=60, rnn_hidden=64):
        super().__init__()
        self.T = T
        self.latent_dim = latent_dim
        self.n_stocks = n_stocks
        self.rnn_hidden = rnn_hidden

        self.encoder = nn.GRU(n_stocks, rnn_hidden, batch_first=True)
        self.sde = GraphConditionedSDE(
            latent_dim=rnn_hidden,
            graph_emb_dim=latent_dim,
            hidden_dim=128,
        )
        self.decoder = nn.Linear(rnn_hidden, n_stocks)

    def forward(self, x, graph_emb, ts=None):
        B, T, N = x.shape
        if ts is None:
            ts = torch.linspace(0, 1, T, device=x.device)

        _, h_n = self.encoder(x)
        z0 = h_n.squeeze(0)

        self.sde.set_graph_context(graph_emb)

        zs = torchsde.sdeint(
            self.sde, z0, ts,
            method="euler",
            dt=1.0 / T,
            names={"drift": "f", "diffusion": "g"},
        )
        zs = zs.permute(1, 0, 2)
        r_hat = self.decoder(zs)
        return r_hat, zs
```

### 6.2 `src/model.py` — Update

```python
"""Specification-compatible Vortex-AI model API."""

from models.gat_encoder import DynamicGATEncoder
from models.neural_sde import GraphConditionedSDE, LatentSDEModel

__all__ = ["DynamicGATEncoder", "LatentSDEModel", "GraphConditionedSDE"]
```

---

## 7. Success Criteria

| Criterion | Threshold | Verification Method |
|-----------|-----------|-------------------|
| No NaN/Inf in output | 100% finite | `torch.isfinite(r_hat).all()` |
| Output shape correct | `(B, T, 50)` | Assert shape |
| Single-stock test passes | GBM-like returns | Check mean ≈ 0, std ≈ 0.01 |
| 50-stock test passes | <30s CPU batch=8 | `time python -m models.neural_sde` |
| Gradient flow | Non-zero grads | `param.grad.abs().sum() > 0` for all params |

---

## 8. Risk Assessment & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| NaN in SDE integration | High (first run) | Block | Start with `dt=1/120`, `max_norm=1.0` clip, `eps=1e-8` in Softplus |
| torchsde API mismatch | Medium | Block | Use exact `SDEIto` subclass pattern from Part D.3 |
| CPU performance too slow | Medium | Delay | Reduce `rnn_hidden` to 32 for initial tests; use 50 stocks only after single-stock passes |
| Graph context dimension mismatch | Low | Block | Verify `graph_emb.shape == (B, 64)` matches `graph_emb_dim` |
| CUDA unavailable | High (current env) | Minor | Use `torch.device("cpu")`; plan GPU run for later phases |

---

## 9. Integration Points for Subsequent Phases

| Phase | Prerequisite from Phase 3 | What Phase 3 Delivers |
|-------|--------------------------|----------------------|
| 4 (Latent Encoder + Decoder) | ✓ | `LatentSDEModel` with encoder + decoder |
| 5 (Contrastive Regime Head) | ✓ | `zs` latent paths for projection + SupCon |
| 6 (Circuit Filter + Loss) | ✓ | `r_hat` decoded returns for clipping + stylized facts loss |
| 7 (Lightning Training Loop) | ✓ | Full `VORTEXModel` component ready for Lightning module |

---

## 10. Timeline

| Task | Estimated Time |
|------|---------------|
| Implement `GraphConditionedSDE` | 1.5 hours |
| Implement `LatentSDEModel` | 1 hour |
| Write single-stock sanity test | 0.5 hours |
| Debug NaN/performance issues | 1-2 hours |
| Run 50-stock validation | 0.5 hours |
| Update `src/model.py` exports | 0.25 hours |
| **Total** | **4.25-6.25 hours** |

**Target completion: 1 day of focused development.**
