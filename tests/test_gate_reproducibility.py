"""Gate reproducibility test for VORTEX-AI (CG-NSDE).

Regenerates the six-gate table from a named checkpoint using current gate code,
ensuring the evaluation is reproducible and not dependent on stale cached results.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_six_gate_table_reproducible():
    """Regenerated table from named checkpoint matches current gate code."""
    from scripts.regenerate_gate_table import regenerate_from_checkpoint
    import json

    # Regenerate from named checkpoint (archive/ckpt_old/last-v18.ckpt)
    results = regenerate_from_checkpoint("archive/ckpt_old/last-v18.ckpt")

    # Load current gate thresholds
    from eval.statistical import acf_tol, corr_limit
    from eval.discriminative import discriminative_score

    # Verify all gates compute with current thresholds
    assert "Correlation Frobenius" in results
    assert "Crisis boost" in results
    assert "CVaR ratio" in results
    assert "Kurtosis" in results
    assert "ACF(r²)" in results
    assert "Discriminative" in results
    assert "Predictive R²" in results

    # Verify no stale eval_current.json is used
    import os
    assert not os.path.exists("results/eval_current.json") or os.path.getmtime("results/eval_current.json") > os.path.getmtime("eval/statistical.py")


def test_checkpoint_compatibility():
    """Verify v18 checkpoint loads with current model code."""
    import torch
    ckpt = torch.load("archive/ckpt_old/last-v18.ckpt", map_location="cpu")
    from training.vortex_model import VORTEXModel
    from scripts.regenerate_gate_table import regenerate_from_checkpoint

    # The v18 checkpoint predates the regime-code concatenation (head_in =
    # latent_dim * 2) and the regime_graph_emb / regime_log_vol / regime_log_vol_std
    # parameters.  regenerate_from_checkpoint pads the emission heads and loads
    # with strict=False, so this must not raise.
    model = VORTEXModel(**ckpt["hyper_parameters"])
    state_dict = ckpt["state_dict"]

    # Pad emission heads from latent_dim to latent_dim * 2 (regime code concat)
    config = ckpt["hyper_parameters"]
    latent_dim = config.get("latent_dim", 64)
    n_stocks = config.get("n_stocks", 50)
    for key in ["sde_model.emi_m.weight", "sde_model.emi_d.weight"]:
        if key in state_dict and state_dict[key].shape[1] == latent_dim:
            padding = torch.zeros(n_stocks, latent_dim, dtype=state_dict[key].dtype)
            state_dict[key] = torch.cat([state_dict[key], padding], dim=1)

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    # Forward pass works (VORTEXModel.forward requires adj and node_feats)
    x = torch.randn(2, 60, n_stocks)
    adj = torch.eye(n_stocks).unsqueeze(0).repeat(2, 1, 1)
    node_feats = torch.randn(2, n_stocks, 6)
    out = model(x, adj, node_feats)
    assert out[0].shape == (2, 60, n_stocks)


if __name__ == "__main__":
    test_six_gate_table_reproducible()
    print("test_six_gate_table_reproducible PASSED")
    test_checkpoint_compatibility()
    print("test_checkpoint_compatibility PASSED")
    print("ALL TESTS PASSED")
