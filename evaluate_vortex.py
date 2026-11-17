"""
Comprehensive evaluation script for VORTEX-AI (CG-NSDE).

Runs all evaluation metrics from masterplan Part F:
1. Statistical fidelity tests
2. Discriminative score test
3. Contagion propagation test
4. CVaR regime ratio

Usage:
    python evaluate_vortex.py --checkpoint models/checkpoints/best_vortex.ckpt
    python evaluate_vortex.py --checkpoint models/checkpoints/best_vortex.ckpt --n_scenarios 500
    python evaluate_vortex.py --checkpoint models/checkpoints/best_vortex.ckpt --split val
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import json
from typing import Dict

from train_vortex import VortexDataset
from training.vortex_model import VORTEXModel
from eval.statistical import evaluate_stylized_facts, compare_distributions, test_no_autocorrelation_returns
from eval.discriminative import discriminative_score, predictive_score
from eval.contagion import evaluate_contagion, cvar_regime_ratio, granger_causality_test


def generate_scenarios(
    model: VORTEXModel,
    regimes: np.ndarray,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate one synthetic scenario per requested regime, using regime only.

    Returns:
        gen_returns: (N, T, n_stocks) generated returns
        regimes: (N,) the regime each scenario was generated for
    """
    model.eval()
    model.to(device)
    regime_t = torch.as_tensor(np.asarray(regimes), dtype=torch.long, device=device)
    gen = model.sample_regime_scenarios(regime_t, device=device)
    return gen.cpu().numpy(), np.asarray(regimes)


def load_generation_priors(model: VORTEXModel, path: Path) -> bool:
    """Install the per-regime generative prior saved by train_vortex.py."""
    if not path.exists():
        return False
    z = np.load(path)
    with torch.no_grad():
        model.prior_mu.copy_(torch.as_tensor(z["prior_mu"], dtype=model.prior_mu.dtype))
        model.prior_logvar.copy_(torch.as_tensor(z["prior_logvar"], dtype=model.prior_logvar.dtype))
        model.regime_graph_emb.copy_(torch.as_tensor(z["regime_graph_emb"], dtype=model.regime_graph_emb.dtype))
        model.regime_log_vol.copy_(torch.as_tensor(z["regime_log_vol"], dtype=model.regime_log_vol.dtype))
        if "regime_log_vol_std" in z and hasattr(model, "regime_log_vol_std"):
            model.regime_log_vol_std.copy_(torch.as_tensor(
                z["regime_log_vol_std"], dtype=model.regime_log_vol_std.dtype))
    print(f"Generation priors loaded from {path}")
    for r in (0, 1):
        print(f"  regime {r}: log_vol {model.regime_log_vol[r].item():+.3f}, "
              f"prior std {model.prior_logvar[r].exp().sqrt().mean().item():.3f}")
    return True


def run_evaluation(args: argparse.Namespace) -> Dict:
    """Run complete evaluation pipeline."""
    print("\n" + "="*80)
    print("VORTEX-AI (CG-NSDE) Comprehensive Evaluation")
    print("="*80 + "\n")
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    # Load model (strict=False: tolerates checkpoints saved before the 4.1
    # gat_fallback ablation module was added)
    model = VORTEXModel.load_from_checkpoint(args.checkpoint, strict=False)
    model.eval()
    
    # Load the requested split. Evaluation must run on a held-out split: the old
    # script loaded data/raw directly and scored the first N windows of the
    # whole panel, i.e. training-period data.
    split_name = args.split
    split_key = f"{split_name}_idx"
    print(f"Loading held-out {split_name} split...")
    dataset = VortexDataset(Path("data/raw"))
    split_path = Path("data/raw/split.npz")
    if split_path.exists():
        sp = np.load(split_path, allow_pickle=True)
        if split_key not in sp:
            raise KeyError(f"split.npz has no {split_key!r} (keys: {list(sp.keys())})")
        eval_idx = np.asarray(sp[split_key])
        scheme = str(sp["scheme"][0])
        print(f"{split_name} split: {scheme} | {len(eval_idx)} windows")
        eval_windows = dataset.windows[eval_idx]
        eval_regimes = dataset.regimes[eval_idx]
        print(f"{split_name} crisis rate: {eval_regimes.mean():.1%}")
        eval_dataset = torch.utils.data.Subset(dataset, eval_idx)
    else:
        print("Warning: split.npz missing; scoring the first windows of the full panel")
        scheme = "full-panel"
        eval_dataset = dataset
        eval_windows = dataset.windows
        eval_regimes = dataset.regimes

    n_scenarios = args.n_scenarios or len(eval_dataset)
    n_scenarios = min(n_scenarios, len(eval_dataset))
    # Regime labelling matches crisisbench.protocol.LABEL_RULE: the labels are
    # derived from the window's own returns (via window_regimes_v2.npy in the
    # dataset) and applied identically to real and generated windows. The
    # generator is asked for the same regime composition as the real evaluation
    # windows, and nothing else. Labels below are the regime that was requested.
    requested = np.asarray(eval_regimes[:n_scenarios], dtype=np.int64)
    real_returns = np.asarray(eval_windows[:n_scenarios], dtype=np.float32)

    priors_path = Path(args.priors) if args.priors else Path("models/checkpoints/generation_priors.npz")
    if not load_generation_priors(model, priors_path):
        print("WARNING: generation priors not found; the prior is whatever the checkpoint holds.")
        print("         Run train_vortex.py to produce models/checkpoints/generation_priors.npz.")

    print(f"\nGenerating {n_scenarios} synthetic scenarios from regime only...")
    gen_returns, regimes = generate_scenarios(model, requested, args.device)

    print(f"Real returns shape: {real_returns.shape}")
    print(f"Generated returns shape: {gen_returns.shape}")
    print(f"Requested regimes: Normal={np.sum(regimes==0)}, Crisis={np.sum(regimes==1)}")
    print("Note: labels are the requested regime (stress-testing use case), not derived from the draws.")

    # Real reference on the exact same windows: the gates compare against the
    # real value, because the absolute targets exceed what real data shows.
    from eval.contagion import real_regime_reference
    real_ref = real_regime_reference(real_returns, regimes)
    results = {"real_reference": real_ref,
               "generation": {"conditioning": "regime only",
                              "labels": "requested regime",
                              "n_scenarios": int(n_scenarios)},
               "split": {"name": split_name,
                         "scheme": scheme,
                         "n_windows": int(len(regimes)),
                         "crisis_rate": float(np.mean(regimes == 1))}}
    
    # 1. Statistical Fidelity Tests
    print("\n" + "-"*80)
    print("1. Statistical Fidelity Tests")
    print("-"*80)
    
    stylized_facts = evaluate_stylized_facts(real_returns, gen_returns)
    results["statistical_fidelity"] = stylized_facts
    
    print(f"Kurtosis (real): {stylized_facts['kurtosis_real']:.3f}  interval {stylized_facts['kurtosis_interval']}")
    print(f"Kurtosis (gen):  {stylized_facts['kurtosis_gen']:.3f}")
    print(f"Kurtosis pass:   {stylized_facts['kurtosis_pass']}")
    print(f"\nACF squared (real): {stylized_facts['acf_sq_real']:.4f}")
    print(f"ACF squared (gen):  {stylized_facts['acf_sq_gen']:.4f}")
    print(f"ACF pass:           {stylized_facts['acf_sq_pass']} (tol {stylized_facts['acf_sq_tolerance']:.4f})")
    print(f"\nCorrelation error: {stylized_facts['corr_error']:.3f} "
          f"(real-vs-real ceiling {stylized_facts['corr_error_ceiling']:.3f}, "
          f"limit {stylized_facts['corr_error_limit']:.3f})")
    print(f"Correlation pass:  {stylized_facts['corr_error_pass']}")
    print(f"\n[All tests pass: {stylized_facts['all_tests_pass']}]")
    
    # Distribution comparison
    dist_comparison = compare_distributions(real_returns, gen_returns)
    results["distribution_comparison"] = dist_comparison
    
    # Test no autocorrelation in returns
    returns_acf = test_no_autocorrelation_returns(gen_returns)
    results["returns_acf"] = returns_acf
    print(f"\nReturns ACF (should be ~0): {returns_acf['returns_acf_lag1']:.4f}")
    print(f"Returns uncorrelated: {returns_acf['returns_uncorrelated']}")
    
    # 2. Discriminative Score Test
    print("\n" + "-"*80)
    print("2. Discriminative Score Test")
    print("-"*80)
    
    disc_results = discriminative_score(real_returns, gen_returns, n_samples=min(500, n_scenarios))
    results["discriminative"] = disc_results

    # Reference: two halves of the real panel scored against each other. This is
    # the floor of the metric on this slice, not 0.5 by assumption.
    h = len(real_returns) // 2
    disc_ref = discriminative_score(real_returns[:h], real_returns[h:], n_samples=min(300, h)) \
        if h >= 20 else {"discriminative_score": float("nan")}
    results["discriminative_reference"] = disc_ref

    print(f"Discriminative score: {disc_results['discriminative_score']:.3f} ± {disc_results['discriminative_std']:.3f}")
    print(f"Real-vs-real floor:   {disc_ref['discriminative_score']:.3f}")
    print(f"Target: <0.60 (closer to 0.5 is better)")
    print(f"Pass: {disc_results['discriminative_pass']}")
    print(f"Indistinguishability: {disc_results['indistinguishability']:.3f} (1.0 = perfect)")
    
    # Predictive score
    print("\nPredictive Score Test...")
    pred_results = predictive_score(real_returns, gen_returns, n_samples=min(500, n_scenarios))
    results["predictive"] = pred_results
    print(f"Predictive R²: {pred_results['predictive_r2']:.4f}")
    print(f"Predictive MSE: {pred_results['predictive_mse']:.6f}")
    
    # 3. Contagion Propagation Test
    print("\n" + "-"*80)
    print("3. Contagion Propagation Test")
    print("-"*80)
    
    contagion_results = evaluate_contagion(gen_returns, regimes,
                                           real_result=real_ref["contagion"])
    results["contagion"] = contagion_results
    
    print(f"Normal correlation:  {contagion_results['corr_normal']:.4f} "
          f"(real {real_ref['contagion']['corr_normal']:.4f})")
    print(f"Crisis correlation:  {contagion_results['corr_crisis']:.4f} "
          f"(real {real_ref['contagion']['corr_crisis']:.4f})")
    print(f"Crisis boost:        {contagion_results['crisis_corr_boost']:.2%}")
    print(f"Real boost:          {contagion_results['crisis_corr_boost_real']:.2%}")
    print(f"Gate: {contagion_results['gate']}")
    print(f"Pass: {contagion_results['contagion_pass']}")
    
    # CVaR ratio
    print("\n" + "-"*80)
    print("4. CVaR Regime Ratio")
    print("-"*80)
    
    cvar_results = cvar_regime_ratio(gen_returns, regimes, real_result=real_ref["cvar"])
    results["cvar"] = cvar_results
    
    print(f"CVaR (normal): {cvar_results['cvar_normal']:.6f} "
          f"(real {real_ref['cvar']['cvar_normal']:.6f})")
    print(f"CVaR (crisis): {cvar_results['cvar_crisis']:.6f} "
          f"(real {real_ref['cvar']['cvar_crisis']:.6f})")
    print(f"CVaR ratio:    {cvar_results['cvar_ratio']:.2f}x")
    print(f"Real ratio:    {cvar_results['cvar_ratio_real']:.2f}x")
    print(f"Gate: {cvar_results['gate']}")
    print(f"Pass: {cvar_results['cvar_pass']}")
    
    # 5. Granger Causality (optional, can be slow)
    if args.test_granger:
        print("\n" + "-"*80)
        print("5. Granger Causality Test")
        print("-"*80)
        
        # Test on a single scenario
        granger_results = granger_causality_test(gen_returns[0])
        results["granger"] = granger_results
        
        print(f"Causality ratio: {granger_results['granger_causality_ratio']:.2%}")
        print(f"Significant pairs: {granger_results['granger_pairs_significant']}/{granger_results['granger_total_tests']}")
    
    # Summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    passes = {
        "Statistical Fidelity": stylized_facts['all_tests_pass'],
        "Discriminative Score": disc_results['discriminative_pass'],
        "Contagion Propagation": contagion_results['contagion_pass'],
        "CVaR Ratio": cvar_results['cvar_pass'],
    }
    
    for test_name, passed in passes.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test_name:.<50} {status}")
    
    total_pass = sum(passes.values())
    print(f"\n{'='*80}")
    print(f"Overall: {total_pass}/{len(passes)} tests passed")
    print(f"{'='*80}\n")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to native Python for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        results_serializable = convert_types(results)
        
        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"Results saved to: {output_path}")
    
    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate VORTEX-AI (CG-NSDE)")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="test",
                       choices=["train", "val", "test"],
                       help="Data split to evaluate (val = 2018-19 pre-2020 windows, "
                            "test = 2022+ holdout)")
    parser.add_argument("--n_scenarios", type=int, default=None,
                       help="Number of scenarios to generate (default: the whole test split)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for generation")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu", "mps"],
                       help="Device to use")
    parser.add_argument("--priors", type=str, default=None,
                       help="Path to generation_priors.npz (default models/checkpoints/)")
    parser.add_argument("--z0_source", type=str, default="aggregate_posterior",
                       choices=["aggregate_posterior", "posterior", "regime_prior"],
                       help="Where the initial latent state is drawn from")
    parser.add_argument("--output", type=str, default="results/evaluation_results.json",
                       help="Path to save results JSON")
    parser.add_argument("--test_granger", action="store_true",
                       help="Run Granger causality test (slow)")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = run_evaluation(args)
