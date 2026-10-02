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
    dataloader: DataLoader,
    n_scenarios: int = 100,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic scenarios using trained model.
    
    Returns:
        real_returns: (N, T, n_stocks) real returns
        gen_returns: (N, T, n_stocks) generated returns
        regimes: (N,) regime labels
        adj_learned: (N, n_stocks, n_stocks) learned adjacency
    """
    model.eval()
    model.to(device)
    
    real_list = []
    gen_list = []
    regime_list = []
    adj_list = []
    
    with torch.no_grad():
        for batch in dataloader:
            if len(real_list) >= n_scenarios:
                break
            
            # Move batch to device
            x = batch["returns"].to(device)
            adj = batch["adj_matrix"].to(device)
            node_feats = batch["node_features"].to(device)
            regimes = batch["regimes"].to(device)
            
            # Generate synthetic data
            r_hat, A_learned, _ = model(x, adj, node_feats)
            
            # Store
            real_list.append(x.cpu().numpy())
            gen_list.append(r_hat.cpu().numpy())
            regime_list.append(regimes.cpu().numpy())
            adj_list.append(A_learned.cpu().numpy())
    
    # Concatenate
    real_returns = np.concatenate(real_list, axis=0)[:n_scenarios]
    gen_returns = np.concatenate(gen_list, axis=0)[:n_scenarios]
    regimes = np.concatenate(regime_list, axis=0)[:n_scenarios]
    adj_learned = np.concatenate(adj_list, axis=0)[:n_scenarios]
    
    return real_returns, gen_returns, regimes, adj_learned


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
    
    # Load test dataset
    print("Loading test dataset...")
    dataset = VortexDataset(Path("data/raw"))
    
    # Create test dataloader
    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )
    
    # Generate scenarios
    print(f"\nGenerating {args.n_scenarios} synthetic scenarios...")
    real_returns, gen_returns, regimes, adj_learned = generate_scenarios(
        model, test_loader, args.n_scenarios, args.device
    )
    
    print(f"Real returns shape: {real_returns.shape}")
    print(f"Generated returns shape: {gen_returns.shape}")
    print(f"Regimes: Normal={np.sum(regimes==0)}, Crisis={np.sum(regimes==1)}")
    
    results = {}
    
    # 1. Statistical Fidelity Tests
    print("\n" + "-"*80)
    print("1. Statistical Fidelity Tests")
    print("-"*80)
    
    stylized_facts = evaluate_stylized_facts(real_returns, gen_returns)
    results["statistical_fidelity"] = stylized_facts
    
    print(f"Kurtosis (real): {stylized_facts['kurtosis_real']:.3f}")
    print(f"Kurtosis (gen):  {stylized_facts['kurtosis_gen']:.3f}")
    print(f"Kurtosis pass:   {stylized_facts['kurtosis_pass']}")
    print(f"\nACF squared (real): {stylized_facts['acf_sq_real']:.4f}")
    print(f"ACF squared (gen):  {stylized_facts['acf_sq_gen']:.4f}")
    print(f"ACF pass:           {stylized_facts['acf_sq_pass']}")
    print(f"\nCorrelation error: {stylized_facts['corr_error']:.3f}")
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
    
    disc_results = discriminative_score(real_returns, gen_returns, n_samples=min(500, args.n_scenarios))
    results["discriminative"] = disc_results
    
    print(f"Discriminative score: {disc_results['discriminative_score']:.3f} ± {disc_results['discriminative_std']:.3f}")
    print(f"Target: <0.60 (closer to 0.5 is better)")
    print(f"Pass: {disc_results['discriminative_pass']}")
    print(f"Indistinguishability: {disc_results['indistinguishability']:.3f} (1.0 = perfect)")
    
    # Predictive score
    print("\nPredictive Score Test...")
    pred_results = predictive_score(real_returns, gen_returns, n_samples=min(500, args.n_scenarios))
    results["predictive"] = pred_results
    print(f"Predictive R²: {pred_results['predictive_r2']:.4f}")
    print(f"Predictive MSE: {pred_results['predictive_mse']:.6f}")
    
    # 3. Contagion Propagation Test
    print("\n" + "-"*80)
    print("3. Contagion Propagation Test")
    print("-"*80)
    
    contagion_results = evaluate_contagion(gen_returns, regimes)
    results["contagion"] = contagion_results
    
    print(f"Normal correlation:  {contagion_results['corr_normal']:.4f}")
    print(f"Crisis correlation:  {contagion_results['corr_crisis']:.4f}")
    print(f"Crisis boost:        {contagion_results['crisis_corr_boost']:.2%}")
    print(f"Target: >50% boost")
    print(f"Pass: {contagion_results['contagion_pass']}")
    
    # CVaR ratio
    print("\n" + "-"*80)
    print("4. CVaR Regime Ratio")
    print("-"*80)
    
    cvar_results = cvar_regime_ratio(gen_returns, regimes)
    results["cvar"] = cvar_results
    
    print(f"CVaR (normal): {cvar_results['cvar_normal']:.6f}")
    print(f"CVaR (crisis): {cvar_results['cvar_crisis']:.6f}")
    print(f"CVaR ratio:    {cvar_results['cvar_ratio']:.2f}x")
    print(f"Target: >3x")
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
    parser.add_argument("--n_scenarios", type=int, default=100,
                       help="Number of scenarios to generate")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for generation")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu", "mps"],
                       help="Device to use")
    parser.add_argument("--output", type=str, default="results/evaluation_results.json",
                       help="Path to save results JSON")
    parser.add_argument("--test_granger", action="store_true",
                       help="Run Granger causality test (slow)")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = run_evaluation(args)
