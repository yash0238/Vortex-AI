#!/usr/bin/env python
"""
Smoke-test wrapper for TimeGAN on NIFTY-50 windows.

This script loads timegan_windows.npy, runs a minimal TimeGAN training
to verify the pipeline works end-to-end, and saves results.

Usage:
  python baselines/run_timegan_smoke.py \
    --input baselines/data/timegan_windows.npy \
    --subset 256 \
    --iterations 5 \
    --output-dir baselines/results/timegan_smoke
"""

import argparse
import sys
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

# Add TimeGAN repo to path
timegan_repo = Path(__file__).parent / 'external' / 'TimeGAN'
sys.path.insert(0, str(timegan_repo))

from timegan import timegan


def load_nifty_windows(data_path, subset_size=None, verbose=True):
    """Load timegan_windows.npy and validate shape."""
    data = np.load(data_path)
    
    if verbose:
        print(f"Loaded {data_path}")
        print(f"  Shape: {data.shape} (samples, seq_len, features)")
        print(f"  Dtype: {data.dtype}")
        print(f"  Finite: {np.isfinite(data).all()}")
        print(f"  Min: {data.min():.4f}, Max: {data.max():.4f}")
    
    # Validate shape is 3D
    if data.ndim != 3:
        raise ValueError(f"Expected 3D array, got {data.ndim}D with shape {data.shape}")
    
    # Take subset for smoke test
    if subset_size is not None:
        subset_size = min(subset_size, data.shape[0])
        data = data[:subset_size]
        if verbose:
            print(f"Using subset: {data.shape[0]} samples")
    
    # Convert to list-of-arrays format expected by TimeGAN
    data_list = [data[i].astype(np.float32) for i in range(data.shape[0])]
    return data_list


def run_timegan_smoke(data, parameters, verbose=True):
    """Run TimeGAN training with smoke-test parameters."""
    if verbose:
        print("\n" + "="*60)
        print("Starting TimeGAN Smoke Test")
        print("="*60)
        print(f"Parameters: {parameters}")
        print("="*60 + "\n")
    
    try:
        generated_data = timegan(data, parameters)
        
        if verbose:
            print("\n" + "="*60)
            print("TimeGAN Training Completed Successfully")
            print("="*60)
            print(f"Generated {len(generated_data)} samples")
            if len(generated_data) > 0:
                first_sample = generated_data[0]
                print(f"First sample shape: {first_sample.shape}")
                print(f"First sample dtype: {first_sample.dtype}")
                print(f"First sample finite: {np.isfinite(first_sample).all()}")
            print("="*60 + "\n")
        
        return generated_data
    
    except Exception as e:
        print(f"\nERROR during TimeGAN training: {e}")
        import traceback
        traceback.print_exc()
        raise


def save_results(generated_data, original_data, output_dir, verbose=True):
    """Save generated samples and metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save generated data as stacked array
    generated_array = np.array([np.asarray(g, dtype=np.float32) for g in generated_data], dtype=object)
    gen_path = output_dir / "generated_samples.npy"
    np.save(gen_path, generated_array, allow_pickle=True)
    
    # Compute basic statistics
    gen_stacked = np.vstack([np.asarray(g) for g in generated_data if g.size > 0])
    orig_stacked = np.vstack([np.asarray(o) for o in original_data if o.size > 0])
    
    stats = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "num_generated_samples": len(generated_data),
        "generated_shape": str(gen_stacked.shape),
        "original_shape": str(orig_stacked.shape),
        "generated_mean": float(gen_stacked.mean()),
        "generated_std": float(gen_stacked.std()),
        "original_mean": float(orig_stacked.mean()),
        "original_std": float(orig_stacked.std()),
        "generated_min": float(gen_stacked.min()),
        "generated_max": float(gen_stacked.max()),
        "original_min": float(orig_stacked.min()),
        "original_max": float(orig_stacked.max()),
    }
    
    stats_path = output_dir / "stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    if verbose:
        print(f"Results saved to {output_dir}")
        print(f"  Generated samples: {gen_path}")
        print(f"  Statistics: {stats_path}")
        print(f"\nStatistics:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="TimeGAN smoke test on NIFTY-50 windows"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to timegan_windows.npy"
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=256,
        help="Number of samples to use (for speed)"
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=60,
        help="Expected sequence length"
    )
    parser.add_argument(
        "--features",
        type=int,
        default=50,
        help="Expected number of features"
    )
    parser.add_argument(
        "--module",
        type=str,
        default="gru",
        choices=["gru", "lstm", "lstmLN"],
        help="RNN module type"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=24,
        help="Hidden dimension"
    )
    parser.add_argument(
        "--num-layer",
        type=int,
        default=3,
        help="Number of RNN layers"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of training iterations"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="baselines/results/timegan_smoke",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    print("TimeGAN Smoke Test Configuration:")
    print(f"  Input: {args.input}")
    print(f"  Subset size: {args.subset}")
    print(f"  Seq len: {args.seq_len}")
    print(f"  Features: {args.features}")
    print(f"  Module: {args.module}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Num layers: {args.num_layer}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Output dir: {args.output_dir}\n")
    
    # Load data
    original_data = load_nifty_windows(args.input, subset_size=args.subset, verbose=True)
    
    # Validate dimensions
    if len(original_data) == 0:
        raise ValueError("No data loaded")
    
    first_sample = original_data[0]
    if first_sample.shape[0] != args.seq_len:
        raise ValueError(
            f"Expected seq_len={args.seq_len}, "
            f"but data has {first_sample.shape[0]}"
        )
    if first_sample.shape[1] != args.features:
        raise ValueError(
            f"Expected features={args.features}, "
            f"but data has {first_sample.shape[1]}"
        )
    
    # Configure parameters
    parameters = {
        'module': args.module,
        'hidden_dim': args.hidden_dim,
        'num_layer': args.num_layer,
        'iterations': args.iterations,
        'batch_size': args.batch_size,
    }
    
    # Run TimeGAN
    generated_data = run_timegan_smoke(original_data, parameters, verbose=True)
    
    # Save results
    stats = save_results(generated_data, original_data, args.output_dir, verbose=True)
    
    print("\n✓ Smoke test completed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
