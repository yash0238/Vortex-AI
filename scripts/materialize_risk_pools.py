#!/usr/bin/env python
"""Materialise risk pools: 20,000 windows per generator and fit.

This script generates the risk pools needed for the VaR/ES backtest (G3 gate).
It uses the existing bench.run function with want_risk_pool=True.

Pools are saved to artifacts/risk_pool/<market>/<gen>/<fit>/seed0.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crisisbench import protocol as P
from crisisbench.bench import run
from crisisbench.generators import REGISTRY


def materialize_risk_pools(market: str = "nse"):
    """Generate risk pools for all generators and fits."""
    generators = list(REGISTRY.keys())
    fits = list(P.FITS)  # ("full", "pre2020")
    
    # Also need upto fits for the rolling backtest refits
    upto_fits = [f"upto{year}" for year in range(2019, 2024)]  # 2019-2023
    
    all_fits = fits + upto_fits
    
    print(f"Materializing risk pools for market={market}")
    print(f"Generators: {generators}")
    print(f"Fits: {all_fits}")
    print(f"Windows per pool: {P.N_GEN_POOL_RISK}")
    print()
    
    for gen_name in generators:
        for fit in all_fits:
            print(f"  {gen_name} / {fit} ... ", end="", flush=True)
            try:
                run(
                    market=market,
                    gen_name=gen_name,
                    fit=fit,
                    seed=0,
                    force=False,
                    want_risk_pool=True,
                    n_gen=P.N_GEN_POOL_RISK,
                )
                print("done")
            except SystemExit as e:
                if e.args[0] == 0:
                    print("skipped (claimed or dropped)")
                else:
                    raise
            except Exception as e:
                print(f"FAILED: {e}")
                # Continue with other generators
    
    print("\nRisk pool materialization complete!")
    print("Pools saved to artifacts/risk_pool/")


if __name__ == "__main__":
    materialize_risk_pools()