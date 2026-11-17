#!/usr/bin/env python
"""Run VaR/ES backtest per protocol (crisisbench/run_risk.py).

This script runs the VaR/ES backtest and produces:
- results/06_risk/T4_risk.md (backtest results table)
- results/06_risk/T5_metric_validity.md (metric validity table)
- results/gates/GATE_G3.md (G3 gate result)
- results/06_risk/G3.json (detailed G3 gate data)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crisisbench.run_risk import main


if __name__ == "__main__":
    print("Running VaR/ES backtest...")
    main()
    print("VaR/ES backtest complete.")