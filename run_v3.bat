@echo off
rem VORTEX-AI: CG-NSDE v3 jobs (resumable: finished jobs are skipped). Keep the laptop plugged in.
cd /d "%~dp0"
set PYTHONWARNINGS=ignore
title VORTEX-AI v3 experiments (do not close)
".venv\Scripts\python.exe" -m crisisbench.queue --stage v3_main,v3_risk,v3_ablation_held,v3_markets,v3_ablation --workers 6 --threads 2 >> results\queue_v3.log 2>&1
