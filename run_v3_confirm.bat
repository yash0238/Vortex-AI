@echo off
cd /d "%~dp0"
set PYTHONWARNINGS=ignore
title VORTEX-AI v3 confirmatory runs (do not close)
".venv\Scripts\python.exe" -m crisisbench.queue --stage v3_confirm --workers 6 --threads 2 >> results\queue_v3.log 2>&1
