@echo off
cd /d "%~dp0"
set PYTHONWARNINGS=ignore
title VORTEX-AI strong baselines (do not close)
".venv\Scripts\python.exe" -m crisisbench.queue --stage strong --workers 6 --threads 2 >> results\queue_strong.log 2>&1
