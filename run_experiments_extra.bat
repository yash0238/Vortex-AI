@echo off
rem VORTEX-AI: one extra worker next to run_experiments.bat. Takes jobs from the END of the list; jobs are claimed,
rem so the two windows never run the same job. Safe to close; finished jobs are kept.
cd /d "%~dp0"
set PYTHONWARNINGS=ignore
title VORTEX-AI experiments, extra worker (do not close)
".venv\Scripts\python.exe" -m crisisbench.queue --stage all --workers 1 --threads 2 --reverse
pause
