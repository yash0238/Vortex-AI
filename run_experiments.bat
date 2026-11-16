@echo off
rem VORTEX-AI: run (or resume) every benchmark experiment. Safe to close and re-run: finished jobs are skipped.
rem Keep the laptop plugged in. The script keeps Windows awake while it runs.
cd /d "%~dp0"
set PYTHONWARNINGS=ignore
title VORTEX-AI experiments (do not close)
echo Started %date% %time% >> results\queue_console.log
".venv\Scripts\python.exe" -m crisisbench.queue --list
".venv\Scripts\python.exe" -m crisisbench.queue --stage all --workers 5 --threads 2
echo.
echo ===== Queue pass finished. Remaining work: =====
".venv\Scripts\python.exe" -m crisisbench.queue --list
echo Finished %date% %time% >> results\queue_console.log
pause
