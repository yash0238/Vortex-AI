@echo off
rem VORTEX-AI dashboard for the demo: starts the live-generation API and the dashboard, then opens the browser.
cd /d "%~dp0"
start "VORTEX-AI live API (close to stop)" /min cmd /c ".venv\Scripts\python.exe -m uvicorn api.server:app --host 127.0.0.1 --port 8000"
start "VORTEX-AI dashboard (close to stop)" /min cmd /c "cd app && npx vite preview --port 4173 --host 127.0.0.1"
timeout /t 6 /nobreak >nul
start "" http://127.0.0.1:4173/
