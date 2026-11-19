@echo off
rem VORTEX Money: the consumer app + its backend (accounts, sync, fund data cache). Opens in your browser.
cd /d "%~dp0"
if not exist "app\dist\money.html" (cd app && call npx vite build && cd ..)
start "VORTEX Money server (close to stop)" /min cmd /c ".venv\Scripts\python.exe -m uvicorn money_api.app:app --host 127.0.0.1 --port 8100"
timeout /t 5 /nobreak >nul
start "" http://127.0.0.1:8100/
