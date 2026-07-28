@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

set "VENV_DIR=.venv_new"
set "PYTHON=%VENV_DIR%\Scripts\python.exe"
set "BACKEND_PORT=8000"
set "DASHBOARD_PORT=8501"

echo Starting MLB Show Roster Predictor...

start "Backend" cmd /c "%PYTHON% -m uvicorn src.api.main:app --host 0.0.0.0 --port %BACKEND_PORT% --log-level warning"

timeout /t 2 /nobreak >nul

start "Dashboard" cmd /c "%PYTHON% -m streamlit run web/dashboard.py --server.port %DASHBOARD_PORT% --browser.gatherUsageStats false"

timeout /t 3 /nobreak >nul

start "" "http://localhost:%DASHBOARD_PORT%"

echo.
echo Backend:  http://localhost:%BACKEND_PORT%
echo Dashboard: http://localhost:%DASHBOARD_PORT%
echo.
echo Close this window to stop both services.
echo.

pause
