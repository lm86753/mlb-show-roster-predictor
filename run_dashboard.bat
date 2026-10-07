@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

set "BACKEND_PORT=8000"
set "FRONTEND_PORT=3000"

rem Usage: run_dashboard.bat [dev]
rem   (no args) -> production build + preview (fast, matches deployment)
rem   dev       -> Vite dev server with hot reload

rem ── Find a Python that has fastapi + uvicorn ──────────────────────────────
set "PYTHON="
if exist ".venv_new\Scripts\python.exe" (
  ".venv_new\Scripts\python.exe" -c "import fastapi, uvicorn" >nul 2>&1 && set "PYTHON=.venv_new\Scripts\python.exe"
)
if not defined PYTHON (
  python3.12 -c "import fastapi, uvicorn" >nul 2>&1 && set "PYTHON=python3.12"
)
if not defined PYTHON (
  python -c "import fastapi, uvicorn" >nul 2>&1 && set "PYTHON=python"
)
if not defined PYTHON (
  echo.
  echo ERROR: No Python with fastapi+uvicorn found.
  echo Install with:  pip install fastapi uvicorn
  pause
  exit /b 1
)
echo Using Python: %PYTHON%

echo Starting MLB Show Roster Predictor...
if /i "%~1"=="dev" ( echo Mode: dev ^(hot reload^) ) else ( echo Mode: production build )

rem ── Skip services that are already running ────────────────────────────────
netstat -ano | findstr /r /c:":%BACKEND_PORT% .*LISTENING" >nul
if %errorlevel% equ 0 (
  echo Backend already running on port %BACKEND_PORT% - skipping.
  set "SKIP_BACKEND=1"
) else (
  start "Backend" cmd /k "cd /d %~dp0 && %PYTHON% -m uvicorn src.api.main:app --host 0.0.0.0 --port %BACKEND_PORT% --log-level warning"
)

timeout /t 3 /nobreak >nul

rem ── Frontend (React + Vite) ────────────────────────────────────────────────
if not exist "web\node_modules" (
  echo Installing frontend dependencies ^(first run^)...
  pushd web
  call npm install
  if errorlevel 1 (
    popd
    echo.
    echo ERROR: npm install failed. Check the web\package.json and Node.js install.
    pause
    exit /b 1
  )
  popd
)

netstat -ano | findstr /r /c:":%FRONTEND_PORT% .*LISTENING" >nul
if %errorlevel% equ 0 (
  echo Frontend already running on port %FRONTEND_PORT% - skipping.
) else if /i "%~1"=="dev" (
  start "Frontend" cmd /k "cd /d %~dp0web && npm run dev"
) else (
  start "Frontend" cmd /k "cd /d %~dp0web && npm run build && npm run preview -- --port %FRONTEND_PORT%"
)

timeout /t 4 /nobreak >nul

start "" "http://localhost:%FRONTEND_PORT%"

echo.
echo Backend:   http://localhost:%BACKEND_PORT%
echo Dashboard: http://localhost:%FRONTEND_PORT%
echo.
echo Frontend window rebuilds on each start; close the Frontend/Backend
echo windows to stop the services, or close this window first.
echo.

pause
