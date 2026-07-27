@echo off
cd /d C:\Users\luked\mlb-show-roster-predictor
set PYTHONPATH=
set PYTHONHOME=

echo ============================================
echo MLB The Show 26 Roster Predictor
echo ============================================
echo.

if "%~1"=="--skip-predictions" goto launch_dashboard

echo [1/2] Running prediction pipeline (this takes ~2-3 minutes)...
echo.
.venv_new\Scripts\python.exe scripts/daily_predict.py --skip-cards --skip-link --fast
if errorlevel 1 (
    echo.
    echo ERROR: Prediction pipeline failed!
    pause
    exit /b 1
)
echo.
echo Predictions complete!
echo.

:launch_dashboard
echo [2/2] Launching dashboard on http://localhost:8501
echo.
.venv_new\Scripts\python.exe -m streamlit run web/dashboard.py --server.port 8501
