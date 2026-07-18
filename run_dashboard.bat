@echo off
cd /d C:\Users\luked\mlb-show-roster-predictor
set PYTHONPATH=
set PYTHONHOME=
.venv_new\Scripts\python.exe -m streamlit run web/dashboard.py --server.port 8501
