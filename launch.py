"""
Launcher for MLB Show Roster Predictor.
Fixes the Hermes agent's broken Python environment before launching Streamlit.
"""
import sys
import os

# ── CRITICAL: Clean paths BEFORE any third-party imports ──
# The Hermes agent injects its own broken numpy/pandas into sys.path.
# We must remove these paths before importing anything else.
_HERMES_BASE = r"C:\Users\luked\AppData\Local\hermes\hermes-agent"
_paths_removed = [p for p in sys.path if p.startswith(_HERMES_BASE)]
for p in _paths_removed:
    sys.path.remove(p)

# Also clean PYTHONPATH
if "PYTHONPATH" in os.environ:
    parts = os.environ["PYTHONPATH"].split(os.pathsep)
    clean = [p for p in parts if not p.startswith(_HERMES_BASE)]
    if clean:
        os.environ["PYTHONPATH"] = os.pathsep.join(clean)
    else:
        del os.environ["PYTHONPATH"]

# ── Now safe to import ────────────────────────────────────
import subprocess

os.chdir(r"C:\Users\luked\mlb-show-roster-predictor")

# Use our venv's Python to run streamlit
venv_python = os.path.join(os.path.dirname(__file__), ".venv_new", "Scripts", "python.exe")

# Verify it works
result = subprocess.run(
    [venv_python, "-c", "import numpy; import pandas; print('OK')"],
    capture_output=True, text=True
)
if result.returncode != 0:
    print(f"[ERROR] venv Python has issues:\n{result.stderr}")
    sys.exit(1)

print(f"[OK] Using Python: {venv_python}")
print("[OK] Starting Streamlit on http://localhost:8501 ...")

# Launch streamlit using our venv's Python
subprocess.run([
    venv_python, "-m", "streamlit", "run", "web/dashboard.py",
    "--server.port", "8501",
    "--global.developmentMode", "false",
])
