"""
SpikeGate — Streamlit Community Cloud Root Entry Point.
Directs Streamlit to dashboard/app.py while ensuring sys.path is configured.
"""
import os
import sys
from pathlib import Path

# Ensure root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Execute dashboard/app.py
app_path = ROOT_DIR / "dashboard" / "app.py"
with open(app_path, "r", encoding="utf-8") as f:
    code = compile(f.read(), str(app_path), "exec")
    exec(code, globals())
