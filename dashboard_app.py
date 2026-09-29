"""
Root launcher for AEGIS Cyclone Tactical Dashboard
===================================================
Run:
    python dashboard_app.py
Then open http://localhost:5000
"""

import os
import sys

base_dir = os.path.dirname(os.path.abspath(__file__))
code_files_dir = os.path.join(base_dir, "Code-files")
sys.path.insert(0, code_files_dir)

from dashboard_app import run_server

if __name__ == "__main__":
    run_server()
