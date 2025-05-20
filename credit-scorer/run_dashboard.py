#!/usr/bin/env python3
"""Entry point script to run the Streamlit dashboard application."""

import os
import subprocess
import sys

# Ensure the project root is in the Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def run_dashboard():
    """Run the Streamlit dashboard."""
    # Set PYTHONPATH environment variable to include the project root
    env = os.environ.copy()

    # Append the current directory to PYTHONPATH if it exists,
    # otherwise set it to the current directory
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = PROJECT_ROOT

    script_path = os.path.join(PROJECT_ROOT, "streamlit_app", "main.py")
    cmd = ["streamlit", "run", script_path]

    print(f"Starting dashboard at: {script_path}")
    print(f"PYTHONPATH: {env['PYTHONPATH']}")
    subprocess.run(cmd, env=env)


if __name__ == "__main__":
    run_dashboard()
