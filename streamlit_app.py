"""Streamlit Community Cloud entrypoint.
Starts the FastAPI backend, then launches the UI.
"""
import os
import runpy
import subprocess
import sys
import time

import requests
import streamlit as st

BACKEND = "http://127.0.0.1:8000"


@st.cache_resource
def start_backend():
    # Copy current environment so secrets are available to the backend
    env = os.environ.copy()

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "app.main:app",
            "--host", "127.0.0.1",
            "--port", "8000",
        ],
        env=env,
    )

    # Wait up to 3 minutes for the backend (first start downloads models)
    for _ in range(180):
        if proc.poll() is not None:
            raise RuntimeError("Backend crashed on startup. Check the logs.")
        try:
            if requests.get(f"{BACKEND}/health", timeout=2).ok:
                return proc
        except requests.RequestException:
            pass
        time.sleep(1)

    raise RuntimeError("Backend did not become healthy in 3 minutes.")


with st.spinner("Starting backend (first load can take 1–2 minutes)..."):
    start_backend()

# Run your existing Streamlit UI
runpy.run_path("ui/app.py", run_name="__main__")