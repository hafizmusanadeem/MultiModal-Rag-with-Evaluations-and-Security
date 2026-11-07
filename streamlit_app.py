"""Streamlit Community Cloud entrypoint.
Starts the FastAPI backend (once), then runs the real UI in ui/app.py.

Main file path on Streamlit Cloud: streamlit_app.py
"""
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path

import requests
import streamlit as st

# MUST be the first Streamlit command. Any st.* call before it (including the
# spinner that @st.cache_resource shows by default) makes ui/app.py's own
# set_page_config() raise, which is what breaks the UI on Cloud.
st.set_page_config(page_title="JAFZA Rules Assistant", page_icon="✦", layout="wide", initial_sidebar_state="collapsed")

ROOT = Path(__file__).resolve().parent
BACKEND = "http://127.0.0.1:8000"
LOG_PATH = "/tmp/backend.log"
STARTUP_TIMEOUT_S = 180


def _healthy() -> bool:
    try:
        return requests.get(f"{BACKEND}/health", timeout=2).ok
    except requests.RequestException:
        return False


@st.cache_resource(show_spinner=False)
def _backend_state() -> dict:
    return {"proc": None}


def _log_tail(lines: int = 40) -> str:
    try:
        with open(LOG_PATH, "r", errors="ignore") as f:
            return "".join(f.readlines()[-lines:])
    except OSError:
        return "(no backend log found)"


def ensure_backend() -> None:
    """Start the backend if it isn't running; restart it if it died."""
    state = _backend_state()
    proc = state["proc"]

    if proc is not None and proc.poll() is None and _healthy():
        return

    if proc is None or proc.poll() is not None:
        env = os.environ.copy()                  # Cloud secrets are exposed as env vars
        env.setdefault("REQUIRE_GEMINI", "1")    # never fall back to the 768-dim local model
        log = open(LOG_PATH, "ab")
        state["proc"] = proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app",
             "--host", "127.0.0.1", "--port", "8000"],
            cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT,
        )

    deadline = time.time() + STARTUP_TIMEOUT_S
    while time.time() < deadline:
        if proc.poll() is not None:
            st.error("The backend crashed on startup. Last log lines:")
            st.code(_log_tail(), language=None)
            st.stop()
        if _healthy():
            return
        time.sleep(1)

    st.error("The backend did not become healthy in time. Last log lines:")
    st.code(_log_tail(), language=None)
    st.stop()


with st.spinner("Starting the assistant (first load can take 1–2 minutes)..."):
    ensure_backend()

runpy.run_path(str(ROOT / "ui" / "app.py"), run_name="__main__")