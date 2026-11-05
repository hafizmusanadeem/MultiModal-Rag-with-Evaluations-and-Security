"""
Streamlit frontend for the JAFZA Rules Assistant.

Run locally (backend already running on :8000):
    streamlit run ui/app.py
On Streamlit Cloud it is launched through streamlit_app.py.
"""
import os
import uuid

import requests
import streamlit as st
from streamlit.errors import StreamlitAPIException


def _cfg(name: str, default: str = "") -> str:
    try:
        return st.secrets[name]          # Streamlit Cloud secrets
    except Exception:
        return os.getenv(name, default)  # local env / .env


BACKEND_URL = _cfg("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = _cfg("API_KEY")
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}
REQUEST_TIMEOUT = 120

APP_NAME = "JAFZA Rules Assistant"
CORPUS_NAME = "Jebel Ali Free Zone Rules, 9th Edition (March 2023)"

# Three one-click starter questions (shown until the first message is sent).
SUGGESTIONS = [
    "What conditions must a customer meet to renew a Licence?",
    "What fines apply for cooking in staff accommodation?",
    "How is the guarantee for a sponsored employee calculated?",
]

# What the assistant covers, shown in the sidebar.
TOPICS = {
    "Licences & leases": "licence types and renewal, sub-leasing, lease and operational deadlines",
    "Employees & labour": "sponsorship, passes, overtime, termination, gratuity, WPS, absconding employees",
    "Inspections & fines": "inspection rules, violation penalties, notices, the violation fine list",
    "Health, safety & environment": "accident reporting, waste disposal, containers, fire safety",
    "Insurance": "building, third-party liability and workmen's compensation cover",
    "Accommodation, traffic & security": "room rules, occupancy, parking, vehicle registration, access and photography",
    "Tariff & fees": "registration, licence, visa, permit and administrative service fees",
}

try:
    st.set_page_config(page_title=APP_NAME, page_icon="📘", layout="centered")
except StreamlitAPIException:
    pass  # already set by streamlit_app.py

# ---------------------------------------------------------------- styling
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Schibsted+Grotesk:wght@400;500;700&family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap');

:root {
  color-scheme: light;
  --ink: #000000;
  --muted: #333333;
  --paper: #ffffff;
  --panel: #ffffff;
  --line: #d9e0e3;
  --accent: #1d6a85;
  --accent-soft: #e2eff3;
  --amber: #8a5a10;
  --amber-soft: #fbf0dc;
}

/* White page, black text everywhere (overrides the browser's dark mode) */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stBottom"], [data-testid="stBottom"] > div {
  background: var(--paper) !important;
  font-family: 'Schibsted Grotesk', system-ui, sans-serif;
}
.stApp p, .stApp span, .stApp li, .stApp label, .stApp div, .stApp a,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
.stApp summary, .stApp small, .stApp code, .stApp pre,
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {
  color: var(--ink) !important;
}
header[data-testid="stHeader"] { background: transparent !important; }

/* Tighter page spacing so less scrolling is needed */
.block-container { padding-top: 2rem !important; padding-bottom: 1rem !important; }

h1.app-title { font-family: 'Newsreader', Georgia, serif; font-weight: 600; font-size: 1.9rem;
  letter-spacing: -0.01em; margin: 0 0 .1rem; padding: 0; }
.app-sub { margin: 0 0 .6rem; max-width: 70ch; font-size: .95rem; }

.hint { background: var(--accent-soft); border: 1px solid var(--line); border-left: 4px solid var(--accent);
  border-radius: 8px; padding: .5rem .8rem; margin-bottom: .6rem; font-size: .88rem; line-height: 1.4; }

[data-testid="stChatMessage"] { background: var(--panel) !important; border: 1px solid var(--line);
  border-radius: 10px; padding: .8rem 1rem; }
[data-testid="stChatMessage"] p { line-height: 1.6; max-width: 72ch; }

/* Chat input */
[data-testid="stChatInput"], [data-testid="stChatInput"] > div { background: #ffffff !important; }
[data-testid="stChatInput"] textarea { color: #000000 !important; background: #ffffff !important;
  -webkit-text-fill-color: #000000 !important; }
[data-testid="stChatInput"] textarea::placeholder { color: #555555 !important; }

.stApp .chip { display: inline-block; font-size: .78rem; padding: .12rem .55rem; border-radius: 999px;
  background: var(--accent-soft); color: var(--accent) !important; margin: 0 .35rem .35rem 0; font-weight: 500; }
.stApp .chip.cache { background: var(--amber-soft); color: var(--amber) !important; }

.source { border-left: 3px solid var(--accent); background: #f4f6f7; padding: .6rem .8rem;
  margin-bottom: .6rem; border-radius: 0 6px 6px 0; font-size: .88rem; line-height: 1.5;
  white-space: pre-wrap; }
.source-label { font-size: .78rem; margin-bottom: .25rem; font-weight: 500; }

/* Sidebar */
section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div {
  background: #f1f4f5 !important; border-right: 1px solid var(--line); }
section[data-testid="stSidebar"] .block-container { padding-top: 1rem !important; }
.topic { font-size: .82rem; line-height: 1.35; margin-bottom: .45rem; }
.topic b { display: block; }
.status-dot { display:inline-block; width:.6rem; height:.6rem; border-radius:50%; margin-right:.45rem; }
.ok { background:#2e9e6b; } .down { background:#c0443a; }

/* Code blocks (thread id) */
[data-testid="stCode"], [data-testid="stCode"] pre, [data-testid="stCode"] code {
  background: #ffffff !important; color: #000000 !important; }

/* Expanders */
[data-testid="stExpander"] { background: #ffffff !important; border: 1px solid var(--line) !important;
  border-radius: 8px; }
[data-testid="stExpander"] summary { background: #ffffff !important; }

/* Buttons */
.stButton > button { width: 100%; text-align: left; border-radius: 8px; border: 1px solid var(--line);
  background: #ffffff !important; color: #000000 !important; font-weight: 500; }
.stButton > button p { color: #000000 !important; }
.stButton > button:hover { border-color: var(--accent); }
.stButton > button:hover p { color: var(--accent) !important; }
.stButton > button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"web-{uuid.uuid4().hex[:8]}"
if "messages" not in st.session_state:
    st.session_state.messages = []  # {role, content, sources, steps, error}


class ApiError(Exception):
    """The backend answered with a non-2xx status; `message` is safe to show the user."""

    def __init__(self, message: str, status: int | None = None, error_id: str | None = None):
        super().__init__(message)
        self.message = message
        self.status = status
        self.error_id = error_id

    def display(self) -> str:
        text = self.message
        if self.error_id:
            text += f"\n\n`Reference: {self.error_id}`"
        return text


def _api_error_from(resp: requests.Response) -> ApiError:
    """Translate the backend's error body into an ApiError."""
    try:
        body = resp.json()
    except ValueError:
        return ApiError(f"The server returned an unexpected response ({resp.status_code}).", resp.status_code)

    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        err = body["error"]
        return ApiError(
            err.get("message") or "The request failed.",
            resp.status_code,
            err.get("error_id"),
        )

    if resp.status_code == 422 and isinstance(body, dict) and isinstance(body.get("detail"), list):
        first = body["detail"][0].get("msg", "invalid input") if body["detail"] else "invalid input"
        return ApiError(f"Your question couldn't be accepted: {first}.", 422)

    return ApiError(f"The server returned an unexpected response ({resp.status_code}).", resp.status_code)


def backend_ok() -> bool:
    try:
        return requests.get(f"{BACKEND_URL}/health", timeout=3).ok  # /health is open: no key needed
    except requests.RequestException:
        return False


def ask(question: str) -> dict:
    resp = requests.post(
        f"{BACKEND_URL}/query",
        json={"q": question, "thread_id": st.session_state.thread_id},
        headers=HEADERS, timeout=REQUEST_TIMEOUT,
    )
    if not resp.ok:
        raise _api_error_from(resp)
    return resp.json()


def submit_feedback(trace_id: str, rating: str, comment: str | None = None):
    response = requests.post(
        f"{BACKEND_URL}/feedback",
        json={"trace_id": trace_id, "rating": rating, "comment": comment},
        headers=HEADERS, timeout=10,
    )
    if not response.ok:
        raise _api_error_from(response)
    return response.json()


def render_extras(sources: list[str], steps: list[str]) -> None:
    """Chips for the agent's steps + an expander with the retrieved passages."""
    if steps:
        chips = "".join(
            f'<span class="chip {"cache" if "Cache" in s else ""}">{s}</span>' for s in steps
        )
        st.markdown(chips, unsafe_allow_html=True)
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for i, src in enumerate(sources, 1):
                text = src.removeprefix("CONTENT: ").strip()
                st.markdown(
                    f'<div class="source-label">Passage {i}</div>'
                    f'<div class="source">{text.replace("<", "&lt;")}</div>',
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown(f"### {APP_NAME}")
    online = backend_ok()
    st.markdown(
        f'<span class="status-dot {"ok" if online else "down"}"></span>'
        f'{"API connected" if online else "API unreachable"}',
        unsafe_allow_html=True,
    )
    if not online:
        st.caption(f"Backend not reachable at {BACKEND_URL}.")

    st.divider()
    st.markdown("**What can I ask?**")
    st.caption("JAFZA rules, requirements, fees and penalties. Other topics are out of scope.")
    topics_html = "".join(
        f'<div class="topic"><b>{t}</b>{d}</div>' for t, d in TOPICS.items()
    )
    st.markdown(topics_html, unsafe_allow_html=True)

    st.divider()
    st.caption("Knowledge base")
    st.markdown(f"**{CORPUS_NAME}**")
    st.caption("Answers come only from this document.")

    st.divider()
    st.caption("Conversation")
    st.code(st.session_state.thread_id, language=None)
    if st.button("New conversation"):
        st.session_state.thread_id = f"web-{uuid.uuid4().hex[:8]}"
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------- main
st.markdown('<h1 class="app-title">Ask about the JAFZA Rules</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-sub">Answers are grounded in the Jebel Ali Free Zone Rules, and every '
    "reply shows the passages it used.</p>",
    unsafe_allow_html=True,
)
if not st.session_state.messages:
    st.markdown(
        '<div class="hint">See the sidebar for the topics you can ask about, or try a starter question below.</div>',
        unsafe_allow_html=True,
    )

for msg_index, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg.get("error"):
            st.error(msg["content"])
        else:
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_extras(msg.get("sources", []), msg.get("steps", []))
                if msg.get("trace_id") and not msg.get("feedback_sent"):
                    up_col, down_col = st.columns(2)
                    try:
                        if up_col.button("👍 Helpful", key=f"feedback_up_{msg_index}"):
                            submit_feedback(msg["trace_id"], "up")
                            msg["feedback_sent"] = True
                            st.success("Thanks for your feedback.")
                            st.rerun()
                        if down_col.button("👎 Not helpful", key=f"feedback_down_{msg_index}"):
                            submit_feedback(msg["trace_id"], "down")
                            msg["feedback_sent"] = True
                            st.success("Thanks for your feedback.")
                            st.rerun()
                    except (ApiError, requests.RequestException):
                        st.warning("Feedback could not be saved right now.")

pending = None
if not st.session_state.messages:
    for i, s in enumerate(SUGGESTIONS):
        if st.button(s, key=f"sugg{i}"):
            pending = s

typed = st.chat_input("Ask about JAFZA, e.g. “What is the fine for hindering an inspection?”")
question = typed or pending

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        is_error = False
        sources: list[str] = []
        steps: list[str] = []
        trace_id = None
        with st.spinner("Searching the JAFZA Rules and drafting an answer…"):
            try:
                data = ask(question)
                answer = data.get("answer") or "No answer was returned."
                sources = data.get("sources", [])
                steps = data.get("thought_process", [])
                trace_id = data.get("trace_id")
            except ApiError as e:
                answer, is_error = e.display(), True
            except requests.Timeout:
                answer, is_error = (
                    "The request timed out. Try again, or ask a narrower question.", True,
                )
            except requests.ConnectionError:
                answer, is_error = (
                    f"Couldn't reach the API at {BACKEND_URL}. Check that the backend is running.", True,
                )
            except requests.RequestException as e:
                answer, is_error = f"The request failed unexpectedly.\n\n`{e}`", True

        if is_error:
            st.error(answer)
        else:
            st.markdown(answer)
            render_extras(sources, steps)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources, "steps": steps,
         "trace_id": trace_id, "feedback_sent": False, "error": is_error}
    )
    st.rerun()  # redraw from history so the starter buttons disappear and feedback buttons appear