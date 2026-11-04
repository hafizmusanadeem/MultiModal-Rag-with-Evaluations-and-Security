"""
Streamlit frontend for the Enterprise RAG API.

Run (from project root, backend already running on :8000):
    streamlit run ui/app.py
"""
import os
import uuid
import requests
import streamlit as st


def _cfg(name: str, default: str = "") -> str:
    try:
        return st.secrets[name]          # Streamlit Cloud secrets
    except Exception:
        return os.getenv(name, default)  # local env / .env


BACKEND_URL = _cfg("BACKEND_URL", "http://localhost:8000").rstrip("/")
API_KEY = _cfg("API_KEY")
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}
REQUEST_TIMEOUT = 120

# backend_ok() - /health is open, header not needed, leave as is

def ask(question: str) -> dict:
    resp = requests.post(
        f"{BACKEND_URL}/query",
        json={"q": question, "thread_id": st.session_state.thread_id},
        headers=HEADERS, timeout=REQUEST_TIMEOUT,
    )
    ...

def submit_feedback(trace_id, rating, comment=None):
    response = requests.post(f"{BACKEND_URL}/feedback",
                             json={"trace_id": trace_id, "rating": rating, "comment": comment},
                             headers=HEADERS, timeout=10)
    ...

SUGGESTIONS = [
    "How does a Kubernetes Service route traffic to pods?",
    "What is the difference between a Deployment and a StatefulSet?",
    "Summarize the networking requirements from the docs",
]

st.set_page_config(page_title="Docs Assistant", page_icon="📘", layout="centered")

# ---------------------------------------------------------------- styling
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Schibsted+Grotesk:wght@400;500;700&family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap');

:root {
  --ink: #1b2830;
  --muted: #5b6b75;
  --paper: #f5f7f6;
  --panel: #ffffff;
  --line: #d9e0e3;
  --accent: #1d6a85;
  --accent-soft: #e2eff3;
  --amber: #b9791a;
  --amber-soft: #fbf0dc;
}
html, body, [class*="css"], .stApp { font-family: 'Schibsted Grotesk', system-ui, sans-serif; color: var(--ink); }
.stApp { background: var(--paper); }
header[data-testid="stHeader"] { background: transparent; }

h1.app-title { font-family: 'Newsreader', Georgia, serif; font-weight: 600; font-size: 2.1rem;
  letter-spacing: -0.01em; margin: 0 0 .15rem; }
.app-sub { color: var(--muted); margin-bottom: 1.5rem; max-width: 60ch; }

[data-testid="stChatMessage"] { background: var(--panel); border: 1px solid var(--line);
  border-radius: 10px; padding: .9rem 1rem; }
[data-testid="stChatMessage"] p { line-height: 1.6; max-width: 72ch; }

.chip { display: inline-block; font-size: .78rem; padding: .12rem .55rem; border-radius: 999px;
  background: var(--accent-soft); color: var(--accent); margin: 0 .35rem .35rem 0; font-weight: 500; }
.chip.cache { background: var(--amber-soft); color: var(--amber); }

.source { border-left: 3px solid var(--accent); background: var(--paper); padding: .6rem .8rem;
  margin-bottom: .6rem; border-radius: 0 6px 6px 0; font-size: .88rem; line-height: 1.5;
  white-space: pre-wrap; }
.source-label { font-size: .78rem; color: var(--muted); margin-bottom: .25rem; font-weight: 500; }

section[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--line); }
.status-dot { display:inline-block; width:.6rem; height:.6rem; border-radius:50%; margin-right:.45rem; }
.ok { background:#2e9e6b; } .down { background:#c0443a; }

.stButton > button { border-radius: 8px; border: 1px solid var(--line); background: var(--panel);
  color: var(--ink); font-weight: 500; }
.stButton > button:hover { border-color: var(--accent); color: var(--accent); }
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
        return requests.get(f"{BACKEND_URL}/health", timeout=3).ok
    except requests.RequestException:
        return False


def ask(question: str) -> dict:
    resp = requests.post(
        f"{BACKEND_URL}/query",
        json={"q": question, "thread_id": st.session_state.thread_id},
        timeout=REQUEST_TIMEOUT,
    )
    if not resp.ok:
        raise _api_error_from(resp)
    return resp.json()


def submit_feedback(trace_id: str, rating: str, comment: str | None = None):
    response = requests.post(f"{BACKEND_URL}/feedback",
                             json={"trace_id": trace_id, "rating": rating, "comment": comment},
                             timeout=10)
    if not response.ok:
        raise _api_error_from(response)
    return response.json()


def render_extras(sources: list[str], steps: list[str]) -> None:
    """Chips for the agent's steps + an expander with the retrieved chunks."""
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
    st.markdown("### Docs Assistant")
    online = backend_ok()
    st.markdown(
        f'<span class="status-dot {"ok" if online else "down"}"></span>'
        f'{"API connected" if online else "API unreachable"}',
        unsafe_allow_html=True,
    )
    if not online:
        st.caption(f"Start the backend: `uvicorn app.main:app --port 8000` ({BACKEND_URL})")

    st.divider()
    st.caption("Conversation")
    st.code(st.session_state.thread_id, language=None)
    if st.button("New conversation", use_container_width=True):
        st.session_state.thread_id = f"web-{uuid.uuid4().hex[:8]}"
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(
        "Answers are grounded in your ingested documents. Open **Sources** under "
        "a reply to see the passages it used."
    )

# ---------------------------------------------------------------- main
st.markdown('<h1 class="app-title">Ask your documentation</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-sub">Technical questions are searched against your knowledge base and '
    "reranked before answering. Follow-ups use the conversation so far.</p>",
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
    st.caption("Try one of these")
    for i, s in enumerate(SUGGESTIONS):
        if st.button(s, key=f"sugg{i}"):
            pending = s

typed = st.chat_input("Ask a question about your documents")
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
        with st.spinner("Searching and drafting an answer…"):
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
    st.rerun() if pending else None