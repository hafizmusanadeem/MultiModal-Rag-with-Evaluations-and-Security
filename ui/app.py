"""
Streamlit frontend for the JAFZA Rules Assistant.

Run locally (backend already running on :8000):
    streamlit run ui/app.py
On Streamlit Cloud it is launched through streamlit_app.py.

Layout: a landing hero with a single input and four starter questions; once the first
question is sent it becomes a clean chat with collapsible sources.
"""
import html
import os
import re
import time
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
AVATARS = {"user": ":material/person:", "assistant": ":material/auto_awesome:"}

# Starter questions: the things someone buying or selling a free-zone business asks first.
SUGGESTIONS = [
    "What does it cost to take over a plot from another Customer?",
    "What must be in place before a Licence can be renewed?",
    "What happens to Employment Agreements if a Customer's legal status changes?",
    "Can a Customer sub-lease its premises, and what does it cost?",
]

try:
    st.set_page_config(
        page_title=APP_NAME, page_icon="✦", layout="wide", initial_sidebar_state="collapsed"
    )
except StreamlitAPIException:
    pass  # already set by streamlit_app.py

# ---------------------------------------------------------------- styling
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Sora:wght@500;600;700&display=swap');

:root {
  color-scheme: dark;
  --bg: #07070c;
  --ink: #ececf3;
  --muted: #9a9ab0;
  --line: rgba(255,255,255,.09);
  --glass: rgba(255,255,255,.045);
  --violet: #7c5cff;
  --blue: #4da3ff;
  --pink: #ff6fd8;
  --grad: linear-gradient(120deg, #9b87ff 0%, #5db0ff 50%, #ff7fdc 100%);
}

html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
  background: var(--bg) !important;
  font-family: 'Inter', system-ui, sans-serif;
  color: var(--ink);
}
.stApp {
  background:
    radial-gradient(60% 38% at 50% -8%, rgba(124,92,255,.20), transparent 70%),
    var(--bg) !important;
}

/* Strip Streamlit chrome */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; height: 0 !important; pointer-events: none !important; }
header[data-testid="stHeader"] * { pointer-events: none !important; }  /* the header is 60px tall and would otherwise block "New chat" */
.block-container { max-width: 1020px; padding: 1.1rem 1.25rem 7rem !important; }

/* ---- top bar ---- */
.brand { display: flex; align-items: center; gap: .65rem; }
.brand .mark { width: 30px; height: 30px; border-radius: 9px; display: grid; place-items: center;
  background: var(--grad); color: #0a0a14; font-weight: 700; font-size: .95rem; }
.brand .name { font-family: 'Sora', sans-serif; font-weight: 600; font-size: 1.02rem; letter-spacing: -.01em; }
.brand .pill { font-size: .66rem; letter-spacing: .08em; padding: .12rem .5rem; border-radius: 999px;
  border: 1px solid var(--line); color: var(--muted); }
.topbar { display: flex; align-items: center; justify-content: space-between; gap: 1rem; margin-bottom: 1rem; }
.status { display: flex; align-items: center; gap: .45rem; font-size: .8rem; color: var(--muted); white-space: nowrap; }
/* chat view: brand + status on the left, "New chat" on the right, in normal flow (never overlaps content) */
.st-key-topbar { margin-bottom: 1rem; }
.st-key-topbar .topbar { margin-bottom: 0; }
.st-key-topbar [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; gap: .6rem; align-items: center; }
.st-key-topbar [data-testid="stColumn"] { min-width: 0 !important; }
.st-key-newchat { display: flex; justify-content: flex-end; }
.st-key-newchat [data-testid="stButton"] { width: auto; }
.status .dot { width: .5rem; height: .5rem; border-radius: 50%; display: inline-block; }
.status .ok { background: #35d39a; box-shadow: 0 0 10px #35d39a; }
.status .down { background: #ff5d6c; box-shadow: 0 0 10px #ff5d6c; }

/* ---- hero ---- */
.st-key-hero { position: relative; overflow: hidden; isolation: isolate; border-radius: 28px;
  padding: 3.6rem 2.8rem 2.4rem; margin-top: .9rem; border: 1px solid var(--line); background: #0a0a12; }
.st-key-hero::before { content: ""; position: absolute; inset: -30%; z-index: -2; filter: blur(26px);
  background:
    radial-gradient(42% 46% at 18% 28%, rgba(132,96,255,1), transparent 70%),
    radial-gradient(40% 44% at 82% 22%, rgba(58,150,255,.95), transparent 70%),
    radial-gradient(44% 48% at 68% 86%, rgba(255,86,200,.92), transparent 70%),
    radial-gradient(36% 42% at 8% 92%, rgba(60,220,255,.55), transparent 70%),
    radial-gradient(34% 38% at 92% 58%, rgba(138,112,255,.85), transparent 70%),
    conic-gradient(from 210deg at 62% 58%, transparent 0deg, rgba(255,140,226,.45) 70deg, rgba(110,170,255,.45) 140deg, transparent 220deg);
  animation: aurora 24s ease-in-out infinite alternate; }
.st-key-hero::after { content: ""; position: absolute; inset: 0; z-index: -1; pointer-events: none;
  background:
    radial-gradient(rgba(255,255,255,.10) 1px, transparent 1.3px) 0 0 / 22px 22px,
    linear-gradient(180deg, rgba(6,6,12,.10), rgba(6,6,12,.34)); }
@keyframes aurora {
  from { transform: translate3d(-3%, -2%, 0) rotate(0deg) scale(1); }
  to   { transform: translate3d(3%, 2%, 0) rotate(14deg) scale(1.08); }
}
.eyebrow { display: inline-block; font-size: .74rem; letter-spacing: .09em; text-transform: uppercase;
  padding: .3rem .8rem; border-radius: 999px; border: 1px solid rgba(255,255,255,.2);
  background: rgba(255,255,255,.07); backdrop-filter: blur(8px); margin-bottom: 1.1rem; }
.hero-title { font-family: 'Sora', sans-serif; font-weight: 600; font-size: clamp(2.1rem, 5.2vw, 3.7rem);
  line-height: 1.07; letter-spacing: -.03em; margin: 0 0 1rem; }
.grad { background: var(--grad); -webkit-background-clip: text; background-clip: text; color: transparent; }
.hero-sub { max-width: 52ch; font-size: 1.04rem; line-height: 1.6; color: rgba(255,255,255,.88); margin: 0 0 1.7rem; text-shadow: 0 1px 12px rgba(0,0,0,.35); }

/* ---- chat input ---- */
[data-testid="stChatInput"] { background: rgba(10,10,20,.78) !important; backdrop-filter: blur(16px);
  border: 1px solid rgba(255,255,255,.16) !important; border-radius: 20px !important;
  box-shadow: 0 10px 40px rgba(0,0,0,.35); }
[data-testid="stChatInput"] > div, [data-testid="stChatInput"] textarea {
  background: transparent !important; border: none !important; box-shadow: none !important; }
[data-testid="stChatInput"] textarea { color: var(--ink) !important; font-size: 1rem; }
[data-testid="stChatInput"] textarea::placeholder { color: #8d8da6 !important; }
[data-testid="stChatInput"] button { background: var(--grad) !important; color: #0a0a14 !important; border: none !important;
  border-radius: 12px !important; }
.st-key-hero [data-testid="stChatInput"] { min-height: 3.6rem; }
[data-testid="stBottom"], [data-testid="stBottom"] > div { background: transparent !important; }
[data-testid="stBottom"]::before { content: ""; position: absolute; inset: -30px 0 0; z-index: -1;
  background: linear-gradient(0deg, var(--bg) 55%, transparent); }

/* ---- starter chips ---- */
.st-key-chips { margin-top: 1rem; }
.st-key-chips [data-testid="stButton"] { width: 100%; }
.st-key-chips .stButton > button { width: 100%; height: auto; min-height: 3.9rem; padding: .7rem 1rem;
  text-align: left; justify-content: flex-start; white-space: normal; line-height: 1.35; font-size: .88rem;
  color: var(--ink); background: rgba(8,8,20,.55); border: 1px solid rgba(255,255,255,.2);
  border-radius: 14px; backdrop-filter: blur(12px); transition: all .18s ease; }
.st-key-chips .stButton > button:hover { background: rgba(255,255,255,.12); border-color: rgba(255,255,255,.4);
  transform: translateY(-1px); }
.st-key-chips .stButton > button > div { width: 100%; justify-content: flex-start; }
.st-key-chips .stButton > button p { color: inherit; text-align: left; white-space: normal; overflow: visible; text-overflow: clip; }

/* ---- value section ---- */
.section-eyebrow { margin: 3.2rem 0 .5rem; font-size: .74rem; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); }
.section-title { font-family: 'Sora', sans-serif; font-weight: 600; font-size: clamp(1.5rem, 3.4vw, 2.1rem);
  letter-spacing: -.02em; line-height: 1.15; margin: 0 0 1.4rem; }
.vgrid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
.vcard { padding: 1.35rem 1.3rem 1.4rem; border-radius: 18px; background: var(--glass); border: 1px solid var(--line); }
.vcard .num { font-family: 'Sora', sans-serif; font-weight: 600; font-size: .8rem; margin-bottom: .8rem; }
.vcard h4 { margin: 0 0 .45rem; font-size: 1.02rem; font-weight: 600; }
.vcard p { margin: 0; color: var(--muted); font-size: .9rem; line-height: 1.55; }
.footnote { margin-top: 2.4rem; text-align: center; font-size: .78rem; color: #6f6f86; line-height: 1.6; }
@media (max-width: 760px) {
  .vgrid { grid-template-columns: 1fr; }
  .st-key-hero { padding: 2.2rem 1.25rem 1.5rem; border-radius: 22px; }
}
@media (max-width: 600px) {
  .brand .pill, .st-key-topbar .status { display: none; }  /* make room for "New chat" */
}

/* ---- chat ---- */
[data-testid="stChatMessage"] { background: transparent; border: none; padding: .35rem 0; }
[data-testid="stChatMessage"]:has([aria-label*="from user"]) {
  background: var(--glass); border: 1px solid var(--line); border-radius: 16px; padding: .85rem 1.1rem; }
[data-testid="stChatMessage"]:has([aria-label*="from assistant"]) {
  background: rgba(124,92,255,.07); border: 1px solid rgba(124,92,255,.22); border-radius: 16px; padding: .95rem 1.15rem; }
[data-testid="stChatMessage"]:has([aria-label*="from assistant"]) [data-testid="stChatMessageAvatarCustom"] {
  background: var(--grad) !important; color: #0a0a14 !important; }
[data-testid="stChatMessage"]:has([aria-label*="from assistant"]) [data-testid="stChatMessageAvatarCustom"] * { color: #0a0a14 !important; }
[data-testid="stChatMessage"]:has([aria-label*="from user"]) [data-testid="stChatMessageAvatarCustom"] { background: rgba(255,255,255,.12) !important; }
[data-testid="stBottomBlockContainer"] { max-width: 1020px !important; margin: 0 auto; padding: 0 1.25rem 1.2rem !important; }
[data-testid="stChatMessage"] p { line-height: 1.65; }

.thinking { display: flex; align-items: center; gap: .6rem; color: var(--muted); font-size: .92rem; }
.thinking span { width: .45rem; height: .45rem; border-radius: 50%; background: var(--grad);
  animation: pulse 1.1s ease-in-out infinite; }
.thinking span:nth-child(2) { animation-delay: .15s; } .thinking span:nth-child(3) { animation-delay: .3s; }
.thinking em { font-style: normal; margin-left: .2rem; }
@keyframes pulse { 0%, 80%, 100% { opacity: .25; transform: scale(.75); } 40% { opacity: 1; transform: scale(1.1); } }

[data-testid="stExpander"] { background: rgba(255,255,255,.03) !important; border: 1px solid var(--line) !important;
  border-radius: 12px !important; margin-top: .6rem; }
[data-testid="stExpander"] summary { font-size: .86rem; color: var(--muted); }
.trail { font-size: .78rem; color: #8585a0; margin-bottom: .8rem; }
.source-label { font-size: .72rem; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); margin-bottom: .25rem; }
.source { border-left: 2px solid var(--violet); background: rgba(255,255,255,.04); padding: .6rem .85rem;
  margin-bottom: .7rem; border-radius: 0 8px 8px 0; font-size: .85rem; line-height: 1.55; white-space: pre-wrap;
  max-height: 210px; overflow-y: auto; color: #d6d6e4; }

/* small controls */
.st-key-newchat .stButton > button, div[class*="st-key-fb_"] button {
  border-radius: 999px; background: var(--glass); border: 1px solid var(--line); color: var(--ink); font-size: .82rem;
  min-height: 2rem; padding: .15rem .85rem; }
.st-key-newchat .stButton > button:hover, div[class*="st-key-fb_"] button:hover { border-color: rgba(255,255,255,.35); }
.thanks { font-size: .8rem; color: var(--muted); margin-top: .4rem; }
/* keep the 👍 👎 pair on one line on narrow screens (Streamlit stacks columns by default) */
div[class*="st-key-fbrow_"] [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; gap: .4rem; }
div[class*="st-key-fbrow_"] [data-testid="stColumn"] { min-width: 0 !important; width: auto !important; flex: 0 0 auto !important; }

@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"web-{uuid.uuid4().hex[:8]}"
if "messages" not in st.session_state:
    st.session_state.messages = []  # {role, content, sources, steps, error, trace_id, feedback}
if "pending" not in st.session_state:
    st.session_state.pending = None  # question waiting for an answer


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


@st.cache_data(ttl=15, show_spinner=False)
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


# ---------------------------------------------------------------- callbacks
def _queue(question: str) -> None:
    """Record the question and mark it pending; the answer is produced later in this run."""
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.pending = question


def _on_submit(key: str) -> None:
    question = (st.session_state.get(key) or "").strip()
    if question:
        _queue(question)


def _new_chat() -> None:
    st.session_state.thread_id = f"web-{uuid.uuid4().hex[:8]}"
    st.session_state.messages = []
    st.session_state.pending = None


def _send_feedback(index: int, rating: str) -> None:
    msg = st.session_state.messages[index]
    try:
        submit_feedback(msg["trace_id"], rating)
        msg["feedback"] = "sent"
    except (ApiError, requests.RequestException):
        msg["feedback"] = "failed"


# ---------------------------------------------------------------- rendering helpers
def _word_stream(text: str, delay: float = 0.009):
    for token in re.split(r"(\s+)", text):
        yield token
        if token.strip():
            time.sleep(delay)


def render_sources(sources: list[str], steps: list[str]) -> None:
    """One collapsed expander: the agent's steps as a muted trail, then the retrieved passages."""
    if not sources:
        return
    with st.expander(f"Sources · {len(sources)} passages"):
        if steps:
            trail = " → ".join(html.escape(s) for s in steps)
            st.markdown(f'<div class="trail">{trail}</div>', unsafe_allow_html=True)
        for i, src in enumerate(sources, 1):
            text = html.escape(src.removeprefix("CONTENT: ").strip())
            st.markdown(
                f'<div class="source-label">Passage {i}</div><div class="source">{text}</div>',
                unsafe_allow_html=True,
            )


def render_feedback(index: int, msg: dict) -> None:
    if not msg.get("trace_id"):
        return
    state = msg.get("feedback")
    if state == "sent":
        st.markdown('<div class="thanks">Thanks for the feedback.</div>', unsafe_allow_html=True)
        return
    with st.container(key=f"fbrow_{index}"):
        up, down, _ = st.columns([1, 1, 16])
        up.button("👍", key=f"fb_up_{index}", on_click=_send_feedback, args=(index, "up"), help="Helpful")
        down.button("👎", key=f"fb_down_{index}", on_click=_send_feedback, args=(index, "down"), help="Not helpful")
    if state == "failed":
        st.markdown('<div class="thanks">Feedback could not be saved right now.</div>', unsafe_allow_html=True)


def render_topbar(online: bool) -> None:
    bar = (
        f'<div class="topbar">'
        f'<div class="brand"><span class="mark">✦</span><span class="name">{APP_NAME}</span>'
        f'<span class="pill">AI</span></div>'
        f'<div class="status"><i class="dot {"ok" if online else "down"}"></i>'
        f'{"Live" if online else "Offline"}</div></div>'
    )
    if not st.session_state.messages:
        st.markdown(bar, unsafe_allow_html=True)
        return
    with st.container(key="topbar"):
        left, right = st.columns([5, 2], vertical_alignment="center")
        left.markdown(bar, unsafe_allow_html=True)
        right.button("New chat", key="newchat", on_click=_new_chat)


def render_landing() -> None:
    with st.container(key="hero"):
        st.markdown(
            '<div class="eyebrow">Jebel Ali Free Zone · Rules, 9th Edition</div>'
            '<div class="hero-title">Free zone rules,<br><span class="grad">answered in seconds.</span></div>'
            '<p class="hero-sub">Ask in plain English. Every answer is grounded in the official '
            'Jebel Ali Free Zone Rules and shows the exact passages behind it.</p>',
            unsafe_allow_html=True,
        )
        st.chat_input(
            "Ask about the JAFZA Rules…",
            key="chat_landing", on_submit=_on_submit, args=("chat_landing",),
        )
        with st.container(key="chips"):
            cols = st.columns(2)
            for i, suggestion in enumerate(SUGGESTIONS):
                cols[i % 2].button(suggestion, key=f"sugg{i}", on_click=_queue, args=(suggestion,), width="stretch")

    st.markdown(
        '<div class="section-eyebrow">Why this matters for BXB</div>'
        '<div class="section-title">Fewer repeat questions.<br>Better-informed buyers.</div>'
        '<div class="vgrid">'
        '<div class="vcard"><div class="num grad">01</div><h4>Faster due diligence</h4>'
        '<p>Buyers can check takeover, licence, lease and fee rules for a free-zone business '
        'before they make an offer, instead of waiting on an email reply.</p></div>'
        '<div class="vcard"><div class="num grad">02</div><h4>Less time on repeat enquiries</h4>'
        '<p>The questions every listing attracts get an instant, cited answer, so your team '
        'can focus on serious buyers and sellers.</p></div>'
        '<div class="vcard"><div class="num grad">03</div><h4>Answers you can verify</h4>'
        '<p>Each reply shows its source passages. When the Rules do not cover something, '
        'it says so rather than guessing.</p></div>'
        '</div>'
        '<div class="footnote">Demo on the JAFZA Rules. The same pipeline can ingest other free-zone '
        'rulebooks or your own listing documents.<br>Information only, not legal advice. '
        'Confirm important decisions with Jafza.</div>',
        unsafe_allow_html=True,
    )


def render_history() -> None:
    for index, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
            if msg.get("error"):
                st.error(msg["content"])
                continue
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_sources(msg.get("sources", []), msg.get("steps", []))
                render_feedback(index, msg)


def answer_pending() -> None:
    question = st.session_state.pending
    if not question:
        return

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        slot = st.empty()
        slot.markdown(
            '<div class="thinking"><span></span><span></span><span></span>'
            '<em>Searching the JAFZA Rules…</em></div>',
            unsafe_allow_html=True,
        )
        is_error, sources, steps, trace_id = False, [], [], None
        try:
            data = ask(question)
            answer = data.get("answer") or "No answer was returned."
            sources = data.get("sources", [])
            steps = data.get("thought_process", [])
            trace_id = data.get("trace_id")
        except ApiError as e:
            answer, is_error = e.display(), True
        except requests.Timeout:
            answer, is_error = "The request timed out. Try again, or ask a narrower question.", True
        except requests.ConnectionError:
            answer, is_error = f"Couldn't reach the API at {BACKEND_URL}. Check that the backend is running.", True
        except requests.RequestException as e:
            answer, is_error = f"The request failed unexpectedly.\n\n`{e}`", True

        slot.empty()
        if is_error:
            st.error(answer)
        else:
            st.write_stream(_word_stream(answer))

    st.session_state.pending = None
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources, "steps": steps,
         "trace_id": trace_id, "error": is_error}
    )
    st.rerun()  # redraw from history so sources and feedback buttons appear


# ---------------------------------------------------------------- page
render_topbar(backend_ok())

if st.session_state.messages:
    render_history()
    st.chat_input(
        "Ask a follow-up…", key="chat_main", on_submit=_on_submit, args=("chat_main",),
    )
    answer_pending()
else:
    render_landing()
