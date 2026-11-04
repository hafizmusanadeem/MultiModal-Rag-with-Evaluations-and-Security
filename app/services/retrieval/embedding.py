import os
import re
import time
import logfire
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import settings

BATCH_SIZE = 50
_GEMINI_DIM = 3072
_FALLBACK_DIM = 768  # all-mpnet-base-v2

# Ingestion path (nobody is waiting): patient, long waits.
EMBED_MAX_RETRIES = int(os.getenv("EMBED_MAX_RETRIES", "10"))   # retries per batch
EMBED_MAX_WAIT_S = float(os.getenv("EMBED_MAX_WAIT_S", "90"))   # cap on a single wait
# Query path (a user is waiting): short budget.
QUERY_MAX_RETRIES = int(os.getenv("EMBED_QUERY_MAX_RETRIES", "3"))
QUERY_MAX_WAIT_S = float(os.getenv("EMBED_QUERY_MAX_WAIT_S", "10"))

_active_model = None
_model_type: str | None = None  # "gemini" or "fallback"


def _require_gemini() -> bool:
    """Production switch. When REQUIRE_GEMINI=1 the local 768-dim fallback is NEVER used:
    it would embed queries at the wrong dimension for a 3072-dim Qdrant collection, and
    search would silently return nothing."""
    return os.getenv("REQUIRE_GEMINI") == "1"


# ── Rate-limit helpers ─────────────────────────────────────────────────────────

_RETRY_DELAY_RE = re.compile(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s")


def _is_rate_limit(err: str) -> bool:
    return any(x in err for x in ("429", "rate limit", "rate_limit", "quota", "resource_exhausted"))


def _wait_seconds(err: str, attempt: int, max_wait: float = EMBED_MAX_WAIT_S, base: float = 5.0) -> float:
    """Honour Gemini's suggested retryDelay, else exponential backoff (base, 2*base, ...), capped."""
    m = _RETRY_DELAY_RE.search(err)
    suggested = float(m.group(1)) + 2 if m else 0.0
    backoff = base * (2 ** attempt)
    return min(max(suggested, backoff), max_wait)


def _call_with_retry(fn, label: str, max_retries: int, max_wait: float, base: float = 5.0):
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            raw = str(e)
            err = raw.lower()
            if not _is_rate_limit(err):
                logfire.error(f"Gemini {label} failed: {raw}")
                raise
            if "perday" in err or "per_day" in err:
                logfire.error("Gemini DAILY quota exhausted — waiting won't help. "
                              "Retry later (ingestion: re-run the same command to resume).")
                raise
            if attempt >= max_retries:
                logfire.error(f"Gemini rate limit persisted after {max_retries + 1} attempts ({label}).")
                raise
            wait = _wait_seconds(raw, attempt, max_wait, base)
            logfire.warning(
                f"Gemini rate limit ({label}) — sleeping {wait:.0f}s "
                f"(retry {attempt + 1}/{max_retries})."
            )
            time.sleep(wait)
    raise RuntimeError("unreachable")


# ── Model initialisation ───────────────────────────────────────────────────────

def _probe_gemini():
    """Try one embed call to verify Gemini is reachable. Returns model or None."""
    attempts = 3 if _require_gemini() else 1   # in prod, ride out a transient 429 at cold start
    last = None
    for i in range(attempts):
        try:
            model = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-2-preview",
                google_api_key=settings.GEMINI_API_KEY,
            )
            model.embed_query("probe")
            logfire.info("Gemini embeddings ready (gemini-embedding-2-preview, 3072-dim).")
            return model
        except Exception as e:
            last = e
            if i < attempts - 1:
                wait = _wait_seconds(str(e), i, max_wait=30)
                logfire.warning(f"Gemini probe failed ({e}); retrying in {wait:.0f}s.")
                time.sleep(wait)
    if _require_gemini():
        logfire.error(f"Gemini probe failed: {last}")
    else:
        logfire.warning(f"Gemini probe failed: {last}. Will use sentence-transformers fallback.")
    return None


def _load_fallback():
    from sentence_transformers import SentenceTransformer
    logfire.info("Loading sentence-transformers fallback (all-mpnet-base-v2, 768-dim).")
    return SentenceTransformer("all-mpnet-base-v2")


def _init():
    """Initialise embedding model once per process. Called lazily on first use."""
    global _active_model, _model_type
    if _active_model is not None:
        return

    if os.getenv("FORCE_LOCAL_EMBEDDINGS") == "1":
        if _require_gemini():
            raise RuntimeError("FORCE_LOCAL_EMBEDDINGS=1 conflicts with REQUIRE_GEMINI=1.")
        _active_model = _load_fallback()
        _model_type = "fallback"
        return

    gemini = _probe_gemini()
    if gemini:
        _active_model = gemini
        _model_type = "gemini"
    elif _require_gemini():
        # _active_model stays None, so the next call retries initialisation.
        raise RuntimeError(
            "REQUIRE_GEMINI=1 but Gemini embeddings are unreachable. Refusing to fall back to the "
            "768-dim local model (it would not match the 3072-dim Qdrant collection)."
        )
    else:
        _active_model = _load_fallback()
        _model_type = "fallback"


# ── Public helpers ─────────────────────────────────────────────────────────────

def get_embedding_dim() -> int:
    """Return the vector dimension for the active model. Call after _init()."""
    _init()
    return _GEMINI_DIM if _model_type == "gemini" else _FALLBACK_DIM


# ── Batch embedding ────────────────────────────────────────────────────────────

def _embed_batch(batch: list[str]) -> list[list[float]]:
    if _model_type != "gemini":
        return _active_model.encode(batch, show_progress_bar=False).tolist()
    return _call_with_retry(
        lambda: _active_model.embed_documents(batch),
        "batch", EMBED_MAX_RETRIES, EMBED_MAX_WAIT_S,
    )


# ── Public API (same signatures as before) ─────────────────────────────────────

def embed_query(query: str) -> list[float]:
    _init()
    if _model_type == "gemini":
        return _call_with_retry(
            lambda: _active_model.embed_query(query),
            "query", QUERY_MAX_RETRIES, QUERY_MAX_WAIT_S, base=1.0,
        )
    return _active_model.encode([query])[0].tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    _init()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        with logfire.span("Embed batch", model=_model_type, start=i, size=len(batch)):
            all_embeddings.extend(_embed_batch(batch))
    return all_embeddings
