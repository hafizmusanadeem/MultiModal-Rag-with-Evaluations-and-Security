# ============================================================
# CRITICAL: logfire MUST be configured before ALL other imports
# so that spans from all modules are captured from the start.
# ============================================================
import logfire
import os
import secrets
import uuid
from dotenv import load_dotenv

load_dotenv()
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))

# Now safe to import app modules - logfire is already active
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from langsmith import traceable
from app.agents.graph import rag_agent
from app.guardrails import initialize_rails, guard

from pydantic import BaseModel, Field
from typing import Literal, Optional


# Initialize FastAPI
app = FastAPI(title="Enterprise Agentic RAG API")

# Shared-secret auth. Set API_KEY in prod (Secret Manager). If unset, auth is OFF (local dev).
API_KEY = os.getenv("API_KEY")
OPEN_PATHS = {"/", "/health"}   # liveness only; everything else needs the key


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    if API_KEY and request.url.path not in OPEN_PATHS:
        provided = request.headers.get("x-api-key", "")
        if not secrets.compare_digest(provided.encode("utf-8"), API_KEY.encode("utf-8")):
            return JSONResponse(
                status_code=401,
                content={"error": {"message": "Invalid or missing API key."}},
            )
    return await call_next(request)


def _verify_vector_store() -> None:
    """Fail fast if the embedding dimension doesn't match the Qdrant collection.
    A mismatch otherwise surfaces as silently-empty retrieval."""
    from app.config import settings
    from app.services.retrieval.embedding import get_embedding_dim
    from app.services.retrieval.qdrant_service import client

    try:
        info = client.get_collection(settings.QDRANT_COLLECTION)
    except Exception as e:
        logfire.error(f"Could not verify Qdrant collection '{settings.QDRANT_COLLECTION}': {e}")
        return

    size = getattr(info.config.params.vectors, "size", None)
    dim = get_embedding_dim()
    if size is not None and size != dim:
        raise RuntimeError(
            f"Qdrant collection '{settings.QDRANT_COLLECTION}' is {size}-dim but the active "
            f"embedding model is {dim}-dim. Re-ingest with --wipe using the same model."
        )
    logfire.info(f"✅ Qdrant collection OK ({size}-dim matches embeddings).")


@app.on_event("startup")
def startup_event():
    if not API_KEY:
        logfire.warning("⚠️ API_KEY is not set — all endpoints are UNPROTECTED.")
    initialize_rails()

    # Warm heavy components now so (a) the first user request isn't slow and
    # (b) a bad deploy (Gemini unreachable, wrong collection) fails at startup, not at query time.
    from app.services.retrieval.ranking_service import warm_up
    warm_up()
    _verify_vector_store()


class QueryRequest(BaseModel):
    q: str = Field(min_length=1, max_length=4000)
    thread_id: Optional[str] = "default_user"


class FeedbackRequest(BaseModel):
    trace_id: str
    rating: Literal["up", "down"]
    comment: Optional[str] = Field(default=None, max_length=2000)


@app.get("/")
def home():
    return {"message": "Enterprise LangGraph RAG API is live."}


@app.get("/health")
def health():
    """Liveness probe used by the Streamlit sidebar status indicator."""
    return {"status": "ok"}


@app.get("/graph")
def get_graph_image():
    """
    Returns the Mermaid image of the agent's workflow.
    """
    try:
        png_bytes = rag_agent.get_graph().draw_mermaid_png()
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        return {"error": f"Could not generate graph image: {e}"}


@app.post("/feedback")
def feedback(request: FeedbackRequest):
    """
    Records thumbs up/down for a previous /query response.
    Stored as a Logfire event keyed by trace_id (no separate database).
    """
    logfire.info(
        "👍 User feedback" if request.rating == "up" else "👎 User feedback",
        trace_id=request.trace_id,
        rating=request.rating,
        comment=request.comment,
    )
    return {"status": "recorded"}


@traceable(name="rag_query", run_type="chain")
def _run_rag_pipeline(q: str, thread_id: str, trace_id: str) -> dict:
    """
    Core RAG path, wrapped so the whole request appears as one LangSmith run.
    """
    initial_state = {
        "messages": [{"role": "user", "content": q}],
        "current_query": q,
        "documents": [],
        "plan": ["Start"],
        "status": "Initializing Graph...",
    }
    config = {"configurable": {"thread_id": thread_id}}

    with logfire.span("📨 Query", trace_id=trace_id, thread_id=thread_id):
        # Gate 1: NeMo Guardrails
        rail_fired, rail_response = guard(q)
        if rail_fired:
            logfire.info(f"🛡️ Request blocked by guardrails | thread={thread_id}")
            return {
                "question": q,
                "answer": rail_response,
                "thought_process": ["Intent: Guardrails Fired", "Retrieval: Skipped"],
                "status": "Blocked by guardrails.",
                "sources": [],
                "trace_id": trace_id,
            }

        # Gate 2: LangGraph RAG pipeline
        final_output = rag_agent.invoke(initial_state, config=config)

    return {
        "question": q,
        "answer": final_output.get("final_answer"),
        "thought_process": final_output.get("plan"),
        "status": final_output.get("status"),
        "sources": final_output.get("documents", []),
        "trace_id": trace_id,
    }


@app.post("/query")
def query(request: QueryRequest):
    """
    Executes the LangGraph RAG flow with memory using a POST request.
    """
    q = request.q
    thread_id = request.thread_id or "default_user"
    trace_id = uuid.uuid4().hex

    try:
        return _run_rag_pipeline(q, thread_id, trace_id)
    except Exception as e:
        error_id = uuid.uuid4().hex[:8]
        logfire.error(f"❌ Backend Execution Failed [{error_id}]: {e}", trace_id=trace_id)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": "Something went wrong while processing your question. Please try again.",
                    "error_id": error_id,
                }
            },
        )