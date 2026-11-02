# ============================================================
# CRITICAL: logfire MUST be configured before ALL other imports
# so that spans from all modules are captured from the start.
# ============================================================
import logfire
import os
import uuid
from dotenv import load_dotenv

load_dotenv()
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))

# Now safe to import app modules - logfire is already active
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from app.agents.graph import rag_agent
from app.guardrails import initialize_rails, guard

from pydantic import BaseModel, Field
from typing import Literal, Optional


# Initialize FastAPI
app = FastAPI(title="Enterprise Agentic RAG API")


@app.on_event("startup")
def startup_event():
    initialize_rails()


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


@app.post("/query")
def query(request: QueryRequest):
    """
    Executes the LangGraph RAG flow with memory using a POST request.
    """
    q = request.q
    thread_id = request.thread_id
    trace_id = uuid.uuid4().hex

    initial_state = {
        "messages": [{"role": "user", "content": q}],
        "current_query": q,
        "documents": [],
        "plan": ["Start"],
        "status": "Initializing Graph..."
    }

    # Configuration for Memory (Thread ID)
    config = {"configurable": {"thread_id": thread_id}}

    try:
        with logfire.span("📨 Query", trace_id=trace_id, thread_id=thread_id):
            # Gate 1: NeMo Guardrails — blocks off-topic, jailbreaks, and handles dialog
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
            # Run the graph synchronously to preserve Logfire context variables
            final_output = rag_agent.invoke(initial_state, config=config)

        return {
            "question": q,
            "answer": final_output.get("final_answer"),
            "thought_process": final_output.get("plan"),
            "status": final_output.get("status"),
            "sources": final_output.get("documents", []),
            "trace_id": trace_id,
        }
    except Exception as e:
        # error_id lets the user quote a reference; the full exception stays in Logfire only.
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