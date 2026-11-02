# evals/rag_adapter.py
"""
The ONE bridge between the eval suite and this project's app/ code.

The original eval files were written against a different stack
(src.rag_pipeline.RagPipeline, src.generator, src.reranker). Instead of editing
every eval, this module re-exposes the same interface on top of the real app:

  RagPipeline().invoke(q)           -> {"query","context","answer","blocked","plan"}
  RagPipeline().retriever.invoke(q) -> [Doc(page_content=...)]   (Qdrant top-15 -> FlashRank top-5)
  VectorRetriever / RerankingRetriever   base vs reranked retriever (A/B)
  generate(q, context)              -> answer   (production prompt, on given context)
  generate_with_usage(q, context)   -> (answer, token usage)
  generate_stream(q, context)       -> yields text pieces (for TTFT)
  RagPipeline().invoke_staged(q)    -> (answer, per-stage ms incl. TTFT)

It runs the app IN-PROCESS (guardrail gate -> LangGraph), mirroring app/main.py,
so evals measure real code, not an HTTP wrapper. Every call gets a fresh
thread_id so LangGraph's MemorySaver never leaks history between eval questions.

Must be imported BEFORE any app module: it flips DISABLE_PORTKEY_CACHE (see the
patch in app/gateway/client.py) so repeated questions are not served from cache.
"""
import os
import sys
import time
import uuid
import hashlib

os.environ.setdefault("DISABLE_PORTKEY_CACHE", "1")
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

import logfire
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"), service_name="evals",
                  send_to_logfire="if-token-present")

# ---- app imports (after logfire + env, same order as app/main.py) ----
from app.config import settings
from app.guardrails import initialize_rails, guard
from app.agents.graph import rag_agent
from app.agents.nodes.planner import planner_node
from app.agents.nodes.retriever import retrieve_node
from app.agents.nodes.responder import build_prompt
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_documents
from app.gateway import portkey_client

FETCH_K = 15   # mirrors app/agents/nodes/retriever.py
TOP_K = 5


# ============================================================
# Retrievers  (page_content interface the evals expect)
# ============================================================
class Doc:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


class VectorRetriever:
    """Qdrant cosine search only -- no reranker. The 'before' in the A/B."""
    def __init__(self, k=TOP_K):
        self.k = k

    def invoke(self, query):
        hits = search_enterprise_knowledge(query, limit=self.k)
        return [Doc(h["content"], {"source": h["source"], "score": h["score"]}) for h in hits]


class RerankingRetriever:
    """Qdrant top-FETCH_K -> FlashRank -> top-TOP_K. Same as the production retriever node."""
    def __init__(self, fetch_k=FETCH_K, top_k=TOP_K):
        self.fetch_k, self.top_k = fetch_k, top_k

    def invoke(self, query):
        hits = search_enterprise_knowledge(query, limit=self.fetch_k)
        by_text = {h["content"]: h for h in hits}
        top = rerank_documents(query, [h["content"] for h in hits], top_n=self.top_k)
        return [Doc(t, {"source": by_text.get(t, {}).get("source")}) for t in top]


# ============================================================
# Generator helpers  (use the REAL production prompt via build_prompt)
# ============================================================
def _state_for(query, context):
    return {
        "messages": [{"role": "user", "content": query}],
        "current_query": query,
        "documents": [f"CONTENT: {c}" for c in context],
        "plan": ["Start"],
        "status": "",
    }


def _complete(prompt):
    return portkey_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}], temperature=0.1)


def _stream(prompt):
    stream = portkey_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}], temperature=0.1, stream=True)
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def generate(query, context):
    """Production generator prompt on a GIVEN context (list[str]) -> answer text."""
    return _complete(build_prompt(_state_for(query, context))).choices[0].message.content


def generate_with_usage(query, context):
    resp = _complete(build_prompt(_state_for(query, context)))
    u = getattr(resp, "usage", None)
    details = getattr(u, "prompt_tokens_details", None)
    usage = {
        "input": getattr(u, "prompt_tokens", 0) or 0,
        "output": getattr(u, "completion_tokens", 0) or 0,
        "cached": getattr(details, "cached_tokens", 0) or 0,
    }
    return resp.choices[0].message.content, usage


def generate_stream(query, context):
    yield from _stream(build_prompt(_state_for(query, context)))


def prompt_fingerprint():
    """Hash of the generator prompt TEMPLATE, so a silent prompt edit changes the hash."""
    tmpl = build_prompt(_state_for("<<Q>>", ["<<C>>"]))
    return hashlib.sha256(tmpl.encode()).hexdigest()[:12]


def pipeline_config():
    """What pipeline a snapshot measured. Stamped into run metadata."""
    from app.services.retrieval import embedding
    embedding._init()
    return {
        "generator_model": settings.GROQ_MODEL,
        "embedding": embedding._model_type,          # "gemini" or "fallback" (silent switch!)
        "embedding_dim": embedding.get_embedding_dim(),
        "fetch_k": FETCH_K,
        "top_k": TOP_K,
        "collection": settings.QDRANT_COLLECTION,
    }


# ============================================================
# The pipeline  (guardrails gate -> LangGraph, as in app/main.py)
# ============================================================
def _initial_state(q):
    return {"messages": [{"role": "user", "content": q}], "current_query": q,
            "documents": [], "plan": ["Start"], "status": "Initializing Graph..."}


class RagPipeline:
    def __init__(self):
        initialize_rails()
        self.retriever = RerankingRetriever()

    def invoke(self, question, thread_id=None, callbacks=None):
        fired, rail_response = guard(question)
        if fired:
            return {"query": question, "context": [], "answer": rail_response,
                    "blocked": True, "plan": ["Intent: Guardrails Fired"]}

        cfg = {"configurable": {"thread_id": thread_id or f"eval-{uuid.uuid4()}"}}
        if callbacks:
            cfg["callbacks"] = callbacks
        out = rag_agent.invoke(_initial_state(question), config=cfg)
        return {
            "query": question,
            "context": [d.removeprefix("CONTENT: ") for d in (out.get("documents") or [])],
            "answer": out.get("final_answer") or "",
            "blocked": False,
            "plan": out.get("plan") or [],
        }

    def invoke_with_usage(self, question):
        """invoke() + planner token usage captured from the LangChain call (for cost)."""
        try:
            from langchain_core.callbacks import UsageMetadataCallbackHandler
            cb = UsageMetadataCallbackHandler()
        except Exception:
            cb = None
        out = self.invoke(question, callbacks=[cb] if cb else None)
        usage = getattr(cb, "usage_metadata", None) or {}
        planner = {
            "input": sum(v.get("input_tokens", 0) for v in usage.values()),
            "output": sum(v.get("output_tokens", 0) for v in usage.values()),
        }
        return out, planner

    def invoke_staged(self, question):
        """
        Same flow as invoke(), run node-by-node with the answer STREAMED so each
        stage and time-to-first-token can be clocked. Returns (answer, ms-dict).
          ttft = query submitted -> first visible token
                 (guardrail + planner + retrieval + generation prefill, i.e. what
                 a streaming UI user actually waits through)
        """
        t0 = time.perf_counter()
        fired, rail = guard(question)
        t1 = time.perf_counter()
        if fired:
            return rail, {"guardrails": (t1 - t0) * 1000, "planner": 0.0, "retrieval": 0.0,
                          "generation": 0.0, "ttft": float("nan")}

        state = _initial_state(question)
        state.update(planner_node(state))
        t2 = time.perf_counter()
        if state["current_query"] != "CONVERSATIONAL":
            state.update(retrieve_node(state))
        t3 = time.perf_counter()

        first, pieces = None, []
        for piece in _stream(build_prompt(state)):
            if first is None:
                first = time.perf_counter()
            pieces.append(piece)
        t4 = time.perf_counter()

        return "".join(pieces), {
            "guardrails": (t1 - t0) * 1000,
            "planner": (t2 - t1) * 1000,
            "retrieval": (t3 - t2) * 1000,
            "generation": (t4 - t3) * 1000,
            "ttft": (first - t0) * 1000 if first else float("nan"),
        }
