# evals/eval_online.py
"""
Online eval worker: scores a sample of LIVE production traces from LangSmith with
the RAG triad and writes the scores back as feedback on each trace.

The FastAPI app runs rag_agent.invoke(), which LangSmith records as a root run
named "LangGraph". Its inputs hold the user's question (current_query) and its
outputs hold final_answer + documents (the post-rerank chunks, "CONTENT: "-prefixed).
Guardrail-blocked requests never reach LangGraph, so they have no trace to score.

    python -m evals.eval_online            # polling loop
    (production: delete the loop and call score_recent_traces() from cron)
"""
import os
import time
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from langsmith import Client

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualRelevancyMetric

from evals.judge import get_judge

load_dotenv()

PROJECT = os.getenv("LANGSMITH_PROJECT") or "rag_scale_test"
THRESHOLD = 0.7
SAMPLE_RATE = 0.3          # 3 metrics x several judge calls each per trace -- keep this low
POLL_SECONDS = 300
LOOKBACK_HOURS = 24
MAX_RUNS = 100

client = Client()


def _sampled(run):
    """Deterministic per-run sampling. (Python's built-in hash() of a str is salted
    per process, so it would re-sample differently on every restart.)"""
    return run.id.int % 100 < SAMPLE_RATE * 100


def _existing_keys(run):
    return {f.key for f in client.list_feedback(run_ids=[run.id])}


def _extract(run):
    inputs, outputs = run.inputs or {}, run.outputs or {}
    query = inputs.get("current_query") or ""
    answer = outputs.get("final_answer")
    context = [d.removeprefix("CONTENT: ") for d in (outputs.get("documents") or [])]
    return query, answer, context


def score_recent_traces():
    runs = client.list_runs(
        project_name=PROJECT,
        is_root=True,
        filter='eq(name, "LangGraph")',
        start_time=datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS),
        limit=MAX_RUNS,
    )
    judge = get_judge()

    for run in runs:
        if not _sampled(run):
            continue
        query, answer, context = _extract(run)
        if not (query and answer and context):
            continue                         # conversational / empty retrieval: nothing to judge

        already = _existing_keys(run)
        jobs = [
            ("faithfulness",
             FaithfulnessMetric(threshold=THRESHOLD, model=judge, include_reason=True),
             dict(input=query, actual_output=answer, retrieval_context=context)),
            ("answer_relevancy",
             AnswerRelevancyMetric(threshold=THRESHOLD, model=judge, include_reason=True),
             dict(input=query, actual_output=answer)),
            ("contextual_relevancy",
             ContextualRelevancyMetric(threshold=THRESHOLD, model=judge, include_reason=True),
             dict(input=query, actual_output=answer, retrieval_context=context)),
        ]
        for key, metric, tc in jobs:
            if key in already:
                continue
            try:
                metric.measure(LLMTestCase(**tc))
                client.create_feedback(run_id=run.id, key=key, score=metric.score, comment=metric.reason)
            except Exception as e:
                print(f"[{key}] failed on run {run.id}: {e}")


if __name__ == "__main__":
    while True:
        score_recent_traces()
        time.sleep(POLL_SECONDS)
