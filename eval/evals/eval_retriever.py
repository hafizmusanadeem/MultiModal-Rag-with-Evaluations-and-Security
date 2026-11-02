# evals/eval_retriever.py
"""
Component eval: the RETRIEVER, in isolation (Qdrant top-15 -> FlashRank top-5).

  ContextualRecall    -- did we miss information needed for the ideal answer?
  ContextualPrecision -- are the relevant chunks ranked above irrelevant ones?
  Chunk Overlap       -- deterministic, NO judge: does a retrieved chunk actually
                         contain each hand-picked relevant_context from the golden
                         set? Zero judge noise, so it is the steadiest retrieval
                         signal to gate on.

Note: production retrieves with the PLANNER's rewritten search term; here we send
the raw golden question, which isolates the retriever from the planner.

    python -m evals.eval_retriever
"""
import re

from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric

from evals.rag_adapter import RerankingRetriever
from evals.judge import get_judge
from evals.harness import load_rag_goldens, run_eval, summarize_by_metric, print_summary, pace

THRESHOLD = 0.7
OVERLAP_HIT = 0.6      # a relevant context counts as "found" if >=60% of its tokens appear in one chunk


def _tokens(s):
    return set(re.findall(r"[a-z0-9_\-\.]+", s.lower()))


def chunk_overlap_score(retrieved, relevant):
    """Fraction of golden relevant_contexts found (token coverage >= OVERLAP_HIT) in any retrieved chunk."""
    chunk_toks = [_tokens(c) for c in retrieved]
    hits = 0
    for rel in relevant:
        rt = _tokens(rel)
        if rt and any(len(rt & ct) / len(rt) >= OVERLAP_HIT for ct in chunk_toks):
            hits += 1
    return hits / len(relevant) if relevant else 0.0


def run(retriever, label="reranker", goldens=None):
    goldens = goldens or load_rag_goldens()

    test_cases, overlaps = [], []
    for g in goldens:
        retrieved = [d.page_content for d in retriever.invoke(g["query"])]
        overlaps.append(chunk_overlap_score(retrieved, g["ideal_context"]))
        test_cases.append(LLMTestCase(
            input=g["query"],
            expected_output=g["ideal_answer"],
            retrieval_context=retrieved,
            actual_output="(generator not evaluated in this run)",
        ))
        pace()    # embedding API + Qdrant; light, but keeps Gemini quota calm

    judge = get_judge()
    metrics = [
        ContextualRecallMetric(threshold=THRESHOLD, model=judge, include_reason=True),
        ContextualPrecisionMetric(threshold=THRESHOLD, model=judge, include_reason=True),
    ]
    result = run_eval(test_cases, metrics, hyperparameters={
        "retriever": label, "judge_model": judge.get_model_name(),
        "golden_set": "golden_dataset.json",
    })

    summary = summarize_by_metric(result)
    n = len(overlaps)
    summary["Chunk Overlap"] = {
        "n": n,
        "pass_rate": 100 * sum(o >= 0.5 for o in overlaps) / n,
        "avg_score": sum(overlaps) / n,
        "min_score": min(overlaps),
        "max_score": max(overlaps),
    }
    return summary


def run_local():
    return run(RerankingRetriever())


if __name__ == "__main__":
    print_summary("retriever", run_local())
