# evals/eval_rag_pipeline.py
"""
Pipeline eval: the RAG triad on LIVE output (guardrails -> planner -> retrieve
-> rerank -> generate).

  ContextualRelevancy -- is what the retriever returned relevant to the query?
  Faithfulness        -- is the answer grounded in that retrieved context?
  AnswerRelevancy     -- does the answer address the question?

Cases where no context came back (guardrail fired, or retrieval returned
nothing) are skipped and counted: Faithfulness/ContextualRelevancy cannot score
an empty retrieval_context, and an empty retrieval is a retriever problem that
eval_ops' reliability check already surfaces.

    python -m evals.eval_rag_pipeline
"""
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualRelevancyMetric

from evals.rag_adapter import RagPipeline
from evals.judge import get_judge
from evals.harness import load_rag_goldens, run_eval, summarize_by_metric, print_summary, pace

THRESHOLD = 0.7


def run(rag, goldens=None):
    goldens = goldens or load_rag_goldens()

    test_cases, skipped = [], 0
    for g in goldens:
        result = rag.invoke(g["query"])
        pace()
        if not result["context"] or not result["answer"]:
            skipped += 1
            continue
        test_cases.append(LLMTestCase(
            input=g["query"],
            actual_output=result["answer"],
            retrieval_context=result["context"],
        ))
    if skipped:
        print(f"[pipeline] skipped {skipped}/{len(goldens)} cases with empty context/answer")

    judge = get_judge()
    metrics = [
        ContextualRelevancyMetric(threshold=THRESHOLD, model=judge, include_reason=True),
        FaithfulnessMetric(threshold=THRESHOLD, model=judge, include_reason=True),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=judge, include_reason=True),
    ]
    return summarize_by_metric(run_eval(test_cases, metrics))


def run_local():
    return run(RagPipeline())


if __name__ == "__main__":
    print_summary("rag_pipeline", run_local())
