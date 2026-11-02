# evals/eval_generator.py
"""
Component eval: the GENERATOR, in isolation.

We feed the production generator prompt the GOLDEN context (relevant_contexts
from golden_dataset.json), NOT the retriever's output, so a low score is purely
the generator's fault.

  Faithfulness     -- are the answer's claims supported by the given context?
  AnswerRelevancy  -- does the answer address the question?

    python -m evals.eval_generator
"""
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

from evals.rag_adapter import generate
from evals.judge import get_judge
from evals.harness import load_rag_goldens, run_eval, summarize_by_metric, print_summary, pace

THRESHOLD = 0.7


def run(goldens=None):
    goldens = goldens or load_rag_goldens()

    test_cases = []
    for g in goldens:
        context = g["ideal_context"]
        answer = generate(g["query"], context)
        test_cases.append(LLMTestCase(
            input=g["query"],
            actual_output=answer,
            retrieval_context=context,
        ))
        pace()

    judge = get_judge()
    metrics = [
        FaithfulnessMetric(threshold=THRESHOLD, model=judge, include_reason=True),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=judge, include_reason=True),
    ]
    return summarize_by_metric(run_eval(test_cases, metrics))


if __name__ == "__main__":
    print_summary("generator", run())
