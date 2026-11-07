# evals/eval_application.py
"""
Application eval: end-to-end answer quality via three GEval metrics.

  Correctness  -- reference-based, judges TRUTH (not coverage or length)
  Completeness -- reference-based, judges COVERAGE of the reference's key points
  Style        -- reference-free, judges the JAFZA Rules Assistant voice

    python -m evals.eval_application
"""
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric

from evals.rag_adapter import RagPipeline
from evals.judge import get_judge
from evals.harness import load_rag_goldens, run_eval, summarize_by_metric, print_summary, pace

THRESHOLD = 0.7


def run(rag, goldens=None):
    goldens = goldens or load_rag_goldens()
    judge = get_judge()

    test_cases = []
    for g in goldens:
        result = rag.invoke(g["query"])
        pace()
        test_cases.append(LLMTestCase(
            input=g["query"],
            actual_output=result["answer"],
            expected_output=g["ideal_answer"],
        ))

    io = [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT]

    correctness = GEval(
        name="Correctness",
        evaluation_steps=[
            "Compare only the factual claims in the actual output against the expected output.",
            "A claim is wrong only if it CONTRADICTS the expected output or is factually false (wrong command, wrong flag, wrong field name, wrong behaviour). Judge truth, not completeness.",
            "A factually accurate answer must score at least 0.9 even if it is shorter or covers fewer points than the expected output.",
            "Do NOT deduct for brevity, missing elaboration, or omitted points --- omissions are not errors here.",
            "Additional correct information must NEVER lower the score.",
        ],
        rubric=[
            Rubric(score_range=(9, 10), expected_outcome="All stated claims are factually correct and consistent. No contradictions. Brevity is fine."),
            Rubric(score_range=(5, 8),  expected_outcome="Mostly correct but one minor inaccuracy."),
            Rubric(score_range=(0, 4),  expected_outcome="Contains a clear factual error or a claim that contradicts the expected output."),
        ],
        evaluation_params=io, threshold=THRESHOLD, model=judge, strict_mode=False,
    )

    completeness = GEval(
        name="Completeness",
        evaluation_steps=[
            "Identify the key points contained in the expected output (commands, field names, steps, conditions).",
            "Check how many of those key points are addressed in the actual output.",
            "Penalize the actual output for each key point from the expected output that it omits or only partially covers.",
            "Judge coverage only. Do NOT lower the score because a covered point is stated incorrectly --- factual correctness is judged separately.",
            "Do NOT penalize the actual output for adding extra information beyond the expected output.",
        ],
        rubric=[
            Rubric(score_range=(9, 10), expected_outcome="Addresses essentially all key points in the expected output."),
            Rubric(score_range=(5, 8),  expected_outcome="Covers the main key points but misses one or more."),
            Rubric(score_range=(0, 4),  expected_outcome="Misses several key points; only partially covers the expected output."),
        ],
        evaluation_params=io, threshold=THRESHOLD, model=judge, strict_mode=False,
    )

    style = GEval(
        name="Style",
        evaluation_steps=[
            "Judge only the style and tone of the actual output, not whether it is factually correct or complete.",
            "Reward the voice of a precise compliance advisor: direct, accurate and actionable, quoting exact figures and conditions and citing rule numbers where available, with a one-line reason for each non-obvious point.",
            "Reward a professional but readable register: plain language, jargon briefly unpacked when it matters, no filler or hedging.",
            "Penalize answers that ramble, bury the command or answer under preamble, are unexplained walls of jargon, or read as robotic boilerplate.",
            "Do NOT reward or penalize based on correctness, completeness, or length --- only on style and tone.",
        ],
        rubric=[
            Rubric(score_range=(9, 10), expected_outcome="Clearly an expert compliance advisor's voice: direct answer first, exact figures and rule references, brief rationale, no fluff."),
            Rubric(score_range=(7, 8),  expected_outcome="Clear, professional and actionable. Fully acceptable even if plain."),
            Rubric(score_range=(4, 6),  expected_outcome="Understandable but wordy, flat, or poorly structured in places."),
            Rubric(score_range=(0, 3),  expected_outcome="Rambling, evasive, jargon-heavy, or robotic; not useful as compliance guidance."),
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=THRESHOLD, model=judge, strict_mode=False,
    )

    return summarize_by_metric(run_eval(test_cases, [correctness, completeness, style]))


def run_local():
    return run(RagPipeline())


if __name__ == "__main__":
    print_summary("application", run_local())
