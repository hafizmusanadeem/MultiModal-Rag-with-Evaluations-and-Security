# evals/harness.py
"""
Shared harness for the eval suite.

  load_goldens(path)          -- read a golden JSON (relative paths resolve inside evals/)
  load_rag_goldens()          -- turn evals/golden_dataset.json rag_samples into the
                                 normalized golden shape every quality eval uses
  run_eval(cases, metrics)    -- DeepEval evaluate() with sequential judging (Groq-friendly)
  summarize_by_metric(result) -- per-metric pass rate + score stats (NOT pooled, so a
                                 regression in one metric can't hide behind the others)
  pace()                      -- sleep between live pipeline calls (Groq RPM)
"""
import json
import os
import time

EVALS_DIR = os.path.dirname(os.path.abspath(__file__))
GOLDEN_DATASET_PATH = os.path.join(EVALS_DIR, "golden_dataset.json")

# Seconds to sleep between live pipeline calls, OUTSIDE any timed region.
# Each pipeline call is ~3-4 Groq requests (guardrail + planner + generator), so
# this keeps you under the on-demand RPM limit. A 429 would trigger Portkey's
# retry/fallback to the 8b model and quietly change what you are measuring.
OPS_PACE_S = float(os.getenv("OPS_PACE_S", "6"))


def pace():
    if OPS_PACE_S > 0:
        time.sleep(OPS_PACE_S)


def _resolve(path):
    return path if os.path.isabs(path) else os.path.join(EVALS_DIR, path)


def load_goldens(path):
    with open(_resolve(path), encoding="utf-8") as f:
        return json.load(f)


def load_rag_goldens():
    """golden_dataset.json -> [{id, domain, query, ideal_answer, ideal_context}]"""
    with open(GOLDEN_DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return [
        {
            "id": s["id"],
            "domain": s.get("domain", ""),
            "query": s["question"],
            "ideal_answer": s["reference"],
            "ideal_context": s["relevant_contexts"],
        }
        for s in data["rag_samples"]
    ]


def run_eval(test_cases, metrics, **kwargs):
    """
    evaluate() with judging run SEQUENTIALLY by default. DeepEval's default
    async mode fires many judge calls at once, which blows through Groq's TPM/RPM
    limits. Set EVAL_ASYNC=1 to opt back in if your judge key has headroom.
    """
    from deepeval import evaluate

    if os.getenv("EVAL_ASYNC") != "1":
        try:
            try:
                from deepeval.evaluate.configs import AsyncConfig
            except ImportError:
                from deepeval.evaluate import AsyncConfig
            return evaluate(
                test_cases=test_cases, metrics=metrics,
                async_config=AsyncConfig(run_async=False), **kwargs,
            )
        except (ImportError, TypeError):
            pass   # older/newer DeepEval without async_config -> default behaviour
    return evaluate(test_cases=test_cases, metrics=metrics, **kwargs)


def summarize_by_metric(result):
    """DeepEval EvaluationResult -> {metric_name: {n, pass_rate, avg/min/max score}}."""
    test_results = getattr(result, "test_results", None)
    if test_results is None:
        test_results = result if isinstance(result, list) else []

    buckets = {}
    for tr in test_results:
        metrics = getattr(tr, "metrics_data", None) or getattr(tr, "metrics", None) or []
        for m in metrics:
            name = getattr(m, "name", "unknown")
            b = buckets.setdefault(name, {"scores": [], "passed": 0, "total": 0})
            b["total"] += 1
            score = getattr(m, "score", None)
            if score is not None:
                b["scores"].append(score)
            if getattr(m, "success", False):
                b["passed"] += 1

    summary = {}
    for name, b in buckets.items():
        scores = b["scores"]
        summary[name] = {
            "n": b["total"],
            "pass_rate": (100 * b["passed"] / b["total"]) if b["total"] else 0.0,
            "avg_score": (sum(scores) / len(scores)) if scores else float("nan"),
            "min_score": min(scores) if scores else float("nan"),
            "max_score": max(scores) if scores else float("nan"),
        }
    return summary


def print_summary(title, summary):
    print("\n" + "=" * 60)
    print(f"{title}  (per-metric summary)")
    print("=" * 60)
    for name, s in summary.items():
        avg = f"{s['avg_score']:.2f}" if s["avg_score"] == s["avg_score"] else "nan"
        print(f"  {name:<26} pass_rate={s['pass_rate']:5.0f}%  avg={avg}  n={s['n']}")
    print("=" * 60)
