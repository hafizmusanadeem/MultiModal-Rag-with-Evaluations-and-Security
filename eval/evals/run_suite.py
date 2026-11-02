# evals/run_suite.py
"""
run_suite.py -- the orchestrator.

Runs the FULL eval suite (quality + safety + operational) against ONE pipeline,
flattens every metric into a dotted id, stamps run metadata, and writes a
snapshot JSON that evals/compare.py diffs against a baseline.

The pipeline is built ONCE and injected into every eval, so a snapshot is
provably the measurement of a single pipeline.

    python -m evals.run_suite --baseline --label "initial baseline"
    python -m evals.run_suite --label "top_k 5 -> 8"      # -> evals/baselines/candidate.json
    python -m evals.compare                               # PASS / REVIEW / FAIL
"""
import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone

from evals.rag_adapter import RagPipeline, RerankingRetriever, prompt_fingerprint, pipeline_config

from evals import (
    eval_retriever,
    eval_generator,
    eval_rag_pipeline,
    eval_application,
    eval_safety,
    eval_ops,
)
from evals.judge import DEFAULT_JUDGE_MODEL
from evals.metric_registry import rule_for

BASELINE_PATH  = "evals/baselines/baseline.json"
CANDIDATE_PATH = "evals/baselines/candidate.json"


def _slug(name):
    return name.strip().lower().replace(" ", "_")


def flatten_nested(namespace, summary):
    """{metric: {stat: val}} -> {'namespace.metric_slug.stat': val}."""
    return {f"{namespace}.{_slug(m)}.{stat}": v for m, stats in summary.items() for stat, v in stats.items()}


def prefix_flat(namespace, flat):
    return {f"{namespace}.{k}": v for k, v in flat.items()}


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def build_metadata(label):
    try:
        config = pipeline_config()
    except Exception as e:
        config = {"error": str(e)}
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "prompt_hash": prompt_fingerprint(),
        "label": label,
        "judge_model": DEFAULT_JUDGE_MODEL,
        "pipeline": config,      # NB: embedding == "fallback" means Gemini was down and 768-dim mpnet was used
    }


def run_suite(label="", quiet=False, full=False):
    verbose = not quiet
    t_start = time.perf_counter()

    print("Building pipeline (once)...")
    rag = RagPipeline()
    retriever = getattr(rag, "retriever", None) or RerankingRetriever()

    metrics = {}

    print("\n[1/6] retriever eval...")
    metrics.update(flatten_nested("retriever", eval_retriever.run(retriever)))

    print("\n[2/6] generator eval...")
    metrics.update(flatten_nested("generator", eval_generator.run()))

    print("\n[3/6] pipeline (triad) eval...")
    metrics.update(flatten_nested("pipeline", eval_rag_pipeline.run(rag)))

    print("\n[4/6] application quality eval...")
    metrics.update(flatten_nested("application", eval_application.run(rag)))

    print("\n[5/6] safety evals...")
    metrics.update(prefix_flat("safety", eval_safety.run_safety(rag, verbose=verbose)))

    print("\n[6/6] operational evals...")
    metrics.update(prefix_flat("ops", eval_ops.run_ops(rag, verbose=verbose)))

    elapsed = time.perf_counter() - t_start

    if not full:   # keep only what drives a verdict (gates + guardrails)
        metrics = {mid: v for mid, v in metrics.items() if rule_for(mid)["kind"] != "info"}

    return {
        "metadata": {**build_metadata(label), "suite_seconds": round(elapsed, 1),
                     "full": full, "n_metrics": len(metrics)},
        "metrics": metrics,
    }


def write_snapshot(snapshot, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2, sort_keys=True)
    return path


def print_snapshot(snapshot):
    meta, metrics = snapshot["metadata"], snapshot["metrics"]
    print("\n" + "=" * 74 + "\nSNAPSHOT\n" + "=" * 74)
    print(f"label       : {meta.get('label') or '(none)'}")
    print(f"created_at  : {meta['created_at']}")
    print(f"git_sha     : {meta['git_sha']}    prompt_hash: {meta['prompt_hash']}")
    print(f"judge       : {meta['judge_model']}    pipeline: {meta['pipeline']}")
    print(f"suite_time  : {meta['suite_seconds']}s     metrics: {len(metrics)}")
    print("-" * 74)
    for key in sorted(metrics):
        v = metrics[key]
        print(f"  {key:<44} {v:.4f}" if isinstance(v, float) else f"  {key:<44} {v}")
    print("=" * 74)


def main():
    p = argparse.ArgumentParser(description="Run the full eval suite and write a snapshot.")
    p.add_argument("--baseline", action="store_true", help=f"write to {BASELINE_PATH}")
    p.add_argument("--out", default=None, help="custom output path")
    p.add_argument("--label", default="", help="describe the change this snapshot represents")
    p.add_argument("--quiet", action="store_true", help="suppress per-eval ops/safety chatter")
    p.add_argument("--full", action="store_true", help="also write info metrics")
    args = p.parse_args()

    out = args.out or (BASELINE_PATH if args.baseline else CANDIDATE_PATH)
    snapshot = run_suite(label=args.label, quiet=args.quiet, full=args.full)
    print_snapshot(snapshot)
    print(f"\nwrote snapshot -> {write_snapshot(snapshot, out)}")


if __name__ == "__main__":
    main()
