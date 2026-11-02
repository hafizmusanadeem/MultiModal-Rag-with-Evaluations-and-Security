# evals/eval_ops.py
"""
Operational evals, merged: LATENCY, COST, RELIABILITY.

Software + telemetry measurements -- no golden dataset, no judge. Each run_*()
prints a report AND returns a flat metrics dict; run_ops() merges all three into
the operational slice of the regression snapshot.

Project-specific notes:
  * One user query = NeMo guardrail LLM call + planner LLM call + embed/Qdrant/
    FlashRank + generator LLM call. Latency is therefore decomposed into
    guardrails / planner / retrieval / generation, and TTFT includes all of them.
  * Portkey's cache is DISABLED for these runs (DISABLE_PORTKEY_CACHE, set in
    rag_adapter) -- otherwise repeated questions come back from cache and the
    numbers are meaningless.
  * Sleeps between samples (OPS_PACE_S) happen OUTSIDE the timed regions.
  * COST covers planner + generator tokens. NeMo's guardrail call is NOT
    instrumented (it goes through its own ChatGroq on llama-3.1-8b-instant), so
    treat the figure as a slight underestimate.
"""
import math
import time

from evals.rag_adapter import RagPipeline, generate_with_usage
from evals.harness import pace

# Representative questions across the corpus (K8s autoscaling, K8s Jobs, Databricks).
QUESTIONS = [
    "How do you start Redis for a Kubernetes work queue?",
    "What is the difference between HPA and VPA in Kubernetes?",
    "What Databricks CLI command retrieves the details of a specific job?",
    "What are the valid restart policy values for a Kubernetes Job pod?",
]


def percentile(values, p):
    values = [v for v in values if not math.isnan(v)]
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    return s[int(k)] if lo == hi else s[lo] * (hi - k) + s[hi] * (k - lo)


def col_avg(rows, key):
    return sum(r[key] for r in rows) / len(rows) if rows else 0.0


# ============================== LATENCY ==============================
LAT_REPEATS = 5
LAT_WARMUP_RUNS = 2       # also loads FlashRank's ONNX model + probes Gemini embeddings
# Starting budgets for a 3-LLM-call pipeline on Groq. Calibrate after your first baseline.
SLO_P95_MS = 6000
SLO_TTFT_P95_MS = 3500


def lat_benchmark(pipeline):
    print(f"[latency] warming up ({LAT_WARMUP_RUNS} runs, discarded)...")
    for i in range(LAT_WARMUP_RUNS):
        pipeline.invoke(QUESTIONS[i % len(QUESTIONS)])
        pace()

    stages = {k: [] for k in ("total", "guardrails", "planner", "retrieval", "generation", "ttft", "answer_len")}
    print("[latency] measuring...")
    for q in QUESTIONS:
        for _ in range(LAT_REPEATS):
            start = time.perf_counter()
            answer, st = pipeline.invoke_staged(q)
            stages["total"].append((time.perf_counter() - start) * 1000)
            for k in ("guardrails", "planner", "retrieval", "generation", "ttft"):
                stages[k].append(st[k])
            stages["answer_len"].append(len(answer or ""))
            pace()
    return stages


def lat_summarize(samples):
    clean = [s for s in samples if not math.isnan(s)]
    return {"n": len(clean), "mean": sum(clean) / len(clean), "p50": percentile(clean, 50),
            "p95": percentile(clean, 95), "p99": percentile(clean, 99),
            "min": min(clean), "max": max(clean)}


def lat_report(r):
    print("\n" + "=" * 78 + "\nLATENCY (milliseconds)\n" + "=" * 78)
    print(f"{'stage':<12} | {'n':<4}{'mean':>9}{'p50':>9}{'p95':>9}{'p99':>9}{'min':>9}{'max':>9}")
    print("-" * 78)
    for label, key in [("end-to-end", "total"), ("ttft", "ttft"), ("guardrails", "guardrails"),
                       ("planner", "planner"), ("retrieval", "retrieval"), ("generation", "generation")]:
        s = lat_summarize(r[key])
        print(f"{label:<12} | {s['n']:<4}{s['mean']:>9.1f}{s['p50']:>9.1f}{s['p95']:>9.1f}"
              f"{s['p99']:>9.1f}{s['min']:>9.1f}{s['max']:>9.1f}")
    print("-" * 78)
    print(f"avg answer length: {sum(r['answer_len']) / len(r['answer_len']):.0f} chars "
          f"(latency scales with output length)")
    print("=" * 78)
    for label, key, budget in [("full answer", "total", SLO_P95_MS), ("first token (perceived)", "ttft", SLO_TTFT_P95_MS)]:
        p95 = lat_summarize(r[key])["p95"]
        print(f"SLO: {label:<24} p95 <= {budget:>5} ms  ->  p95 = {p95:7.0f} ms   [{'PASS' if p95 <= budget else 'FAIL'}]")
    print("=" * 78)


def run_latency(pipeline, verbose=True):
    r = lat_benchmark(pipeline)
    if verbose:
        lat_report(r)
    total, ttft = lat_summarize(r["total"]), lat_summarize(r["ttft"])
    return {
        "e2e_mean_ms": total["mean"], "e2e_p50_ms": total["p50"],
        "e2e_p95_ms": total["p95"], "e2e_p99_ms": total["p99"],
        "avg_answer_len": sum(r["answer_len"]) / len(r["answer_len"]),
        "slo_e2e_pass": total["p95"] <= SLO_P95_MS,
        "ttft_p95_ms": ttft["p95"], "slo_ttft_pass": ttft["p95"] <= SLO_TTFT_P95_MS,
        "guardrails_p95_ms": lat_summarize(r["guardrails"])["p95"],
        "planner_p95_ms": lat_summarize(r["planner"])["p95"],
        "retrieval_p95_ms": lat_summarize(r["retrieval"])["p95"],
        "generation_p95_ms": lat_summarize(r["generation"])["p95"],
    }


# ================================ COST ===============================
COST_REPEATS = 3

# Groq on-demand list price for llama-3.3-70b-versatile, USD per 1M tokens.
# *** Re-check https://groq.com/pricing before trusting a budget -- prices change. ***
# Planner and generator both run on this model via Portkey (primary target).
PRICE_INPUT_PER_1M        = 0.59
PRICE_CACHED_INPUT_PER_1M = 0.59    # no cached-input discount assumed for this model
PRICE_OUTPUT_PER_1M       = 0.79

QUERIES_PER_DAY = 1000
COST_BUDGET_PER_QUERY_USD = 0.0025


def cost_measure_tokens(pipeline, question):
    """planner tokens (LangChain usage callback) + generator tokens (re-run of the same prompt)."""
    out, planner = pipeline.invoke_with_usage(question)
    _, gen = generate_with_usage(question, out["context"])
    return {"input": planner["input"] + gen["input"],
            "output": planner["output"] + gen["output"],
            "cached": gen["cached"]}


def cost_usd(inp, out, cached):
    uncached = max(inp - cached, 0)
    c_in = uncached / 1e6 * PRICE_INPUT_PER_1M
    c_cached = cached / 1e6 * PRICE_CACHED_INPUT_PER_1M
    c_out = out / 1e6 * PRICE_OUTPUT_PER_1M
    return {"input": c_in, "cached": c_cached, "output": c_out, "total": c_in + c_cached + c_out}


def cost_benchmark(pipeline):
    rows = []
    print("[cost] measuring token usage...")
    for q in QUESTIONS:
        for _ in range(COST_REPEATS):
            tok = cost_measure_tokens(pipeline, q)
            rows.append({**tok, **{f"cost_{k}": v for k, v in cost_usd(tok["input"], tok["output"], tok["cached"]).items()}})
            pace()
    return rows


def cost_report(rows):
    avg_cost = col_avg(rows, "cost_total")
    out_share = 100 * col_avg(rows, "cost_output") / avg_cost if avg_cost else 0
    print("\n" + "=" * 70)
    print(f"COST  (llama-3.3-70b-versatile @ ${PRICE_INPUT_PER_1M}/${PRICE_OUTPUT_PER_1M} per 1M in/out; planner + generator)")
    print("=" * 70)
    print(f"samples                : {len(rows)}")
    print(f"avg input tokens       : {col_avg(rows, 'input'):8.0f}   ({col_avg(rows, 'cached'):.0f} cached)")
    print(f"avg output tokens      : {col_avg(rows, 'output'):8.0f}")
    print("-" * 70)
    print(f"avg cost / query       : ${avg_cost:.6f}")
    print(f"   min / max           : ${min(r['cost_total'] for r in rows):.6f} / ${max(r['cost_total'] for r in rows):.6f}")
    print(f"   input vs output     : {100 - out_share:.0f}% input / {out_share:.0f}% output")
    print("-" * 70)
    print(f"projection @ {QUERIES_PER_DAY}/day : ${avg_cost * QUERIES_PER_DAY:8.2f}/day   ${avg_cost * QUERIES_PER_DAY * 30:8.2f}/month")
    print("=" * 70)
    print(f"BUDGET: cost/query <= ${COST_BUDGET_PER_QUERY_USD:.6f}  ->  ${avg_cost:.6f}   "
          f"[{'PASS' if avg_cost <= COST_BUDGET_PER_QUERY_USD else 'FAIL'}]")
    print("=" * 70)
    print("note: excludes the NeMo guardrail LLM call (llama-3.1-8b-instant) -- slight underestimate.")


def run_cost(pipeline, verbose=True):
    rows = cost_benchmark(pipeline)
    if verbose:
        cost_report(rows)
    avg_cost = col_avg(rows, "cost_total")
    return {
        "cost_per_query_usd": avg_cost,
        "avg_input_tokens": col_avg(rows, "input"),
        "avg_output_tokens": col_avg(rows, "output"),
        "avg_cached_tokens": col_avg(rows, "cached"),
        "output_cost_share_pct": 100 * col_avg(rows, "cost_output") / avg_cost if avg_cost else 0.0,
        "monthly_usd": avg_cost * QUERIES_PER_DAY * 30,
        "budget_pass": avg_cost <= COST_BUDGET_PER_QUERY_USD,
    }


# ============================= RELIABILITY ============================
REL_REPEATS = 5
MAX_RETRIES = 2
BACKOFF_BASE_S = 0.5


class Reliability:
    def __init__(self):
        self.calls = self.successes = self.failures = self.retries = 0   # retries = total retry ATTEMPTS


def _checked_invoke(pipeline, question):
    """
    A request only counts as a success if it returns a real answer. Two silent
    failures in this app would otherwise read as 100% success:
      * search_enterprise_knowledge() swallows Qdrant/embedding errors and returns []
      * the planner/generator can return an empty string
    """
    out = pipeline.invoke(question)
    if not out["answer"]:
        raise RuntimeError("empty answer")
    if not out["blocked"] and not out["context"] and "Conversational" not in " ".join(out["plan"]):
        raise RuntimeError("retrieval returned no context (Qdrant / embedding failure swallowed?)")
    return out


def call_with_retries(fn, rel):
    rel.calls += 1
    for attempt in range(MAX_RETRIES + 1):
        try:
            result = fn()
            rel.successes += 1
            return result
        except Exception as e:
            if attempt < MAX_RETRIES:
                rel.retries += 1
                time.sleep(BACKOFF_BASE_S * (2 ** attempt))
            else:
                rel.failures += 1
                print(f"[reliability] FAILED after {MAX_RETRIES} retries: {e}")
                return None


def run_reliability(pipeline, verbose=True):
    rel = Reliability()
    print("[reliability] measuring...")
    for q in QUESTIONS:
        for _ in range(REL_REPEATS):
            call_with_retries(lambda q=q: _checked_invoke(pipeline, q), rel)
            pace()

    calls = rel.calls or 1
    m = {"total_requests": rel.calls, "success_rate": 100 * rel.successes / calls,
         "error_rate": 100 * rel.failures / calls, "retry_rate": 100 * rel.retries / calls}
    if verbose:
        print("\n" + "=" * 60 + "\nRELIABILITY\n" + "=" * 60)
        print(f"total requests : {rel.calls}\nsuccessful     : {rel.successes}\nfailed         : {rel.failures}")
        print("-" * 60)
        print(f"success rate   : {m['success_rate']:.2f}%\nerror rate     : {m['error_rate']:.2f}%\nretry rate     : {m['retry_rate']:.2f}%")
        print("=" * 60)
    return m


# ============================== ENTRYPOINT ============================
def run_ops(pipeline=None, verbose=True):
    pipeline = pipeline or RagPipeline()
    snapshot = {}
    snapshot.update({f"latency.{k}": v for k, v in run_latency(pipeline, verbose).items()})
    snapshot.update({f"cost.{k}": v for k, v in run_cost(pipeline, verbose).items()})
    snapshot.update({f"reliability.{k}": v for k, v in run_reliability(pipeline, verbose).items()})
    return snapshot


def main():
    snapshot = run_ops(verbose=True)
    print("\n" + "=" * 70 + "\nOPERATIONAL SNAPSHOT  (feeds regression testing)\n" + "=" * 70)
    for k, v in snapshot.items():
        print(f"  {k:<30} {v:.6f}" if isinstance(v, float) else f"  {k:<30} {v}")
    print("=" * 70)


if __name__ == "__main__":
    main()
