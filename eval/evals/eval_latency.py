# evals/eval_latency.py -- standalone entrypoint (logic lives in eval_ops.py)
from evals.rag_adapter import RagPipeline
from evals.eval_ops import run_latency

if __name__ == "__main__":
    run_latency(RagPipeline(), verbose=True)
