# evals/eval_reliability.py -- standalone entrypoint (logic lives in eval_ops.py)
from evals.rag_adapter import RagPipeline
from evals.eval_ops import run_reliability

if __name__ == "__main__":
    run_reliability(RagPipeline(), verbose=True)
