# evals/eval_cost.py -- standalone entrypoint (logic lives in eval_ops.py)
from evals.rag_adapter import RagPipeline
from evals.eval_ops import run_cost

if __name__ == "__main__":
    run_cost(RagPipeline(), verbose=True)
