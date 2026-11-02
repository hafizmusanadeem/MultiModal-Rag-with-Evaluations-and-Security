# evals/eval_toxicity.py -- standalone entrypoint (logic lives in eval_safety.py)
from evals.rag_adapter import RagPipeline
from evals.eval_safety import run_toxicity

if __name__ == "__main__":
    run_toxicity(RagPipeline())
