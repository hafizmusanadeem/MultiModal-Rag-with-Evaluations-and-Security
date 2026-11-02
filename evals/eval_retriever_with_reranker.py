# evals/eval_retriever_with_reranker.py
"""
A/B: plain Qdrant vector search (top-5) vs Qdrant top-15 + FlashRank rerank (top-5).
Answers "is the reranker earning its latency?" on YOUR golden set.

    python -m evals.eval_retriever_with_reranker
"""
from evals.rag_adapter import VectorRetriever, RerankingRetriever
from evals import eval_retriever


def main():
    base = eval_retriever.run(VectorRetriever(k=5), label="vector_top5")
    rer = eval_retriever.run(RerankingRetriever(), label="rerank_15_to_5")

    print("\n" + "=" * 78)
    print(f"{'metric':<24}{'vector_top5':>16}{'rerank_15->5':>16}{'delta':>12}")
    print("-" * 78)
    for name in rer:
        b, r = base.get(name, {}).get("avg_score", float("nan")), rer[name]["avg_score"]
        print(f"{name:<24}{b:>16.3f}{r:>16.3f}{r - b:>+12.3f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
