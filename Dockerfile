FROM python:3.11-slim-bookworm

# Patch OS-level CVEs, then install system deps required by native packages
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLASHRANK_CACHE_DIR=/opt/flashrank \
    FASTEMBED_CACHE_PATH=/opt/fastembed \
    REQUIRE_GEMINI=1

# Requirements first so this layer is cached across code-only changes.
COPY requirements-prod.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements-prod.txt

# Bake models into the image so cold starts don't download them.
# 1) FlashRank reranker (ONNX)
RUN python -c "from flashrank import Ranker; Ranker(cache_dir='/opt/flashrank')"
# 2) NeMo Guardrails' default intent-matching embedder (FastEmbed all-MiniLM-L6-v2).
#    Best effort: if this ever fails the build continues and NeMo downloads it at startup instead.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2', cache_dir='/opt/fastembed')" \
    || echo "WARN: fastembed pre-bake skipped; NeMo will download it at startup"

# Only the app package — evals/, ui/, DATA/, processed_data/ stay out.
COPY app/ ./app/

EXPOSE 8080
# Cloud Run injects $PORT (8080). One worker only: LangGraph MemorySaver is in-process.
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}
