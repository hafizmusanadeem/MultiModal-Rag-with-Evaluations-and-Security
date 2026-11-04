import os
import sys
import time
import uuid
import json
import hashlib
import logfire

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings
from app.services.retrieval.embedding import embed_texts, get_embedding_dim
from app.ingestion.loaders.pdf import parse_pdf
from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.text import parse_text
from app.ingestion.chunking.splitter import chunk_text

logfire.configure(service_name="enterprise-ingestion-service")

# Local folder where parsed + chunked JSON metadata is saved (replaces GCS processed bucket)
PROCESSED_DATA_DIR = "processed_data"

# Chunks embedded + upserted per step. Each step is checkpointed in Qdrant,
# so a crash/quota failure loses at most one batch.
UPSERT_BATCH = int(os.getenv("INGEST_BATCH", "50"))
# Pause between batches to stay under the per-minute embedding quota.
INGEST_PACE_S = float(os.getenv("INGEST_PACE_S", "2"))

# Initialize Qdrant Client
qdrant_client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)


def save_processed_locally(data: dict, source_type: str, filename: str) -> str:
    """Save parsed chunk metadata as JSON in processed_data/<source_type>/."""
    folder = os.path.join(PROCESSED_DATA_DIR, source_type)
    os.makedirs(folder, exist_ok=True)
    dest = os.path.join(folder, f"{filename}.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return dest


def _chunk_id(source_type: str, filename: str, index: int, chunk: str) -> str:
    """Deterministic point ID: same file + position + content => same ID.
    Makes ingestion idempotent and resumable (re-run skips what's already stored)."""
    digest = hashlib.sha1(chunk.encode("utf-8")).hexdigest()[:16]
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_type}/{filename}#{index}:{digest}"))


def _existing_ids(ids: list[str]) -> set[str]:
    """Which of these point IDs are already in the collection?"""
    found: set[str] = set()
    for i in range(0, len(ids), 256):
        recs = qdrant_client.retrieve(
            collection_name=settings.QDRANT_COLLECTION,
            ids=ids[i : i + 256],
            with_payload=False,
            with_vectors=False,
        )
        found.update(str(r.id) for r in recs)
    return found


def process_file(file_path: str, filename: str, source_type: str):
    """Parse → chunk → save locally → embed → index in Qdrant (resumable)."""
    with logfire.span("Processing File", file=filename, source=source_type):
        try:
            # 1. Extract text based on file extension
            ext = filename.lower().rsplit(".", 1)[-1]
            if ext == "pdf":
                full_text = parse_pdf(file_path)
            elif ext in ("html", "htm"):
                full_text = parse_html(file_path)
            elif ext == "txt":
                full_text = parse_text(file_path)
            elif ext in ("docx", "pptx"):
                from app.ingestion.loaders.office import parse_office
                full_text = parse_office(file_path)
            else:
                logfire.warning(f"Skipping unsupported file type: {filename}")
                return

            if not full_text or not full_text.strip():
                logfire.warning(f"No text extracted from {filename} — skipping.")
                return

            # 2. Chunk text
            chunks = chunk_text(full_text)
            if not chunks:
                return

            # 3. Save processed metadata locally
            processed_data = {
                "filename": filename,
                "source_type": source_type,
                "chunks": chunks,
            }
            local_path = save_processed_locally(processed_data, source_type, filename)
            logfire.info(f"Saved processed data → {local_path}")

            # 4. Embed and index in Qdrant, batch by batch, skipping chunks already stored
            with logfire.span("Vectorizing & Indexing"):
                ids = [_chunk_id(source_type, filename, i, c) for i, c in enumerate(chunks)]
                already = _existing_ids(ids)
                todo = [i for i in range(len(chunks)) if ids[i] not in already]

                if already:
                    logfire.info(
                        f"Resuming {filename}: {len(already)}/{len(chunks)} chunks already "
                        f"indexed, {len(todo)} remaining."
                    )
                if not todo:
                    logfire.info(f"{filename} is already fully indexed — nothing to do.")
                    return

                indexed = len(already)
                for start in range(0, len(todo), UPSERT_BATCH):
                    idxs = todo[start : start + UPSERT_BATCH]
                    vectors = embed_texts([chunks[i] for i in idxs])  # retries/waits on 429
                    points = [
                        models.PointStruct(
                            id=ids[i],
                            vector=vec,
                            payload={
                                "text": chunks[i],
                                "source": filename,
                                "source_type": source_type,
                            },
                        )
                        for i, vec in zip(idxs, vectors)
                    ]
                    qdrant_client.upsert(
                        collection_name=settings.QDRANT_COLLECTION,
                        points=points,
                    )
                    indexed += len(points)
                    logfire.info(f"Indexed {indexed}/{len(chunks)} chunks of {filename}.")
                    if start + UPSERT_BATCH < len(todo):
                        time.sleep(INGEST_PACE_S)

                logfire.info(f"Finished indexing {filename}: {indexed}/{len(chunks)} chunks.")

        except Exception as e:
            logfire.error(
                f"Failed to process {filename}: {e}. "
                f"Progress is saved — re-run the same command (without --wipe) to resume."
            )


def process_directory(dir_path: str, source_type: str):
    """Process every file in a directory."""
    with logfire.span("Scanning Directory", path=dir_path, source=source_type):
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        logfire.info(f"Found {len(files)} files in {dir_path}.")
        for filename in files:
            process_file(os.path.join(dir_path, filename), filename, source_type)


def run_universal_ingestion(base_dir: str, explicit_source_type: str = None, wipe: bool = False):
    """
    Scan base_dir, map sub-folders to source types, and ingest all documents.
    Pass --wipe to drop and recreate the Qdrant collection before ingestion.
    """
    with logfire.span("Universal Ingestion Started", base_directory=base_dir):

        # Wipe collection if requested
        if wipe:
            with logfire.span("Wiping Collection"):
                if qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
                    qdrant_client.delete_collection(settings.QDRANT_COLLECTION)
                    logfire.info(f"Collection '{settings.QDRANT_COLLECTION}' deleted.")

        # Recreate collection — dimension resolved at runtime after embedding model probe
        if not qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
            dim = get_embedding_dim()
            qdrant_client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                ),
            )
            logfire.info(
                f"Created collection '{settings.QDRANT_COLLECTION}' "
                f"({dim}-dim, Cosine)."
            )

        # Route to sub-folders or treat the whole dir as one source
        subdirs = [
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
        ]

        if not subdirs:
            if explicit_source_type:
                source_type = explicit_source_type
            else:
                base_name = os.path.basename(os.path.normpath(base_dir)).lower()
                source_type = (
                    "true" if "true" in base_name
                    else "noisy" if "noisy" in base_name
                    else "general"
                )
            logfire.info(f"No sub-folders found — processing '{base_dir}' as '{source_type}'.")
            process_directory(base_dir, source_type)
        else:
            for subdir in subdirs:
                source_type = (
                    "true" if "true" in subdir.lower()
                    else "noisy" if "noisy" in subdir.lower()
                    else subdir
                )
                process_directory(os.path.join(base_dir, subdir), source_type)


if __name__ == "__main__":
    # Usage:
    #   python -m app.ingestion.processor DATA --wipe
    #   python -m app.ingestion.processor DATA/true_data true
    wipe_requested = "--wipe" in sys.argv
    clean_args = [a for a in sys.argv if a != "--wipe"]

    target_dir = clean_args[1] if len(clean_args) > 1 else "DATA"
    explicit_type = clean_args[2] if len(clean_args) > 2 else None

    if not os.path.exists(target_dir):
        print(f"Error: path '{target_dir}' does not exist.")
        sys.exit(1)

    run_universal_ingestion(target_dir, explicit_source_type=explicit_type, wipe=wipe_requested)
    logfire.info("Ingestion job completed.")