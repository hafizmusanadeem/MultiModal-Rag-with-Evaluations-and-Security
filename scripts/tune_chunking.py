#!/usr/bin/env python
"""
tune_chunking.py -- find the best chunk_size / chunk_overlap for YOUR corpus.

Place at:  scripts/tune_chunking.py        (project root = folder containing app/, evals/)
Run from the project root:
    python -m scripts.tune_chunking --dry-run          # show chunk counts / embed cost, embed nothing
    python -m scripts.tune_chunking                    # full grid search
    python -m scripts.tune_chunking --sizes 600 1000 1500 --overlaps 0 0.15 --k 5 --rerank

HOW IT WORKS
  1. Rebuilds the full text of every document from processed_data/**/*.json
     (the old chunks joined back together).
  2. For each (chunk_size, overlap) pair: re-chunks the corpus, embeds the chunks
     with the SAME embedding module as production (app.services.retrieval.embedding),
     and retrieves top-k for every golden question using IN-MEMORY cosine search.
     => your Qdrant collection is NOT touched, no need to wipe/re-ingest to experiment.
  3. NO golden dataset needed: the first run samples passages from processed_data/,
     asks your Groq LLM (via Portkey) to write one question + the verbatim evidence
     sentence(s) per passage, and saves them to evals/chunk_tuning/synthetic_qa.json.
     Every later run reuses that file, so all configs are judged on the SAME questions.
     Retrieval is then scored with deterministic token-overlap metrics (no LLM judge).
  4. Prints a ranked table, saves CSV/JSON, and recommends a config.

Embeddings are cached on disk (keyed by model + text hash), so re-runs and
overlapping configs only pay for chunks never embedded before.

METRICS (per question, averaged)
  recall     avg fraction of each golden relevant_context's tokens covered by the
             union of the top-k chunks            -> "did we retrieve the needed info?"
  hit        fraction of golden contexts covered >= HIT_THRESHOLD by the top-k union
  mrr        1/rank of the first relevant chunk   -> "how high is the right chunk ranked?"
  precision  fraction of top-k chunks that are relevant -> "how much junk did we send?"
  ctx_chars  avg characters sent to the LLM (cost / latency / dilution proxy)
  score      composite = 0.4*recall + 0.3*hit + 0.2*mrr + 0.1*precision
A chunk is "relevant" if it covers >=60% of a golden context's tokens, OR >=50% of
the chunk's own tokens lie inside a golden context (so small chunks inside a long
golden passage still count).
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import pickle
import random
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ----------------------------------------------------------------------------

DEFAULT_SIZES = [100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650, 675, 700, 725, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]
DEFAULT_OVERLAP_FRACS = [0.1, 0.2, 0.3]       # overlap as a fraction of chunk_size
PROCESSED_DIR = os.path.join(ROOT, "processed_data")
OUT_DIR = os.path.join(ROOT, "evals", "chunk_tuning")
CACHE_PATH = os.path.join(OUT_DIR, "embedding_cache.pkl")
QA_PATH = os.path.join(OUT_DIR, "synthetic_qa.json")

HIT_THRESHOLD = 0.6
CHUNK_IN_REL_THRESHOLD = 0.5
FETCH_K = 15                                  # candidates fed to the reranker (mirrors retriever node)
W_RECALL, W_HIT, W_MRR, W_PREC = 0.4, 0.3, 0.2, 0.1


# ============================================================ corpus + goldens
def load_corpus(processed_dir, source_types=None, max_files=None, name_contains=None):
    """processed_data/<source_type>/<file>.json -> {filename: full_text}"""
    docs = {}
    paths = sorted(glob.glob(os.path.join(processed_dir, "**", "*.json"), recursive=True))
    for p in paths:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        if source_types and d.get("source_type") not in source_types:
            continue
        if name_contains and name_contains.lower() not in d.get("filename", "").lower():
            continue
        text = "\n\n".join(d.get("chunks", []))
        if text.strip():
            docs[f"{d.get('source_type', '?')}/{d['filename']}"] = text
        if max_files and len(docs) >= max_files:
            break
    return docs


_ws = re.compile(r"\s+")


def _corpus_fingerprint(docs):
    return hashlib.sha1(json.dumps(sorted((k, len(v)) for k, v in docs.items())).encode()).hexdigest()[:12]


def _pick_anchors(docs, n, passage_chars, seed):
    """Non-overlapping, sentence-trimmed, prose-like passages; round-robin across docs."""
    rng = random.Random(seed)
    per_doc = {}
    for name, text in docs.items():
        flat = _ws.sub(" ", text)
        cands = []
        for i in range(0, max(len(flat) - passage_chars, 1), passage_chars):
            seg = flat[i:i + passage_chars]
            m = re.search(r"[.!?]\s", seg)                    # start at a sentence boundary
            if m and i > 0:
                seg = seg[m.end():]
            last = max(seg.rfind(". "), seg.rfind("? "), seg.rfind("! "))
            if last > len(seg) * 0.6:
                seg = seg[:last + 1]
            seg = seg.strip()
            alpha = sum(c.isalpha() for c in seg) / max(len(seg), 1)
            digit = sum(c.isdigit() for c in seg) / max(len(seg), 1)
            if len(seg) >= 0.6 * passage_chars and alpha >= 0.70 and digit < 0.08:   # skip tables / TOCs
                cands.append(seg)
        rng.shuffle(cands)
        if cands:
            per_doc[name] = cands
    anchors, names = [], list(per_doc)
    while len(anchors) < n and any(per_doc.values()):
        for name in names:
            if per_doc[name] and len(anchors) < n:
                anchors.append((name, per_doc[name].pop()))
    return anchors


_QA_PROMPT = """You are building a retrieval test set. Read the PASSAGE and write ONE question that a real user
could ask, whose answer is stated in the passage.

Rules:
- The question must be self-contained and specific: name the topic/entity. Never say "the passage", "this document" or "the text".
- Paraphrase; do not copy long phrases from the passage into the question.
- Then copy VERBATIM the shortest 1-3 consecutive sentences from the passage that contain the answer ("evidence").

Return ONLY a JSON object: {{"question": "...", "evidence": "..."}}

PASSAGE:
{passage}"""


def generate_questions(docs, n, passage_chars, seed, pace_s=2.0):
    os.environ.setdefault("DISABLE_PORTKEY_CACHE", "1")
    from app.gateway import portkey_client                     # same Groq-via-Portkey path as production

    anchors = _pick_anchors(docs, n, passage_chars, seed)
    if not anchors:
        raise SystemExit("Could not find prose-like passages in the corpus to build questions from.")
    print(f"Generating {len(anchors)} synthetic questions (one-time; cached to {QA_PATH})...")
    out = []
    for i, (src, passage) in enumerate(anchors, 1):
        try:
            resp = portkey_client.chat.completions.create(
                messages=[{"role": "user", "content": _QA_PROMPT.format(passage=passage)}], temperature=0.2)
            raw = resp.choices[0].message.content
            data = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
            q, ev = data["question"].strip(), data["evidence"].strip()
        except Exception as e:
            print(f"  [{i}/{len(anchors)}] skipped ({type(e).__name__}: {str(e)[:80]})")
            time.sleep(pace_s)
            continue
        et, pt = tokens(ev), tokens(passage)
        if len(q) < 15 or len(ev) < 40 or not et or len(et & pt) / len(et) < 0.9:
            print(f"  [{i}/{len(anchors)}] skipped (evidence not verbatim / too short)")
        else:
            out.append({"query": q, "relevant": [ev], "source": src})
            print(f"  [{i}/{len(anchors)}] {q[:90]}")
        time.sleep(pace_s)
    if len(out) < 10:
        raise SystemExit(f"Only {len(out)} valid questions generated -- check your Groq/Portkey keys and retry.")
    return out


def get_questions(docs, args):
    fp = _corpus_fingerprint(docs)
    if os.path.exists(args.qa) and not args.regen:
        with open(args.qa, encoding="utf-8") as f:
            data = json.load(f)
        print(f"Reusing {len(data['questions'])} questions from {args.qa}")
        if data.get("fingerprint") not in (None, fp):
            print("  WARNING: processed_data changed since these questions were made. Use --regen to rebuild.")
        return data["questions"]
    qs = generate_questions(docs, args.n_questions, args.passage_chars, args.seed)
    os.makedirs(os.path.dirname(args.qa), exist_ok=True)
    with open(args.qa, "w", encoding="utf-8") as f:
        json.dump({"fingerprint": fp, "questions": qs}, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(qs)} questions -> {args.qa}  (open it and delete/edit any bad ones)")
    return qs


# ============================================================ chunker under test
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _units(text, size):
    """Sentence/line-level units, each guaranteed <= size (hard-split on words if needed)."""
    for raw in _SENT_SPLIT.split(text):
        u = raw.strip()
        if not u:
            continue
        if len(u) <= size:
            yield u
            continue
        buf = ""
        for w in u.split():
            if len(w) > size:                        # pathological token (long URL/base64)
                if buf:
                    yield buf
                    buf = ""
                for i in range(0, len(w), size):
                    yield w[i:i + size]
            elif len(buf) + len(w) + 1 > size:
                yield buf
                buf = w
            else:
                buf = f"{buf} {w}".strip()
        if buf:
            yield buf


def chunk_text(text, chunk_size, overlap):
    """
    Greedy sentence packer with overlap. Chunks never exceed chunk_size chars.
    Overlap = trailing sentences of the previous chunk (up to `overlap` chars)
    repeated at the start of the next one, so sentence boundaries are respected.
    Drop-in candidate for app/ingestion/chunking/splitter.py once you pick winners.
    """
    chunks, cur, cur_len = [], [], 0
    for u in _units(text, chunk_size):
        add = len(u) + (1 if cur else 0)
        if cur and cur_len + add > chunk_size:
            chunks.append(" ".join(cur))
            tail, tl = [], 0
            for t in reversed(cur):
                if tl + len(t) + 1 > overlap:
                    break
                tail.insert(0, t)
                tl += len(t) + 1
            while tail and sum(len(t) + 1 for t in tail) + len(u) > chunk_size:
                tail.pop(0)
            cur = tail
            cur_len = len(" ".join(cur))
            add = len(u) + (1 if cur else 0)
        cur.append(u)
        cur_len += add
    if cur:
        chunks.append(" ".join(cur))
    return [c for c in chunks if c.strip()]


def chunk_corpus(docs, size, overlap):
    out = []
    for text in docs.values():
        out.extend(chunk_text(text, size, overlap))
    return out


# ============================================================ embeddings (cached)
class Embedder:
    def __init__(self):
        from app.services.retrieval import embedding
        self.emb = embedding
        self.emb.get_embedding_dim()              # triggers lazy init + Gemini probe
        self.model_type = self.emb._model_type
        self.cache = {}
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, "rb") as f:
                self.cache = pickle.load(f)
        self.new_since_save = 0

    def _key(self, text):
        return hashlib.sha1(f"{self.model_type}|{text}".encode("utf-8")).hexdigest()

    def save(self):
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(self.cache, f)

    def embed_documents(self, texts):
        keys = [self._key(t) for t in texts]
        missing = {}
        for k, t in zip(keys, texts):
            if k not in self.cache:
                missing[k] = t
        if missing:
            ks, ts = list(missing), list(missing.values())
            print(f"      embedding {len(ts)} new chunks ({len(texts) - len(ts)} cached)...")
            vecs = self.emb.embed_texts(ts)
            for k, v in zip(ks, vecs):
                self.cache[k] = np.asarray(v, dtype=np.float32)
            self.save()
        return np.vstack([self.cache[k] for k in keys])

    def embed_queries(self, queries):
        out = []
        for q in queries:
            k = self._key("Q::" + q)
            if k not in self.cache:
                self.cache[k] = np.asarray(self.emb.embed_query(q), dtype=np.float32)
            out.append(self.cache[k])
        self.save()
        return np.vstack(out)


def _normalize(m):
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


# ============================================================ scoring
_TOK = re.compile(r"[a-z0-9_\-\.]+")


def tokens(s):
    return set(_TOK.findall(s.lower()))


def score_config(chunks, chunk_vecs, q_vecs, goldens, k, rerank):
    M = _normalize(chunk_vecs)
    Q = _normalize(q_vecs)
    sims = Q @ M.T
    fetch = max(FETCH_K, k) if rerank else k

    if rerank:
        from app.services.retrieval.ranking_service import rerank_documents

    recalls, hits, mrrs, precs, ctx = [], [], [], [], []
    for qi, g in enumerate(goldens):
        order = np.argsort(-sims[qi])[:fetch]
        cand = [chunks[i] for i in order]
        retrieved = rerank_documents(g["query"], cand, top_n=k) if rerank else cand[:k]

        rel_toks = [t for t in (tokens(r) for r in g["relevant"]) if t]
        ret_toks = [tokens(t) for t in retrieved]
        union = set().union(*ret_toks) if ret_toks else set()

        covs = [len(rt & union) / len(rt) for rt in rel_toks]
        recalls.append(float(np.mean(covs)))
        hits.append(float(np.mean([c >= HIT_THRESHOLD for c in covs])))

        flags = []
        for ct in ret_toks:
            rel = False
            if ct:
                for rt in rel_toks:
                    inter = len(rt & ct)
                    if inter / len(rt) >= HIT_THRESHOLD or inter / len(ct) >= CHUNK_IN_REL_THRESHOLD:
                        rel = True
                        break
            flags.append(rel)
        first = next((i for i, f in enumerate(flags) if f), None)
        mrrs.append(1.0 / (first + 1) if first is not None else 0.0)
        precs.append(float(np.mean(flags)) if flags else 0.0)
        ctx.append(sum(len(t) for t in retrieved))

    recall, hit, mrr, prec = map(lambda a: float(np.mean(a)), (recalls, hits, mrrs, precs))
    return {
        "recall": recall, "hit": hit, "mrr": mrr, "precision": prec,
        "ctx_chars": float(np.mean(ctx)),
        "score": W_RECALL * recall + W_HIT * hit + W_MRR * mrr + W_PREC * prec,
    }


# ============================================================ main
def main():
    ap = argparse.ArgumentParser(description="Grid-search chunk size / overlap against your golden set.")
    ap.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES, help="chunk sizes in chars")
    ap.add_argument("--overlaps", type=float, nargs="+", default=DEFAULT_OVERLAP_FRACS, help="overlap as a FRACTION of chunk size (e.g. 0 0.1 0.2). Values >=1 are treated as absolute chars.")
    ap.add_argument("--name-contains", default=None, help="only keep docs whose filename contains this substring (case-insensitive)")
    ap.add_argument("--k", type=int, default=5, help="chunks passed to the generator (production: 5)")
    ap.add_argument("--rerank", action="store_true", help="mimic production: top-15 -> FlashRank -> top-k")
    ap.add_argument("--processed-dir", default=PROCESSED_DIR)
    ap.add_argument("--qa", default=QA_PATH, help="synthetic question file (created on first run; you can hand-edit it)")
    ap.add_argument("--n-questions", type=int, default=40, help="how many questions to generate")
    ap.add_argument("--passage-chars", type=int, default=1500, help="size of passage each question is written from")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--regen", action="store_true", help="regenerate the question file")
    ap.add_argument("--questions-only", action="store_true", help="generate the question file, then stop")
    ap.add_argument("--source-types", nargs="+", default=None, help="e.g. true  (default: all)")
    ap.add_argument("--max-files", type=int, default=None, help="limit docs while experimenting")
    ap.add_argument("--dry-run", action="store_true", help="only print chunk counts; no embedding calls")
    args = ap.parse_args()

    docs = load_corpus(args.processed_dir, args.source_types, args.max_files, args.name_contains)
    if not docs:
        raise SystemExit(f"No documents found under {args.processed_dir}")
    total_chars = sum(len(t) for t in docs.values())
    print(f"Corpus: {len(docs)} docs, {total_chars:,} chars | k={args.k}"
          f"{' (+rerank)' if args.rerank else ''}")

    grid = []
    for s in args.sizes:
        for o in sorted(set(args.overlaps)):
            ov = int(o) if o >= 1 else int(round(s * o))
            if ov < s:
                grid.append((s, ov))

    plan, est_total = [], 0
    for s, ov in grid:
        chunks = chunk_corpus(docs, s, ov)
        plan.append((s, ov, chunks))
        est_total += len(chunks)
        print(f"  size={s:<5} overlap={ov:<4} -> {len(chunks):>6} chunks")
    print(f"Upper bound on embedding calls: {est_total:,} chunks (cached ones are free).")
    if args.dry_run:
        return

    goldens = get_questions(docs, args)
    if args.questions_only:
        return

    emb = Embedder()
    print(f"Embedding model: {emb.model_type}"
          f"{'  <-- FALLBACK in use (Gemini unreachable)!' if emb.model_type == 'fallback' else ''}")
    q_vecs = emb.embed_queries([g["query"] for g in goldens])

    rows = []
    for n, (s, ov, chunks) in enumerate(plan, 1):
        t0 = time.time()
        print(f"[{n}/{len(plan)}] size={s} overlap={ov} ({len(chunks)} chunks)")
        vecs = emb.embed_documents(chunks)
        m = score_config(chunks, vecs, q_vecs, goldens, args.k, args.rerank)
        row = {"chunk_size": s, "overlap": ov, "overlap_pct": round(100 * ov / s),
               "n_chunks": len(chunks),
               "avg_chunk_chars": round(sum(map(len, chunks)) / len(chunks), 1), **m}
        rows.append(row)
        print(f"      score={m['score']:.3f} recall={m['recall']:.3f} hit={m['hit']:.3f} "
              f"mrr={m['mrr']:.3f} prec={m['precision']:.3f} ctx={m['ctx_chars']:.0f} ({time.time() - t0:.1f}s)")

    rows.sort(key=lambda r: (-r["score"], r["ctx_chars"]))

    print("\n" + "=" * 100)
    print(f"{'rank':<5}{'size':>6}{'ovlp':>6}{'ovlp%':>7}{'chunks':>8}{'score':>8}{'recall':>8}"
          f"{'hit':>7}{'mrr':>7}{'prec':>7}{'ctx_chars':>11}")
    print("-" * 100)
    for i, r in enumerate(rows, 1):
        print(f"{i:<5}{r['chunk_size']:>6}{r['overlap']:>6}{r['overlap_pct']:>7}{r['n_chunks']:>8}"
              f"{r['score']:>8.3f}{r['recall']:>8.3f}{r['hit']:>7.3f}{r['mrr']:>7.3f}"
              f"{r['precision']:>7.3f}{r['ctx_chars']:>11.0f}")
    print("=" * 100)

    best = rows[0]
    econ = min((r for r in rows if r["score"] >= best["score"] - 0.02), key=lambda r: r["ctx_chars"])
    print(f"BEST SCORE      : chunk_size={best['chunk_size']}, overlap={best['overlap']} "
          f"({best['overlap_pct']}%)  score={best['score']:.3f}")
    print(f"MOST ECONOMICAL : chunk_size={econ['chunk_size']}, overlap={econ['overlap']} "
          f"({econ['overlap_pct']}%)  score={econ['score']:.3f}, ctx={econ['ctx_chars']:.0f} chars "
          f"(within 0.02 of best, least context sent to the LLM)")
    if len(goldens) < 30:
        print(f"NOTE: only {len(goldens)} golden questions -- differences under ~0.02-0.03 are noise. "
              f"Prefer the economical pick, or add more goldens.")

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUT_DIR, f"results_{tag}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT_DIR, f"results_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"embedding_model": emb.model_type, "k": args.k, "rerank": args.rerank,
                   "n_goldens": len(goldens), "n_docs": len(docs), "results": rows}, f, indent=2)
    print(f"\nSaved -> {csv_path}")
    print("Next: set the winning size/overlap in app/ingestion/chunking/splitter.py "
          "(port chunk_text above), then re-run: python -m app.ingestion.processor DATA --wipe")


if __name__ == "__main__":
    main()