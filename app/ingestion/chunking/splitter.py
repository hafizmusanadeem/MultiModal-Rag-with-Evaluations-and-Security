import re
from typing import List

import logfire

# Best config from evals/chunk_tuning (score 0.906): size 950, overlap 30% (285).
DEFAULT_CHUNK_SIZE = 950
DEFAULT_OVERLAP = 285

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _units(text: str, size: int):
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
            if len(w) > size:  # pathological token (long URL/base64)
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


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[str]:
    """
    Greedy sentence packer with overlap. Chunks never exceed chunk_size chars.
    Does NOT depend on blank lines (PDF text from pypdf rarely contains them):
    it splits on sentence ends and newlines, then packs units up to chunk_size.
    The trailing sentences of each chunk (up to `overlap` chars) are repeated
    at the start of the next.
    """
    with logfire.span("✂️ Text Chunking", text_length=len(text)):
        if not text.strip():
            return []

        chunks: List[str] = []
        cur: List[str] = []
        cur_len = 0

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

        valid = [c for c in chunks if c.strip()]
        logfire.info(f"✅ Generated {len(valid)} chunks (size={chunk_size}, overlap={overlap})")
        return valid