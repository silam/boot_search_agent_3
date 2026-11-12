"""
Keyword search — BM25 (Okapi).

    score(q, d) = Σ_t IDF(t) · tf(t,d)·(k1+1) / ( tf(t,d) + k1·(1 − b + b·|d|/avgdl) )

  * IDF: rare terms ("kiltie", "Vibram") count more than common ones ("boot").
  * k1 (~1.2-2.0): term-frequency saturation — the 5th "waterproof" adds little.
  * b (~0.75): length normalisation — long chunks don't win just by being long.

DISCUSSION NOTE — why keyword search in an "AI" system?
  Dense embeddings are bad at exact tokens: SKUs ("BOOT-042"), model names ("Cascade"),
  rare jargon ("kiltie", "EH rated"), and numbers. BM25 nails those and is cheap and
  explainable. Dense vectors nail paraphrase ("keeps feet dry" ~ "waterproof"). Hybrid
  search = use both and fuse (see fusion.py). This is the standard production pattern.

Implementation: rank_bm25 in memory. Rebuilt from the processed chunks file at startup
(seconds for thousands of chunks). At millions of docs you'd move BM25 into the search
engine itself: Qdrant sparse vectors with the IDF modifier, OpenSearch, or Elasticsearch.
"""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from app.core.models import Chunk, Filters, RetrievedItem

STOPWORDS = {
    "a", "an", "the", "and", "or", "for", "with", "of", "to", "in", "on", "at", "by", "is", "it",
    "this", "that", "my", "me", "i", "you", "are", "be", "boot", "boots", "want", "need", "looking",
    "show", "find", "some", "any", "good", "best", "pair",
}
_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def stem(token: str) -> str:
    """Tiny, conservative suffix stripper (a real system would use Snowball/Porter).
    Must be applied identically to documents and queries."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    if len(token) > 6 and token.endswith("ing"):
        return token[:-3]
    return token


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        if tok in STOPWORDS:
            continue
        out.append(stem(tok))
        if "-" in tok:  # "slip-resistant" -> also "slip", "resistant"
            out.extend(stem(p) for p in tok.split("-") if p not in STOPWORDS)
    return out


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks], k1=k1, b=b)

    def search(self, query: str, limit: int, filters: Filters | None = None, catalog=None) -> list[RetrievedItem]:
        q = tokenize(query)
        if not q:
            return []
        scores = self.bm25.get_scores(q)
        best: dict[str, RetrievedItem] = {}
        for idx in scores.argsort()[::-1]:
            score = float(scores[idx])
            if score <= 0:
                break
            c = self.chunks[idx]
            # BM25 lives outside Qdrant, so filters are applied here (post-filter)
            if filters and catalog is not None and not filters.matches(catalog[c.boot_id]):
                continue
            if c.boot_id not in best:
                best[c.boot_id] = RetrievedItem(boot_id=c.boot_id, score=score, source="bm25", evidence=c.text)
            if len(best) >= limit:
                break
        return list(best.values())
