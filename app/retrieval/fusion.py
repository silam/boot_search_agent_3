"""
Reciprocal Rank Fusion (RRF) — how we merge keyword, text-vector and image-vector results.

    RRF(d) = Σ_r  w_r / (k + rank_r(d))          (rank starts at 1, k = 60)

INTERVIEW NOTE — why RRF instead of adding scores?
  Scores from different retrievers are on different scales: BM25 is unbounded (0..30+),
  cosine is in [-1, 1], and image cosine has its own distribution. Adding them requires
  fragile normalisation. RRF uses only RANKS, so it is scale-free, needs no training, and
  rewards documents that several retrievers agree on. `k` dampens the advantage of
  rank 1 vs rank 2. Per-retriever weights let the agent say "this is mostly an image
  query" (image weight 1.0, text 0.5).

  Alternatives to mention: weighted score fusion with min-max normalisation, or a
  learned/cross-encoder re-ranker on the fused top-50 (better quality, more latency).
"""
from __future__ import annotations

from collections import defaultdict

from app.core.models import RetrievedItem


def reciprocal_rank_fusion(
    result_lists: dict[str, list[RetrievedItem]],
    weights: dict[str, float] | None = None,
    k: int = 60,
) -> list[tuple[str, float, list[str], str | None]]:
    """Returns [(boot_id, fused_score, contributing_sources, best_evidence)] sorted by score."""
    weights = weights or {}
    scores: dict[str, float] = defaultdict(float)
    sources: dict[str, list[str]] = defaultdict(list)
    evidence: dict[str, tuple[int, str]] = {}

    for name, items in result_lists.items():
        w = weights.get(name, 1.0)
        for rank, item in enumerate(items, start=1):
            scores[item.boot_id] += w / (k + rank)
            sources[item.boot_id].append(name)
            # keep the evidence from whichever retriever ranked this boot highest
            if item.evidence and (item.boot_id not in evidence or rank < evidence[item.boot_id][0]):
                evidence[item.boot_id] = (rank, item.evidence)

    fused = [(bid, s, sources[bid], evidence.get(bid, (0, None))[1]) for bid, s in scores.items()]
    return sorted(fused, key=lambda x: x[1], reverse=True)
