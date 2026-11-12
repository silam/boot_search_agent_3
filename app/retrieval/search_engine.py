"""
SearchEngine — the retrieval "tools" the agent can call, plus hybrid orchestration.

Retrievers (each returns product-level RetrievedItems):
  keyword(query)        -> BM25 over chunks                       (exact terms, SKUs, jargon)
  semantic(query)       -> text embedding vs text-chunk vectors   (meaning / paraphrase)
  image(imgs)           -> image embedding vs EVERY product photo (visual similarity)
  text_to_image(query)  -> CLIP text embedding vs product photos  (cross-modal; clip backend only)

Multi-image search, both sides:
  * catalog side: each boot has 8-10 photos, each its own vector. Qdrant groups photo hits by
    boot and the boot's score is its best photo's score (MaxSim), see QdrantStore.search_images.
  * query side: a shopper may upload several photos of the boot they want (e.g. a side shot
    and a sole shot). Each query photo is searched separately and a boot's final score is the
    AVERAGE over query photos of its best match:  score(b) = mean_q max_p sim(q, p)
    That's the symmetric "MaxSim" (Chamfer) similarity between two sets of images: a boot
    must look right from every angle the shopper showed, while each query photo only needs
    to match the one catalog view that shows the same angle.

`run(plan)` executes the retrievers the agent selected, fuses them with weighted RRF,
applies hard filters, and hydrates full Boot objects.

DISCUSSION NOTE — combined "description + image" search:
  The user uploads a photo and types "but waterproof and under $200". The image retriever
  finds visually similar boots; the text retrievers find boots matching the words; the
  filters (waterproof, max_price) are enforced on all of them; RRF rewards boots that
  appear in BOTH lists. That's multimodal hybrid retrieval.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
from pydantic import BaseModel, ConfigDict

from app.core.models import Boot, Chunk, Filters, RetrievedItem, SearchResult
from app.embeddings.base import ImageEmbedder, TextEmbedder
from app.retrieval.bm25_index import BM25Index
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.qdrant_store import QdrantStore

ALL_TOOLS = ("keyword", "semantic", "image", "text_to_image")


class RetrievalPlan(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    query: str = ""
    images: list[Image.Image] = []          # 0..N query photos
    tools: list[str] = ["keyword", "semantic"]
    weights: dict[str, float] = {}
    filters: Filters = Filters()
    top_k: int = 8


class SearchEngine:
    def __init__(self, store: QdrantStore, text_embedder: TextEmbedder, image_embedder: ImageEmbedder,
                 catalog: dict[str, Boot], chunks: list[Chunk], candidate_k: int = 40, rrf_k: int = 60,
                 photos_per_boot: int = 3, image_aggregation: str = "max"):
        self.store = store
        self.text_embedder = text_embedder
        self.image_embedder = image_embedder
        self.catalog = catalog
        self.bm25 = BM25Index(chunks)
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.photos_per_boot = photos_per_boot
        self.image_aggregation = image_aggregation
        self._pool = ThreadPoolExecutor(max_workers=4)

    # ------------------------------------------------------------- individual tools
    def keyword(self, query: str, filters: Filters | None = None, k: int | None = None) -> list[RetrievedItem]:
        return self.bm25.search(query, k or self.candidate_k, filters, self.catalog)

    def semantic(self, query: str, filters: Filters | None = None, k: int | None = None) -> list[RetrievedItem]:
        vec = self.text_embedder.embed_query(query)
        # chunks -> boots aggregation shrinks the list, so over-fetch chunks
        return self.store.search_text(vec, filters, (k or self.candidate_k) * 3)

    def image(self, imgs: Image.Image | list[Image.Image], filters: Filters | None = None, k: int | None = None,
              exclude_paths: list[str] | None = None) -> list[RetrievedItem]:
        imgs = [imgs] if isinstance(imgs, Image.Image) else list(imgs)
        if not imgs:
            return []
        vectors = self.image_embedder.embed_images(imgs)            # one batch for all query photos
        per_query = [
            self.store.search_images(v, filters, k or self.candidate_k, photos_per_boot=self.photos_per_boot,
                                     aggregation=self.image_aggregation, exclude_paths=exclude_paths)
            for v in vectors
        ]
        return per_query[0] if len(per_query) == 1 else combine_query_images(per_query)

    def text_to_image(self, query: str, filters: Filters | None = None, k: int | None = None) -> list[RetrievedItem]:
        if not self.image_embedder.supports_text_to_image:
            return []
        vec = self.image_embedder.embed_text_for_image(query)
        return self.store.search_images(vec, filters, k or self.candidate_k, source="clip_text_to_image",
                                        photos_per_boot=self.photos_per_boot, aggregation=self.image_aggregation)

    def available_tools(self) -> list[str]:
        tools = ["keyword", "semantic", "image"]
        if self.image_embedder.supports_text_to_image:
            tools.append("text_to_image")
        return tools

    # ------------------------------------------------------------- orchestration
    def retrieve(self, plan: RetrievalPlan) -> tuple[dict[str, list[RetrievedItem]], dict[str, float]]:
        """Run the selected retrievers IN PARALLEL (they're I/O + CPU bound and independent).
        Returns (results per tool, latency ms per tool)."""
        jobs = {}
        for tool in plan.tools:
            if tool in ("keyword", "semantic", "text_to_image") and plan.query.strip():
                jobs[tool] = (getattr(self, tool), plan.query)
            elif tool == "image" and plan.images:
                jobs[tool] = (self.image, plan.images)

        def timed(fn, arg):
            t0 = time.perf_counter()
            out = fn(arg, plan.filters)
            return out, (time.perf_counter() - t0) * 1000

        futures = {tool: self._pool.submit(timed, fn, arg) for tool, (fn, arg) in jobs.items()}
        results, latencies = {}, {}
        for tool, fut in futures.items():
            results[tool], latencies[tool] = fut.result()
        return results, latencies

    def fuse(self, results: dict[str, list[RetrievedItem]], plan: RetrievalPlan) -> list[SearchResult]:
        fused = reciprocal_rank_fusion(results, plan.weights, self.rrf_k)
        out: list[SearchResult] = []
        for boot_id, score, sources, evidence, photo in fused:
            boot = self.catalog.get(boot_id)
            if boot is None or not plan.filters.matches(boot):   # final safety net for every retriever
                continue
            out.append(SearchResult(boot=boot, score=round(score, 5), sources=sources, evidence=evidence,
                                    matched_image=photo[0] if photo else None,
                                    matched_view=photo[1] if photo else None))
            if len(out) >= plan.top_k:
                break
        return out

    def run(self, plan: RetrievalPlan) -> list[SearchResult]:
        results, _ = self.retrieve(plan)
        return self.fuse(results, plan)


def combine_query_images(per_query: list[list[RetrievedItem]]) -> list[RetrievedItem]:
    """Average each boot's best-photo score over all query photos (symmetric MaxSim).

    A boot missing from one query photo's result list scored below that list's cut-off, so we
    use that list's lowest score as an upper-bound estimate instead of 0 (less harsh, still
    ranks it below boots that did make the list).
    """
    floors = [min((r.score for r in lst), default=0.0) for lst in per_query]
    boots = {r.boot_id for lst in per_query for r in lst}
    combined = []
    for bid in boots:
        scores, best = [], None
        for lst, floor in zip(per_query, floors):
            hit = next((r for r in lst if r.boot_id == bid), None)
            scores.append(hit.score if hit else floor)
            if hit and (best is None or hit.score > best.score):
                best = hit
        combined.append(RetrievedItem(
            boot_id=bid, score=sum(scores) / len(scores), source="image_vector",
            matched_image=best.matched_image, matched_view=best.matched_view,
            image_hits=sum(1 for lst in per_query if any(r.boot_id == bid for r in lst)),
        ))
    return sorted(combined, key=lambda r: r.score, reverse=True)
