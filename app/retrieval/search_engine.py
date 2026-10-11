"""
SearchEngine — the retrieval "tools" the agent can call, plus hybrid orchestration.

Retrievers (each returns product-level RetrievedItems):
  keyword(query)        -> BM25 over chunks                       (exact terms, SKUs, jargon)
  semantic(query)       -> text embedding vs text-chunk vectors   (meaning / paraphrase)
  image(img)            -> image embedding vs product images      (visual similarity)
  text_to_image(query)  -> CLIP text embedding vs product images  (cross-modal; clip backend only)

`run(plan)` executes the retrievers the agent selected, fuses them with weighted RRF,
applies hard filters, and hydrates full Boot objects.

INTERVIEW NOTE — combined "description + image" search:
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
    image: Image.Image | None = None
    tools: list[str] = ["keyword", "semantic"]
    weights: dict[str, float] = {}
    filters: Filters = Filters()
    top_k: int = 8


class SearchEngine:
    def __init__(self, store: QdrantStore, text_embedder: TextEmbedder, image_embedder: ImageEmbedder,
                 catalog: dict[str, Boot], chunks: list[Chunk], candidate_k: int = 40, rrf_k: int = 60):
        self.store = store
        self.text_embedder = text_embedder
        self.image_embedder = image_embedder
        self.catalog = catalog
        self.bm25 = BM25Index(chunks)
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self._pool = ThreadPoolExecutor(max_workers=4)

    # ------------------------------------------------------------- individual tools
    def keyword(self, query: str, filters: Filters | None = None, k: int | None = None) -> list[RetrievedItem]:
        return self.bm25.search(query, k or self.candidate_k, filters, self.catalog)

    def semantic(self, query: str, filters: Filters | None = None, k: int | None = None) -> list[RetrievedItem]:
        vec = self.text_embedder.embed_query(query)
        # chunks -> boots aggregation shrinks the list, so over-fetch chunks
        return self.store.search_text(vec, filters, (k or self.candidate_k) * 3)

    def image(self, img: Image.Image, filters: Filters | None = None, k: int | None = None) -> list[RetrievedItem]:
        vec = self.image_embedder.embed_images([img])[0]
        return self.store.search_images(vec, filters, k or self.candidate_k)

    def text_to_image(self, query: str, filters: Filters | None = None, k: int | None = None) -> list[RetrievedItem]:
        if not self.image_embedder.supports_text_to_image:
            return []
        vec = self.image_embedder.embed_text_for_image(query)
        return self.store.search_images(vec, filters, k or self.candidate_k, source="clip_text_to_image")

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
            elif tool == "image" and plan.image is not None:
                jobs[tool] = (self.image, plan.image)

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
        for boot_id, score, sources, evidence in fused:
            boot = self.catalog.get(boot_id)
            if boot is None or not plan.filters.matches(boot):   # final safety net for every retriever
                continue
            out.append(SearchResult(boot=boot, score=round(score, 5), sources=sources, evidence=evidence))
            if len(out) >= plan.top_k:
                break
        return out

    def run(self, plan: RetrievalPlan) -> list[SearchResult]:
        results, _ = self.retrieve(plan)
        return self.fuse(results, plan)
