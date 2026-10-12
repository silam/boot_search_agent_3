"""
Vector database layer — Qdrant.

Two collections:
  boot_text_chunks : one point per text chunk    (vector = text embedding)
  boot_images      : one point per product PHOTO (vector = image embedding). A boot with
                     9 photos (side, front, back, sole, top, close-ups...) is 9 points that
                     share the same `boot_id` payload.

Multi-image search uses Qdrant's GROUPING API (`query_points_groups`, group_by="boot_id"):
Qdrant returns the top boots, each with its best few matching photos, so one boot can't
take all top-k slots just because it has many similar photos. Each boot is then scored
from its matching photos:
  * "max"       -> MaxSim: the single best photo decides (default; a sole photo only has
                   to match the sole view, not every view)
  * "mean_top3" -> mean of the best 3 photos: rewards boots that match from several angles

Each point carries a PAYLOAD (the product attributes). Qdrant can filter on payload
*during* the HNSW graph search ("filterable HNSW"), which is better than filtering after:
post-filtering can return 0 results if all top-k neighbours fail the filter.

INTERVIEW NOTES
  * Why Qdrant? Open source (run it in Docker locally, managed Qdrant Cloud in prod),
    payload filtering with indexes, named/multi vectors, sparse vectors, quantization,
    written in Rust. Pinecone is managed-only; Chroma is great for prototypes but has
    weaker filtering/ops; Weaviate is comparable.
  * Distance = cosine. Vectors are already normalised, so cosine == dot product.
  * HNSW = approximate nearest neighbour graph: O(log n) search instead of a brute-force
    scan. Trade-off knobs: `m`, `ef_construct` (build), `hnsw_ef` (query): recall vs latency.
  * Point ids must be ints or UUIDs -> deterministic uuid5(chunk_id), so re-ingesting
    the same chunk UPSERTS (idempotent pipeline) instead of duplicating.
"""
from __future__ import annotations

import uuid
import warnings
from collections.abc import Sequence

import numpy as np
from qdrant_client import QdrantClient, models

from app.core.logging import get_logger
from app.core.models import Boot, Chunk, Filters, ProductImage, RetrievedItem

log = get_logger(__name__)
_NS = uuid.UUID("6f1c2b1e-3c3a-4f0e-9b4a-2d5c9b7f0a11")


def point_id(key: str) -> str:
    return str(uuid.uuid5(_NS, key))


def boot_payload(boot: Boot) -> dict:
    return {
        "boot_id": boot.id,
        "category": boot.category,
        "color": boot.color,
        "waterproof": boot.waterproof,
        "safety_toe": boot.safety_toe,
        "price": boot.price,
    }


def with_conditions(base: models.Filter | None, must: list | None = None,
                    must_not: list | None = None) -> models.Filter | None:
    """Add extra conditions (e.g. "only these views", "not this exact photo") to a filter."""
    must, must_not = list(must or []), list(must_not or [])
    if base is not None:
        must += list(base.must or [])
        must_not += list(base.must_not or [])
    if not must and not must_not:
        return None
    return models.Filter(must=must or None, must_not=must_not or None)


def to_qdrant_filter(f: Filters | None) -> models.Filter | None:
    """Translate our Filters model into a Qdrant payload filter (pushdown)."""
    if f is None or f.is_empty():
        return None
    must: list[models.Condition] = []
    must_not: list[models.Condition] = []
    for key in ("category", "color"):
        val = getattr(f, key)
        if val:
            must.append(models.FieldCondition(key=key, match=models.MatchValue(value=val)))
    if f.waterproof is not None:
        must.append(models.FieldCondition(key="waterproof", match=models.MatchValue(value=f.waterproof)))
    if f.safety_toe == "any":
        must_not.append(models.FieldCondition(key="safety_toe", match=models.MatchValue(value="none")))
    elif f.safety_toe:
        must.append(models.FieldCondition(key="safety_toe", match=models.MatchValue(value=f.safety_toe)))
    if f.max_price is not None or f.min_price is not None:
        must.append(models.FieldCondition(key="price", range=models.Range(lte=f.max_price, gte=f.min_price)))
    return models.Filter(must=must or None, must_not=must_not or None)


class QdrantStore:
    def __init__(self, url: str, api_key: str | None, text_collection: str, image_collection: str):
        if url == ":memory:" or url.startswith("path:"):
            # Local mode does exact search and ignores payload indexes/HNSW params; that's expected
            warnings.filterwarnings("ignore", message=".*(Payload indexes|Local mode).*")
        if url == ":memory:":
            self.client = QdrantClient(location=":memory:")
        elif url.startswith("path:"):
            self.client = QdrantClient(path=url.removeprefix("path:"))   # embedded, persisted to disk
        else:
            self.client = QdrantClient(url=url, api_key=api_key, timeout=30)
        self.text_collection = text_collection
        self.image_collection = image_collection

    # ------------------------------------------------------------------ schema
    def ensure_collections(self, text_dim: int, image_dim: int, recreate: bool = False) -> None:
        for name, dim in ((self.text_collection, text_dim), (self.image_collection, image_dim)):
            exists = self.client.collection_exists(name)
            if exists and recreate:
                self.client.delete_collection(name)
                exists = False
            if exists:
                current = self.client.get_collection(name).config.params.vectors.size
                if current != dim:  # embedding model changed -> old vectors are meaningless
                    log.warning(f"Recreating {name}: dim {current} -> {dim}")
                    self.client.delete_collection(name)
                    exists = False
            if not exists:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
                    hnsw_config=models.HnswConfigDiff(m=16, ef_construct=128),
                )
                # Payload indexes make filtered search fast (no effect in :memory: mode, required at scale)
                for field, schema in (("category", "keyword"), ("color", "keyword"), ("safety_toe", "keyword"),
                                      ("waterproof", "bool"), ("price", "float"), ("boot_id", "keyword"),
                                      ("view", "keyword"), ("path", "keyword")):
                    self.client.create_payload_index(name, field_name=field, field_schema=schema)

    def count(self, collection: str) -> int:
        if not self.client.collection_exists(collection):
            return 0
        return self.client.count(collection, exact=True).count

    # ------------------------------------------------------------------ writes
    def upsert_chunks(self, chunks: Sequence[Chunk], vectors: np.ndarray, boots: dict[str, Boot], batch: int = 256) -> None:
        points = [
            models.PointStruct(
                id=point_id(c.chunk_id),
                vector=v.tolist(),
                payload={**boot_payload(boots[c.boot_id]), "chunk_id": c.chunk_id, "section": c.section, "text": c.text},
            )
            for c, v in zip(chunks, vectors)
        ]
        for i in range(0, len(points), batch):
            self.client.upsert(self.text_collection, points=points[i:i + batch], wait=True)

    def upsert_images(self, pairs: Sequence[tuple[Boot, ProductImage]], vectors: np.ndarray) -> None:
        """One point per photo. The id is derived from (boot, photo path), so re-indexing
        overwrites instead of duplicating; the payload carries the boot's attributes (for
        filters) plus which photo and which view this is."""
        points = [
            models.PointStruct(
                id=point_id(f"img:{b.id}:{im.path}"),
                vector=v.tolist(),
                payload={**boot_payload(b), "path": im.path, "view": im.view,
                         "image_index": next(i for i, x in enumerate(b.images) if x.path == im.path)},
            )
            for (b, im), v in zip(pairs, vectors)
        ]
        self.client.upsert(self.image_collection, points=points, wait=True)

    # ------------------------------------------------------------------ reads
    def _query(self, collection: str, vector: np.ndarray, filters: Filters | None, limit: int):
        return self.client.query_points(
            collection_name=collection,
            query=vector.tolist(),
            query_filter=to_qdrant_filter(filters),
            limit=limit,
            with_payload=True,
            search_params=models.SearchParams(hnsw_ef=128),
        ).points

    def search_text(self, vector: np.ndarray, filters: Filters | None, limit: int) -> list[RetrievedItem]:
        """Chunk-level search, aggregated to product level with MAX score per boot."""
        best: dict[str, RetrievedItem] = {}
        for p in self._query(self.text_collection, vector, filters, limit):
            bid = p.payload["boot_id"]
            if bid not in best or p.score > best[bid].score:
                best[bid] = RetrievedItem(boot_id=bid, score=p.score, source="text_vector", evidence=p.payload.get("text"))
        return sorted(best.values(), key=lambda r: r.score, reverse=True)

    def search_images(self, vector: np.ndarray, filters: Filters | None, limit: int,
                      source: str = "image_vector", photos_per_boot: int = 3, aggregation: str = "max",
                      views: list[str] | None = None, exclude_paths: list[str] | None = None) -> list[RetrievedItem]:
        """Photo-level search grouped to boot level.

        limit           -> how many BOOTS to return (not photos)
        photos_per_boot -> how many of each boot's best photos Qdrant returns (group_size)
        aggregation     -> "max" (MaxSim) or "mean_top3"
        views           -> optional: only search these views (e.g. ["sole"])
        exclude_paths   -> optional: ignore these photos (used by the leave-one-out eval)
        """
        must = [models.FieldCondition(key="view", match=models.MatchAny(any=views))] if views else []
        must_not = ([models.FieldCondition(key="path", match=models.MatchAny(any=exclude_paths))]
                    if exclude_paths else [])
        result = self.client.query_points_groups(
            collection_name=self.image_collection,
            query=vector.tolist(),
            group_by="boot_id",
            limit=limit,
            group_size=photos_per_boot,
            query_filter=with_conditions(to_qdrant_filter(filters), must, must_not),
            with_payload=True,
            search_params=models.SearchParams(hnsw_ef=128),
        )
        items = []
        for group in result.groups:
            hits = sorted(group.hits, key=lambda h: h.score, reverse=True)
            scores = [h.score for h in hits]
            score = scores[0] if aggregation == "max" else sum(scores[:3]) / len(scores[:3])
            best = hits[0].payload
            items.append(RetrievedItem(
                boot_id=str(group.id), score=float(score), source=source,
                matched_image=best.get("path"), matched_view=best.get("view"), image_hits=len(hits),
            ))
        return sorted(items, key=lambda r: r.score, reverse=True)
