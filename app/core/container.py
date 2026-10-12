"""
Composition root: builds the long-lived objects ONCE per process and wires them together.

INTERVIEW NOTE:
  Startup is "self-healing": if the processed data is missing it runs the ETL; if the
  Qdrant collections are empty (fresh container, in-memory mode, new embedding model)
  it re-indexes. That makes stateless deployments (Vercel, Kubernetes pods) safe —
  any new replica can come up on its own. In a big system indexing would be a separate
  batch job and the API would only read.
"""
from __future__ import annotations

from functools import lru_cache

from app.core.config import get_settings
from app.core.logging import get_logger
from app.embeddings.factory import get_embedders
from app.ingestion.pipeline import index, load_processed, processed_is_stale, transform
from app.retrieval.qdrant_store import QdrantStore
from app.retrieval.search_engine import SearchEngine

log = get_logger(__name__)


@lru_cache
def get_engine() -> SearchEngine:
    s = get_settings()
    text_emb, img_emb = get_embedders()
    store = QdrantStore(s.qdrant_url, s.qdrant_api_key, s.text_collection, s.image_collection)

    reprocessed = processed_is_stale(s)
    if reprocessed:
        log.info("Processed data missing or stale -> running ETL")
        boots, chunks = transform(s)
    else:
        boots, chunks = load_processed(s)

    n_images = sum(len(b.images) for b in boots)       # one Qdrant point per photo
    needs_index = (
        reprocessed
        or store.count(s.text_collection) != len(chunks)
        or store.count(s.image_collection) != n_images
    )
    if needs_index:
        log.info("Vector index empty or stale -> indexing")
        index(s, store, text_emb, img_emb, boots, chunks, recreate=reprocessed)
    else:
        # also catches an embedding-model change (dimension mismatch -> recreate)
        store.ensure_collections(text_emb.dim, img_emb.dim)
        if store.count(s.text_collection) == 0:
            index(s, store, text_emb, img_emb, boots, chunks)

    return SearchEngine(store, text_emb, img_emb, {b.id: b for b in boots}, chunks,
                        candidate_k=s.candidate_k, rrf_k=s.rrf_k,
                        photos_per_boot=s.image_photos_per_boot, image_aggregation=s.image_aggregation)
