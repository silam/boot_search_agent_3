"""
ETL pipeline:  EXTRACT (loaders) -> TRANSFORM (clean, validate, chunk) -> LOAD (files + Qdrant)

  data/raw/catalog.json  ──►  data/processed/boots.jsonl      (clean products)
                              data/processed/chunks.jsonl     (retrievable text)
                              data/processed/rejected.json    (data-quality report)
                              data/processed/manifest.json    (lineage: hashes, counts, model)
                         ──►  Qdrant: boot_text_chunks, boot_images

INTERVIEW NOTES
  * ETL vs ELT: here we transform BEFORE loading into the vector DB (ETL) because the
    vector DB is not a place to clean data. In a warehouse world (Snowflake/BigQuery)
    you'd land raw data first and transform with SQL/dbt (ELT).
  * Idempotent: deterministic point ids -> re-running upserts, never duplicates.
  * Lineage: manifest.json records the SHA-256 of the raw input and the embedding model.
    If either changes, the index must be rebuilt. DVC (dvc.yaml) uses the same idea to
    version the data and cache pipeline stages.
  * Batch embedding: embed in batches (GPU-friendly), never one call per chunk.
  * Streaming variant: for real-time catalog updates, a Kafka/Kinesis consumer would call
    `index_boots([boot])` per change event instead of a nightly batch run.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from PIL import Image

from app.core.config import Settings
from app.core.logging import get_logger, log_event
from app.core.models import Boot, Chunk
from app.embeddings.base import ImageEmbedder, TextEmbedder
from app.ingestion.chunking import chunk_boot
from app.ingestion.cleaning import clean_catalog
from app.ingestion.loaders import loader_for
from app.retrieval.qdrant_store import QdrantStore

log = get_logger(__name__)

# Bump when cleaning/chunking LOGIC changes, so deployed indexes know they are stale
PIPELINE_VERSION = "1.0.0"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def transform(settings: Settings, source: str | Path | None = None) -> tuple[list[Boot], list[Chunk]]:
    """Extract + Transform, and persist processed artifacts. No embeddings needed."""
    source = source or settings.raw_dir / "catalog.json"
    t0 = time.perf_counter()
    raw = loader_for(source).load()
    boots, rejected = clean_catalog(raw)
    chunks = [c for b in boots for c in chunk_boot(b, settings.chunk_size, settings.chunk_overlap)]

    out = settings.processed_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "boots.jsonl").write_text("\n".join(b.model_dump_json() for b in boots) + "\n")
    (out / "chunks.jsonl").write_text("\n".join(c.model_dump_json() for c in chunks) + "\n")
    (out / "rejected.json").write_text(json.dumps(rejected, indent=2))
    (out / "manifest.json").write_text(json.dumps({
        "pipeline_version": PIPELINE_VERSION,
        "source": str(source),
        "source_sha256": _sha256(Path(source)) if not str(source).startswith("http") else None,
        "raw_records": len(raw),
        "clean_boots": len(boots),
        "rejected": len(rejected),
        "chunks": len(chunks),
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2))
    log_event(log, "transform_done", raw=len(raw), clean=len(boots), rejected=len(rejected),
              chunks=len(chunks), ms=round((time.perf_counter() - t0) * 1000))
    return boots, chunks


def processed_is_stale(settings: Settings) -> bool:
    """True if the raw data, chunking params or pipeline code changed since the last ETL run."""
    manifest_path = settings.processed_dir / "manifest.json"
    if not (settings.processed_dir / "boots.jsonl").exists() or not manifest_path.exists():
        return True
    m = json.loads(manifest_path.read_text())
    raw = settings.raw_dir / "catalog.json"
    return (m.get("pipeline_version") != PIPELINE_VERSION
            or m.get("chunk_size") != settings.chunk_size
            or m.get("chunk_overlap") != settings.chunk_overlap
            or (raw.exists() and m.get("source_sha256") not in (None, _sha256(raw))))


def load_processed(settings: Settings) -> tuple[list[Boot], list[Chunk]]:
    p = settings.processed_dir
    boots = [Boot.model_validate_json(line) for line in (p / "boots.jsonl").read_text().splitlines() if line]
    chunks = [Chunk.model_validate_json(line) for line in (p / "chunks.jsonl").read_text().splitlines() if line]
    return boots, chunks


def index(settings: Settings, store: QdrantStore, text_emb: TextEmbedder, img_emb: ImageEmbedder,
          boots: list[Boot], chunks: list[Chunk], recreate: bool = False) -> None:
    """LOAD step: embed and upsert into Qdrant."""
    t0 = time.perf_counter()
    store.ensure_collections(text_emb.dim, img_emb.dim, recreate=recreate)
    catalog = {b.id: b for b in boots}

    vectors = text_emb.embed_texts([c.text for c in chunks])
    store.upsert_chunks(chunks, vectors, catalog)

    with_images = [b for b in boots if b.image and (settings.raw_dir / b.image).exists()]
    for i in range(0, len(with_images), 32):
        batch = with_images[i:i + 32]
        imgs = [Image.open(settings.raw_dir / b.image) for b in batch]
        store.upsert_images(batch, img_emb.embed_images(imgs))

    manifest_path = settings.processed_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest.update({"text_embedder": text_emb.name, "image_embedder": img_emb.name,
                         "text_dim": text_emb.dim, "image_dim": img_emb.dim})
        manifest_path.write_text(json.dumps(manifest, indent=2))
    log_event(log, "index_done", chunks=len(chunks), images=len(with_images),
              text_embedder=text_emb.name, image_embedder=img_emb.name,
              ms=round((time.perf_counter() - t0) * 1000))
