"""
Batch ingestion job: ETL + embed + upsert into Qdrant.

  python scripts/ingest.py                         # uses QDRANT_URL from env/.env
  python scripts/ingest.py --source data/raw/catalog.json --recreate
  python scripts/ingest.py --source https://api.example.com/boots   # HTTP loader

In Docker Compose this runs as the one-shot `ingest` service before the API starts.
In Kubernetes it would be a Job (or CronJob for nightly refreshes).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.embeddings.factory import get_embedders  # noqa: E402
from app.ingestion.pipeline import index, transform  # noqa: E402
from app.retrieval.qdrant_store import QdrantStore  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=None, help="catalog JSON/CSV path or http(s) URL")
    ap.add_argument("--recreate", action="store_true", help="drop and rebuild the Qdrant collections")
    ap.add_argument("--transform-only", action="store_true", help="run ETL without embedding/indexing")
    args = ap.parse_args()

    s = get_settings()
    setup_logging(s.log_level)
    boots, chunks = transform(s, args.source)
    if args.transform_only:
        return
    if s.qdrant_url == ":memory:":
        print("QDRANT_URL is ':memory:' -> nothing to persist. The API indexes itself at startup.")
        return
    text_emb, img_emb = get_embedders()
    store = QdrantStore(s.qdrant_url, s.qdrant_api_key, s.text_collection, s.image_collection)
    index(s, store, text_emb, img_emb, boots, chunks, recreate=args.recreate)
    print(f"Indexed {len(boots)} boots / {len(chunks)} chunks into {s.qdrant_url}")


if __name__ == "__main__":
    main()
