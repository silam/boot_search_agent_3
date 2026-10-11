"""Builds the configured embedders once per process (models are expensive to load)."""
from __future__ import annotations

from functools import lru_cache

from app.core.config import get_settings
from app.embeddings.base import ImageEmbedder, TextEmbedder


@lru_cache
def get_embedders() -> tuple[TextEmbedder, ImageEmbedder]:
    s = get_settings()
    if s.embedding_backend == "clip":
        from app.embeddings.clip import OpenClipImageEmbedder, SentenceTransformerTextEmbedder

        return SentenceTransformerTextEmbedder(s.text_model), OpenClipImageEmbedder(s.clip_model, s.clip_pretrained)

    from app.embeddings.lite import ColorShapeImageEmbedder, HashingTextEmbedder

    return HashingTextEmbedder(), ColorShapeImageEmbedder()
