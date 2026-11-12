"""
Embedding interfaces.

DISCUSSION NOTE — why an interface?
  The retrieval code depends on `TextEmbedder` / `ImageEmbedder`, never on a concrete
  model. That gives us:
    * a CPU-only "lite" backend for unit tests, CI and small serverless containers,
    * a "clip" backend (sentence-transformers + OpenCLIP) for real semantic quality,
    * a one-line swap to a hosted API (OpenAI, Voyage, Bedrock) later.
  This is the Strategy pattern / dependency inversion.

Vector basics worth saying out loud:
  * An embedding maps text or an image to a point in R^d where "similar meaning" ==
    "nearby point". We L2-normalise every vector so dot product == cosine similarity.
  * Text and image vectors are only comparable if the SAME model produced both
    (CLIP's shared space). That's why text->image search needs `embed_text_for_image`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from PIL import Image


def l2_normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.clip(norms, 1e-12, None)


class TextEmbedder(ABC):
    name: str
    dim: int

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed documents. Returns (n, dim) L2-normalised."""

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a query. Some models (e5, bge) want a different prefix for queries."""
        return self.embed_texts([text])[0]


class ImageEmbedder(ABC):
    name: str
    dim: int
    supports_text_to_image: bool = False

    @abstractmethod
    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        """Returns (n, dim) L2-normalised."""

    def embed_text_for_image(self, text: str) -> np.ndarray:
        """Embed text INTO the image space (cross-modal). Only CLIP-style models can."""
        raise NotImplementedError(f"{self.name} has no shared text-image space")
