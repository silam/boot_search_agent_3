"""
"lite" embedders — no PyTorch, no downloads, deterministic, a few MB.

Used for: unit tests, GitHub Actions CI, and the smallest serverless deployment.

HashingTextEmbedder
  The "hashing trick": every token / bigram / character trigram is hashed into one of
  `dim` buckets and counted (sublinear tf), then L2-normalised. Character trigrams give
  typo tolerance ("waterprof" still overlaps "waterproof"). It is LEXICAL, not semantic:
  "cowboy" will not match "western". Be upfront about that in the discussion — it's why
  the production backend is CLIP / sentence-transformers.

ColorShapeImageEmbedder
  A hand-crafted image descriptor: a coarse RGB colour histogram of the foreground plus a
  16x16 silhouette mask. Two boots with the same colour and silhouette land close
  together. This is classic content-based image retrieval (pre-deep-learning) — a nice
  baseline to compare CLIP against.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np
from PIL import Image

from app.embeddings.base import ImageEmbedder, TextEmbedder, l2_normalize

_TOKEN = re.compile(r"[a-z0-9]+")


def _bucket(feature: str, dim: int) -> tuple[int, float]:
    h = int(hashlib.md5(feature.encode()).hexdigest(), 16)
    return h % dim, (1.0 if (h >> 64) & 1 else -1.0)  # signed hashing reduces collision bias


class HashingTextEmbedder(TextEmbedder):
    name = "hashing-lite"

    def __init__(self, dim: int = 768):
        self.dim = dim

    def _features(self, text: str) -> list[tuple[str, float]]:
        toks = _TOKEN.findall(text.lower())
        feats: list[tuple[str, float]] = [(f"w:{t}", 1.0) for t in toks]
        feats += [(f"b:{a}_{b}", 0.7) for a, b in zip(toks, toks[1:])]
        for t in toks:
            padded = f"#{t}#"
            feats += [(f"c:{padded[i:i + 3]}", 0.25) for i in range(len(padded) - 2)]
        return feats

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[int, float] = {}
            for feat, weight in self._features(text):
                idx, sign = _bucket(feat, self.dim)
                counts[idx] = counts.get(idx, 0.0) + sign * weight
            for idx, v in counts.items():
                out[row, idx] = np.sign(v) * np.log1p(abs(v))  # sublinear tf
        return l2_normalize(out)


class ColorShapeImageEmbedder(ImageEmbedder):
    name = "color-shape-lite"
    supports_text_to_image = False

    BINS = 6            # per RGB channel -> 216 colour bins
    MASK = 16           # 16x16 silhouette -> 256 dims

    def __init__(self):
        self.dim = self.BINS ** 3 + self.MASK * self.MASK

    def _embed_one(self, img: Image.Image) -> np.ndarray:
        img = img.convert("RGB").resize((128, 128))
        arr = np.asarray(img, dtype=np.int32)
        # Foreground = anything that isn't near-white background
        fg = (arr.min(axis=-1) < 225)
        q = (arr // (256 // self.BINS)).clip(0, self.BINS - 1)
        codes = q[..., 0] * self.BINS * self.BINS + q[..., 1] * self.BINS + q[..., 2]
        hist = np.bincount(codes[fg], minlength=self.BINS ** 3).astype(np.float32)
        hist = hist / max(hist.sum(), 1.0)
        mask = Image.fromarray((fg * 255).astype(np.uint8)).resize((self.MASK, self.MASK))
        mask_v = np.asarray(mask, dtype=np.float32).ravel() / 255.0
        # Weight colour vs shape; both blocks normalised first so neither dominates by scale
        color_part = l2_normalize(np.sqrt(hist)) * 0.6
        shape_part = l2_normalize(mask_v) * 0.4
        return np.concatenate([color_part, shape_part])

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        return l2_normalize(np.stack([self._embed_one(im) for im in images]))
