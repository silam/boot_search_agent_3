"""
Production embedders (Hugging Face ecosystem).

  * SentenceTransformerTextEmbedder — `all-MiniLM-L6-v2` (384-d) from the Hugging Face Hub.
    Real semantic similarity: "cowboy boots" ~ "western pull-on boot with pointed toe".
  * OpenClipImageEmbedder — OpenCLIP ViT-B/32 (512-d). Image and text share one space, so
    it powers BOTH image->image similarity AND text->image search ("tan boot with red
    laces" against product photos, with no captions needed).

Why two text models? MiniLM is better at sentence-level semantics of long product copy;
CLIP's text tower is trained on short captions (77 token limit) and is best used only to
search the IMAGE space. Using each model for what it was trained on is a deliberate
design decision.

Heavy imports (torch, open_clip) happen lazily inside __init__, so the lite backend and
the test suite never need PyTorch installed.

Upgrade path: ViT-L/14 for better accuracy (768-d, ~4x slower), or serve the model
behind a dedicated GPU inference server (TGI / Triton / vLLM for generative models) and
call it over HTTP — the interface stays the same.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from app.embeddings.base import ImageEmbedder, TextEmbedder, l2_normalize


class SentenceTransformerTextEmbedder(TextEmbedder):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.name = model_name
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = int(get_dim())

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


class OpenClipImageEmbedder(ImageEmbedder):
    supports_text_to_image = True

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k", device: str | None = None):
        import open_clip
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.name = f"openclip-{model_name}-{pretrained}"
        with torch.no_grad():
            self.dim = int(self.model.encode_text(self.tokenizer(["probe"]).to(self.device)).shape[-1])

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        batch = self.torch.stack([self.preprocess(im.convert("RGB")) for im in images]).to(self.device)
        with self.torch.no_grad():
            feats = self.model.encode_image(batch)
        return l2_normalize(feats.cpu().numpy())

    def embed_text_for_image(self, text: str) -> np.ndarray:
        # Prompt template: CLIP was trained on captions, so phrasing the query like a caption helps
        tokens = self.tokenizer([f"a product photo of {text}"]).to(self.device)
        with self.torch.no_grad():
            feats = self.model.encode_text(tokens)
        return l2_normalize(feats.cpu().numpy())[0]
