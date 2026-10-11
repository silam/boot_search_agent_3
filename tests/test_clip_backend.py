"""Runs only when the clip extras are installed (pip install -r requirements/clip.txt).
Checks the real models load and that CLIP's shared space works for text -> image."""
import pytest

pytest.importorskip("open_clip")
pytest.importorskip("sentence_transformers")


def test_clip_embedders_and_cross_modal(settings):
    from PIL import Image

    from app.embeddings.clip import OpenClipImageEmbedder, SentenceTransformerTextEmbedder

    text = SentenceTransformerTextEmbedder(settings.text_model)
    a, b, c = text.embed_texts(["cowboy boots", "western pull-on boot with pointed toe", "rubber rain boot"])
    assert a @ b > a @ c          # real semantic similarity, not lexical overlap

    clip = OpenClipImageEmbedder(settings.clip_model, settings.clip_pretrained)
    img = clip.embed_images([Image.open(settings.raw_dir / "images" / "BOOT-001.png")])[0]
    assert img.shape == (clip.dim,)
    assert abs(float((clip.embed_text_for_image("a brown boot") ** 2).sum()) - 1) < 1e-4
