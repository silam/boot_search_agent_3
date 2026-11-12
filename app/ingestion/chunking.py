"""
Chunking: split each product into retrievable text pieces.

Two strategies are combined — this is a good discussion talking point:

1. STRUCTURE-AWARE chunking (by section). A product has natural sections:
     - "profile":     a synthetic summary of the structured attributes
                      ("Northline Shift — work boot, olive, waterproof, steel toe, $244.99")
     - "description": marketing copy
     - "reviews":     customer voice ("kept my feet dry all day")
   Keeping sections separate means a review mention can't be diluted by marketing copy,
   and we can tell the user WHY a boot matched (the evidence snippet).

2. RECURSIVE CHARACTER chunking within a section, with overlap — the same algorithm as
   LangChain's RecursiveCharacterTextSplitter: try to split on paragraph, then sentence,
   then word boundaries, so chunks don't cut words or sentences in half. Overlap keeps
   context that would otherwise be lost at a boundary.

Why 320 chars? Product text is short and specific. Small chunks give precise matches;
the cost (more vectors) is trivial at catalog scale. For long documents (manuals,
policies) you'd use ~800-1200 tokens. The right size is something you TUNE with the
eval harness (app/evaluation), not guess.
"""
from __future__ import annotations

from app.core.models import Boot, Chunk

SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " "]


def recursive_split(text: str, chunk_size: int, overlap: int, separators: list[str] | None = None) -> list[str]:
    separators = SEPARATORS if separators is None else separators
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    # Pick the coarsest separator that actually occurs in the text
    sep = next((s for s in separators if s in text), "")
    if not sep:  # no separator left: hard cut
        step = max(1, chunk_size - overlap)
        return [text[i:i + chunk_size] for i in range(0, len(text), step)]

    pieces = [p + (sep if sep.strip() else " ") for p in text.split(sep)]
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if len(piece) > chunk_size:  # a single piece is too big -> recurse with finer separators
            if current:
                chunks.append(current.strip())
                current = ""
            finer = separators[separators.index(sep) + 1:]
            chunks.extend(recursive_split(piece, chunk_size, overlap, finer))
            continue
        if len(current) + len(piece) <= chunk_size:
            current += piece
        else:
            chunks.append(current.strip())
            tail = current[-overlap:] if overlap else ""
            # start the overlap at a word boundary
            tail = tail[tail.find(" ") + 1:] if " " in tail else tail
            current = tail + piece
    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if c]


def profile_text(boot: Boot) -> str:
    """Turn structured attributes into a sentence. Embedding models and BM25 both
    benefit from seeing attributes as natural language."""
    parts = [
        f"{boot.name} by {boot.brand} (SKU {boot.id})",
        f"{boot.category} boot",
        f"color {boot.color}",
        boot.material,
        "waterproof" if boot.waterproof else "water-resistant only",
        f"{boot.safety_toe} safety toe" if boot.safety_toe != "none" else "soft toe",
        f"price ${boot.price:.2f}",
        "features: " + ", ".join(boot.features),
    ]
    return ". ".join(parts)


def chunk_boot(boot: Boot, chunk_size: int = 320, overlap: int = 60) -> list[Chunk]:
    chunks: list[Chunk] = []
    sections = {
        "profile": profile_text(boot),
        "description": boot.description,
        "reviews": "\n".join(boot.reviews),
    }
    for section, text in sections.items():
        for i, piece in enumerate(recursive_split(text, chunk_size, overlap)):
            chunks.append(Chunk(chunk_id=f"{boot.id}:{section}:{i}", boot_id=boot.id, section=section, text=piece))
    return chunks
