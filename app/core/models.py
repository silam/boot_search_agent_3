"""
Domain models shared by every layer (ingestion, retrieval, agent, API).

DISCUSSION NOTE:
  Pydantic models are the "contract" between modules. Validation at the edge
  (ingestion and API) means the inner layers can trust their inputs.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Category = Literal["work", "hiking", "western", "chelsea", "rain", "logger"]
SafetyToe = Literal["steel", "composite", "none"]
SearchMode = Literal["auto", "keyword", "semantic", "image", "hybrid"]


class ProductImage(BaseModel):
    """One product photo. A boot has many (side, front, back, sole, top, close-ups...)."""
    path: str                      # relative to data/raw/, e.g. images/BOOT-001/01_side_right.png
    view: str = "unknown"          # which angle the photo shows


class Boot(BaseModel):
    """A clean, validated catalog product."""
    id: str
    name: str = Field(min_length=1)
    brand: str
    category: Category
    color: str
    material: str
    price: float = Field(gt=0)
    waterproof: bool
    safety_toe: SafetyToe = "none"
    features: list[str] = []
    description: str
    reviews: list[str] = []
    images: list[ProductImage] = []

    @property
    def image(self) -> str | None:
        """Primary photo (the first one): used for thumbnails and as the default image."""
        return self.images[0].path if self.images else None


class Chunk(BaseModel):
    """A retrievable piece of text. Many chunks -> one boot (chunk-level retrieval,
    product-level results)."""
    chunk_id: str
    boot_id: str
    section: Literal["profile", "description", "reviews"]
    text: str


class Filters(BaseModel):
    """Structured constraints extracted from the query (by the LLM or the rule parser)."""
    category: Category | None = None
    color: str | None = None
    waterproof: bool | None = None
    safety_toe: SafetyToe | Literal["any"] | None = None   # "any" = must have some safety toe
    max_price: float | None = None
    min_price: float | None = None

    @field_validator("color")
    @classmethod
    def lower(cls, v: str | None) -> str | None:
        return v.lower() if v else v

    def is_empty(self) -> bool:
        return all(v is None for v in self.model_dump().values())

    def matches(self, boot: Boot) -> bool:
        if self.category and boot.category != self.category:
            return False
        if self.color and boot.color != self.color:
            return False
        if self.waterproof is not None and boot.waterproof != self.waterproof:
            return False
        if self.safety_toe == "any" and boot.safety_toe == "none":
            return False
        if self.safety_toe in ("steel", "composite", "none") and boot.safety_toe != self.safety_toe:
            return False
        if self.max_price is not None and boot.price > self.max_price:
            return False
        if self.min_price is not None and boot.price < self.min_price:
            return False
        return True


class RetrievedItem(BaseModel):
    """One boot returned by one retriever, with the evidence that matched."""
    boot_id: str
    score: float
    source: Literal["bm25", "text_vector", "image_vector", "clip_text_to_image"]
    evidence: str | None = None
    # image retrievers only: which of the boot's photos matched best, and how many matched
    matched_image: str | None = None
    matched_view: str | None = None
    image_hits: int | None = None


class SearchResult(BaseModel):
    """Final fused result returned to the client."""
    boot: Boot
    score: float
    sources: list[str]
    evidence: str | None = None
    matched_image: str | None = None   # best-matching photo when an image retriever found this boot
    matched_view: str | None = None
