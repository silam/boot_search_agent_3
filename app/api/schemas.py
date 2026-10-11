"""Request/response contracts for the REST API (they also generate the OpenAPI docs at /docs)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.core.models import Boot, Filters, SearchMode


class SearchRequest(BaseModel):
    query: str = Field("", max_length=2000, examples=["waterproof hiking boots under $200"])
    image_base64: str | None = Field(None, description="Optional reference photo (PNG/JPEG/WEBP), base64-encoded")
    mode: SearchMode = "auto"
    filters: Filters = Filters()
    top_k: int = Field(8, ge=1, le=24)
    session_id: str | None = Field(None, description="Stable per-user session id (drives A/B variant)")
    use_cache: bool = True


class ResultItem(BaseModel):
    boot: Boot
    score: float
    sources: list[str]
    evidence: str | None = None
    image_url: str | None = None


class SearchResponse(BaseModel):
    request_id: str
    answer: str
    results: list[ResultItem]
    clarifying_question: str | None = None
    filters_applied: dict
    relaxed_filters: list[str] = []
    tools_used: list[str]
    understood_by: str
    grounded: bool = True
    variant: str
    latency_ms: float
    cache_hit: bool = False
    llm: dict = {}
    trace: list[dict] = []


class FeedbackRequest(BaseModel):
    request_id: str
    rating: Literal[-1, 1] = Field(..., description="+1 thumbs up, -1 thumbs down")
    boot_id: str | None = None
    comment: str | None = Field(None, max_length=1000)
