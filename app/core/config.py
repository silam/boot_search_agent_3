"""
Central configuration (12-factor style): every setting comes from environment variables
or a .env file, so the SAME Docker image runs locally, in CI, and on Vercel.

INTERVIEW NOTE:
  "Config is code-free" — switching from local Qdrant to Qdrant Cloud, or from the
  lightweight embedder to CLIP, is an env-var change, not a code change.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---- app
    app_name: str = "Boot Search Agent"
    environment: Literal["local", "ci", "production"] = "local"
    log_level: str = "INFO"

    # ---- data
    data_dir: Path = PROJECT_ROOT / "data"

    # ---- embeddings
    # "lite": hashing text vectors + color/shape image vectors (no torch, tiny, CI/serverless friendly)
    # "clip": sentence-transformers for text + OpenCLIP for images and cross-modal text->image
    embedding_backend: Literal["lite", "clip"] = "lite"
    text_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"

    # ---- vector DB (Qdrant)
    # ":memory:" -> embedded, in-process Qdrant (tests / quick demo)
    # "http://qdrant:6333" -> Docker Compose service; "https://xxx.cloud.qdrant.io" -> Qdrant Cloud
    qdrant_url: str = ":memory:"
    qdrant_api_key: str | None = None
    text_collection: str = "boot_text_chunks"
    image_collection: str = "boot_images"

    # ---- chunking
    chunk_size: int = 320        # characters
    chunk_overlap: int = 60

    # ---- retrieval
    top_k: int = 8
    candidate_k: int = 40        # how many chunks each retriever returns before fusion
    rrf_k: int = 60              # Reciprocal Rank Fusion constant (60 is the value from the original paper)

    # ---- LLM (Anthropic). No key -> deterministic rule-based fallback, so demos never break.
    anthropic_api_key: str | None = None
    llm_model: str = "claude-haiku-5-5"
    llm_max_tokens: int = 600
    llm_timeout_s: float = 20.0

    # ---- cache
    redis_url: str | None = None  # e.g. redis://redis:6379/0 ; None -> in-process TTL cache
    cache_ttl_s: int = 600

    # ---- SQL store for feedback + query logs (human-in-the-loop, continuous improvement)
    database_url: str = f"sqlite:///{PROJECT_ROOT / 'data' / 'app.db'}"

    # ---- API security
    api_key: str | None = None          # if set, clients must send X-API-Key
    rate_limit_per_minute: int = 60     # per client IP (in-process; use Redis/API gateway at scale)
    cors_origins: str = "*"             # comma-separated list in production

    # ---- guardrails
    max_query_chars: int = 500
    max_image_bytes: int = 5 * 1024 * 1024

    # ---- cost tracking (USD per 1M tokens) — edit to match your provider's price sheet
    llm_input_cost_per_mtok: float = 1.0
    llm_output_cost_per_mtok: float = 5.0

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"


@lru_cache
def get_settings() -> Settings:
    return Settings()
