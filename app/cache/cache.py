"""
Response cache — Redis in production, an in-process TTL dict as fallback.

INTERVIEW NOTES
  * What we cache: the full search response, keyed by a hash of everything that affects
    it (query, mode, filters, top_k, prompt variant, image bytes hash, index version).
    A hit skips embedding, retrieval AND the LLM — the biggest cost and latency win.
  * Why Redis (vs in-process)? Shared across replicas/pods/serverless instances, survives
    restarts, TTL eviction built in. The fallback keeps local dev dependency-free.
  * Invalidation: the key includes `index_version`, so re-indexing the catalog naturally
    invalidates old entries ("version the key, don't delete the cache").
  * Next step: SEMANTIC caching — embed the query and reuse answers for near-duplicate
    queries (cosine > 0.95). Higher hit rate, but you must guard against wrong reuse.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from abc import ABC, abstractmethod
from typing import Any

from app.core.logging import get_logger

log = get_logger(__name__)


def make_key(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return "search:" + hashlib.sha256(blob).hexdigest()[:32]


class Cache(ABC):
    backend: str

    @abstractmethod
    def get(self, key: str) -> dict | None: ...

    @abstractmethod
    def set(self, key: str, value: dict, ttl_s: int) -> None: ...


class MemoryCache(Cache):
    backend = "memory"

    def __init__(self, max_items: int = 1000):
        self._data: dict[str, tuple[float, dict]] = {}
        self._lock = threading.Lock()
        self.max_items = max_items

    def get(self, key: str) -> dict | None:
        with self._lock:
            item = self._data.get(key)
            if not item:
                return None
            expires, value = item
            if expires < time.time():
                del self._data[key]
                return None
            return value

    def set(self, key: str, value: dict, ttl_s: int) -> None:
        with self._lock:
            if len(self._data) >= self.max_items:      # crude eviction: drop the oldest-expiring item
                self._data.pop(min(self._data, key=lambda k: self._data[k][0]))
            self._data[key] = (time.time() + ttl_s, value)


class RedisCache(Cache):
    backend = "redis"

    def __init__(self, url: str):
        import redis

        self.client = redis.Redis.from_url(url, socket_timeout=0.5, socket_connect_timeout=0.5)
        self.client.ping()

    def get(self, key: str) -> dict | None:
        try:
            raw = self.client.get(key)
            return json.loads(raw) if raw else None
        except Exception as e:  # cache must never break search
            log.warning(f"redis get failed: {e}")
            return None

    def set(self, key: str, value: dict, ttl_s: int) -> None:
        try:
            self.client.setex(key, ttl_s, json.dumps(value, default=str))
        except Exception as e:
            log.warning(f"redis set failed: {e}")


def build_cache(redis_url: str | None) -> Cache:
    if redis_url:
        try:
            return RedisCache(redis_url)
        except Exception as e:
            log.warning(f"Redis unavailable ({e}); falling back to in-memory cache")
    return MemoryCache()
