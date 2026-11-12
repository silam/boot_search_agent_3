"""
FastAPI application — the public API of the Boot Search Agent.

Endpoints
  GET  /health                liveness probe (process is up)
  GET  /ready                 readiness probe (index loaded, counts)
  POST /search                agentic multimodal search (text and/or base64 image)
  GET  /boots/{id}            product details
  GET  /boots/{id}/image      primary product photo
  GET  /boots/{id}/images     the boot's photo gallery (8-10 views)
  GET  /boots/{id}/images/{i} one photo
  POST /feedback              thumbs up/down on a search            (human-in-the-loop)
  GET  /feedback/review       negative-feedback review queue        (human-in-the-loop)
  GET  /feedback/stats        A/B results per prompt variant
  GET  /agent/graph           Mermaid diagram of the LangGraph agent
  GET  /metrics               Prometheus metrics
  GET  /docs                  OpenAPI / Swagger UI

Every route is served both at "/" and under "/api", so the same container works behind
a path-prefix router (Vercel Services rewrites `/api/*` to this service).

DISCUSSION NOTES
  * Sync endpoints (`def`, not `async def`) on purpose: embedding + Qdrant local mode are
    CPU-bound/blocking, so FastAPI runs them in its threadpool instead of blocking the
    event loop.
  * Heavy objects (models, index) load ONCE in the lifespan hook, not per request.
  * Middleware adds a request id (propagated into logs and the response header),
    measures latency, and records Prometheus metrics.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app import __version__
from app.agent.graph import BootSearchAgent
from app.agent.llm import LLMClient
from app.agent.prompts import PROMPT_VERSION
from app.api import metrics as m
from app.api.schemas import BootImage, FeedbackRequest, ResultItem, SearchRequest, SearchResponse
from app.cache.cache import build_cache, make_key
from app.core.config import get_settings
from app.core.container import get_engine
from app.core.logging import get_logger, log_event, request_id_var, setup_logging
from app.guardrails.input_guard import load_image
from app.storage.feedback import FeedbackStore

settings = get_settings()
setup_logging(settings.log_level)
log = get_logger("api")


class AppState:
    agent: BootSearchAgent
    cache = None
    store: FeedbackStore


state = AppState()


@asynccontextmanager
async def lifespan(_: FastAPI):
    t0 = time.perf_counter()
    engine = get_engine()
    state.agent = BootSearchAgent(engine, LLMClient(settings), settings)
    state.cache = build_cache(settings.redis_url)
    state.store = FeedbackStore(settings.database_url)
    log_event(log, "startup_complete", boots=len(engine.catalog), tools=engine.available_tools(),
              llm=state.agent.llm.available, cache=state.cache.backend,
              ms=round((time.perf_counter() - t0) * 1000))
    yield


app = FastAPI(title=settings.app_name, version=__version__, lifespan=lifespan,
              description="Agentic multimodal boot search: BM25 + dense vectors + image similarity, fused with RRF.")
app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
                   allow_methods=["*"], allow_headers=["*"])


# ------------------------------------------------------------------ middleware
@app.middleware("http")
async def observe(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
    request_id_var.set(rid)
    t0 = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
    finally:
        route = request.scope.get("route")
        path = route.path if route else "unmatched"     # route template -> low-cardinality labels
        elapsed = time.perf_counter() - t0
        m.HTTP_REQUESTS.labels(request.method, path, str(status)).inc()
        m.HTTP_LATENCY.labels(request.method, path).observe(elapsed)
    response.headers["x-request-id"] = rid
    response.headers["x-response-time-ms"] = f"{elapsed * 1000:.1f}"
    return response


# ------------------------------------------------------------------ dependencies
_hits: dict[str, deque] = defaultdict(deque)


def security(request: Request) -> None:
    """Optional API key + per-IP sliding-window rate limit."""
    if settings.api_key and request.headers.get("x-api-key") != settings.api_key:
        raise HTTPException(401, "Invalid or missing X-API-Key")
    ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "?").split(",")[0]
    now, window = time.time(), _hits[ip]
    while window and window[0] < now - 60:
        window.popleft()
    if len(window) >= settings.rate_limit_per_minute:
        raise HTTPException(429, "Rate limit exceeded, slow down")
    window.append(now)


def ab_variant(session_id: str | None) -> str:
    """Stable 50/50 split: the same session always sees the same prompt variant."""
    if not session_id:
        return "A"
    return "A" if int(hashlib.md5(session_id.encode()).hexdigest(), 16) % 2 == 0 else "B"


router = APIRouter()


# ------------------------------------------------------------------ health
@router.get("/health", tags=["ops"])
def health():
    return {"status": "ok", "version": __version__}


@router.get("/ready", tags=["ops"])
def ready():
    engine = state.agent.engine
    return {
        "status": "ready",
        "boots": len(engine.catalog),
        "text_chunks": engine.store.count(engine.store.text_collection),
        "images": engine.store.count(engine.store.image_collection),
        "embedders": {"text": engine.text_embedder.name, "image": engine.image_embedder.name},
        "tools": engine.available_tools(),
        "llm": state.agent.llm.model if state.agent.llm.available else "disabled (rule-based fallback)",
        "cache": state.cache.backend,
        "prompt_version": PROMPT_VERSION,
    }


# ------------------------------------------------------------------ search
@router.post("/search", response_model=SearchResponse, tags=["search"], dependencies=[Depends(security)])
def search(req: SearchRequest, request: Request):
    t0 = time.perf_counter()
    rid = request_id_var.get()
    variant = ab_variant(req.session_id)

    images, image_hashes = [], []
    encoded = ([req.image_base64] if req.image_base64 else []) + list(req.images_base64)
    if len(encoded) > settings.max_query_images:
        raise HTTPException(422, f"Too many photos: send at most {settings.max_query_images}")
    for n, item in enumerate(encoded, start=1):
        try:
            raw = base64.b64decode(item.split(",")[-1], validate=True)
            images.append(load_image(raw, settings.max_image_bytes))
        except (ValueError, binascii.Error) as e:
            raise HTTPException(422, f"Bad image #{n}: {e}") from e
        image_hashes.append(hashlib.sha256(raw).hexdigest())
    if not req.query.strip() and not images:
        raise HTTPException(422, "Provide a query, an image, or both")
    if req.mode == "image" and not images:
        raise HTTPException(422, "mode=image requires an image")

    # ---------------- cache lookup
    key = make_key({"q": req.query, "mode": req.mode, "f": req.filters.model_dump(), "k": req.top_k,
                    "v": variant, "img": image_hashes, "pv": PROMPT_VERSION, "n": len(state.agent.engine.catalog)})
    if req.use_cache and (cached := state.cache.get(key)):
        m.CACHE_HITS.inc()
        cached = dict(cached)   # never mutate the cached object (in-memory backend shares it)
        cached.update(request_id=rid, cache_hit=True, latency_ms=round((time.perf_counter() - t0) * 1000, 2))
        _log(rid, req, cached, variant)
        return cached
    m.CACHE_MISSES.inc()

    # ---------------- run the agent
    out = state.agent.run(query=req.query, images=images, mode=req.mode, user_filters=req.filters,
                          top_k=req.top_k, variant=variant)
    usage = out.get("usage")
    results = out.get("results", [])

    # ---------------- metrics
    for step in out.get("trace", []):
        m.NODE_LATENCY.labels(step["node"]).observe(step["ms"] / 1000)
    if out.get("blocked"):
        m.GUARDRAIL_BLOCKS.inc()
    elif not results:
        m.ZERO_RESULTS.inc()
    if out.get("grounded") is False:
        m.GROUNDING_FAILURES.inc()
    m.SEARCHES.labels(req.mode, out.get("understood_by", "blocked")).inc()
    if usage:
        m.LLM_TOKENS.labels("input").inc(usage.input_tokens)
        m.LLM_TOKENS.labels("output").inc(usage.output_tokens)
        m.LLM_COST.inc(usage.cost_usd)
        for err in usage.errors:
            m.LLM_ERRORS.labels(err).inc()

    body = SearchResponse(
        request_id=rid,
        answer=out.get("answer", ""),
        results=[ResultItem(boot=r.boot, score=r.score, sources=r.sources, evidence=r.evidence,
                            image_url=_image_url(r.boot, r.matched_image), matched_view=r.matched_view,
                            image_urls=[f"/boots/{r.boot.id}/images/{i}" for i in range(len(r.boot.images))])
                 for r in results],
        clarifying_question=out.get("clarifying_question"),
        filters_applied=out["filters"].model_dump(exclude_none=True) if out.get("filters") else {},
        relaxed_filters=out.get("relaxed", []),
        tools_used=out.get("tools", []),
        understood_by=out.get("understood_by", "blocked"),
        grounded=out.get("grounded", True),
        variant=variant,
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        llm={"calls": usage.calls, "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
             "cost_usd": round(usage.cost_usd, 6), "latency_ms": round(usage.latency_ms, 1)} if usage else {},
        trace=out.get("trace", []),
    ).model_dump(mode="json")

    if req.use_cache and not out.get("blocked"):
        state.cache.set(key, body, settings.cache_ttl_s)
    _log(rid, req, body, variant)
    return body


def _log(rid: str, req: SearchRequest, body: dict, variant: str) -> None:
    try:
        safe_query = next((t.get("search_text") for t in body.get("trace", []) if t["node"] == "understand"), None)
        state.store.log_search(
            request_id=rid, session_id=req.session_id, query=safe_query, mode=req.mode,
            has_image=int(bool(req.image_base64)) + len(req.images_base64), variant=variant, tools=body["tools_used"],
            filters=body["filters_applied"], result_ids=[r["boot"]["id"] for r in body["results"]],
            n_results=len(body["results"]), latency_ms=body["latency_ms"],
            cost_usd=body.get("llm", {}).get("cost_usd", 0.0), grounded=int(body["grounded"]),
            understood_by=body["understood_by"], cache_hit=int(body.get("cache_hit", False)),
        )
    except Exception as e:  # logging must never fail a search
        log.warning(f"search log failed: {e}")
    log_event(log, "search", mode=req.mode, n_results=len(body["results"]), latency_ms=body["latency_ms"],
              cache_hit=body.get("cache_hit", False), variant=variant, understood_by=body["understood_by"])


# ------------------------------------------------------------------ catalog
@router.get("/boots/{boot_id}", tags=["catalog"])
def get_boot(boot_id: str):
    boot = state.agent.engine.catalog.get(boot_id)
    if not boot:
        raise HTTPException(404, "Boot not found")
    return boot


def _image_url(boot, matched_path: str | None) -> str | None:
    """URL of the photo to show for a result: the one that matched the query photo(s), if any."""
    if not boot.images:
        return None
    idx = next((i for i, im in enumerate(boot.images) if im.path == matched_path), 0)
    return f"/boots/{boot.id}/images/{idx}"


def _photo_file(boot_id: str, index: int):
    boot = state.agent.engine.catalog.get(boot_id)
    if not boot or not 0 <= index < len(boot.images):
        raise HTTPException(404, "Image not found")
    path = (settings.raw_dir / boot.images[index].path).resolve()
    if settings.raw_dir.resolve() not in path.parents or not path.is_file():   # path-traversal safe
        raise HTTPException(404, "Image not found")
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})


@router.get("/boots/{boot_id}/image", tags=["catalog"])
def get_boot_image(boot_id: str):
    """Primary photo (kept for backwards compatibility)."""
    return _photo_file(boot_id, 0)


@router.get("/boots/{boot_id}/images", response_model=list[BootImage], tags=["catalog"])
def list_boot_images(boot_id: str):
    """The boot's whole gallery: index, view and URL of every photo."""
    boot = state.agent.engine.catalog.get(boot_id)
    if not boot:
        raise HTTPException(404, "Boot not found")
    return [BootImage(index=i, view=im.view, url=f"/boots/{boot_id}/images/{i}") for i, im in enumerate(boot.images)]


@router.get("/boots/{boot_id}/images/{index}", tags=["catalog"])
def get_boot_photo(boot_id: str, index: int):
    return _photo_file(boot_id, index)


@router.get("/boots", tags=["catalog"])
def list_boots(limit: int = 100):
    return list(state.agent.engine.catalog.values())[:limit]


# ------------------------------------------------------------------ feedback / HITL / A-B
@router.post("/feedback", tags=["feedback"], dependencies=[Depends(security)])
def post_feedback(fb: FeedbackRequest):
    state.store.add_feedback(fb.request_id, fb.rating, fb.boot_id, fb.comment)
    m.FEEDBACK.labels("up" if fb.rating > 0 else "down").inc()
    return {"status": "recorded"}


@router.get("/feedback/review", tags=["feedback"], dependencies=[Depends(security)])
def review_queue(limit: int = 50):
    return state.store.review_queue(limit)


@router.get("/feedback/stats", tags=["feedback"], dependencies=[Depends(security)])
def ab_stats():
    return state.store.ab_stats()


# ------------------------------------------------------------------ ops
@router.get("/agent/graph", tags=["ops"])
def agent_graph():
    return {"mermaid": state.agent.mermaid()}


@router.get("/metrics", tags=["ops"])
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


app.include_router(router)
app.include_router(router, prefix="/api", include_in_schema=False)
