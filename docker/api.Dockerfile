# ---------------------------------------------------------------------------
# Boot Search API image
#   docker build -f docker/api.Dockerfile -t boot-search-api .
#   docker build -f docker/api.Dockerfile --build-arg EMBEDDING_BACKEND=clip -t boot-search-api:clip .
#
# discussion talking points:
#   * Multi-stage build: compilers/wheels stay in the builder, the runtime is slim.
#   * Dependencies are installed BEFORE copying code, so code edits reuse the cached layer.
#   * ETL runs at build time and the clip models are downloaded at build time, so a new
#     container starts fast and never depends on the internet at runtime.
#   * Runs as a non-root user; HEALTHCHECK for orchestrators.
# ---------------------------------------------------------------------------
ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS builder
ARG EMBEDDING_BACKEND=lite
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY requirements/ requirements/
RUN python -m venv /opt/venv && . /opt/venv/bin/activate \
 && if [ "$EMBEDDING_BACKEND" = "clip" ]; then \
        pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements/clip.txt; \
    else \
        pip install -r requirements/api.txt; \
    fi

FROM python:${PYTHON_VERSION}-slim AS runtime
ARG EMBEDDING_BACKEND=lite
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    EMBEDDING_BACKEND=${EMBEDDING_BACKEND} \
    HF_HOME=/opt/hf-cache \
    PORT=8000
RUN useradd --create-home --uid 10001 app
COPY --from=builder /opt/venv /opt/venv
WORKDIR /srv
COPY app/ app/
COPY scripts/ scripts/
COPY data/raw/ data/raw/
COPY data/eval/ data/eval/
# Build-time ETL (clean + chunk) and, for clip, model download into the image
RUN mkdir -p /srv/state && python scripts/ingest.py --transform-only \
 && if [ "$EMBEDDING_BACKEND" = "clip" ]; then python -c "from app.embeddings.factory import get_embedders; get_embedders()"; fi \
 && chown -R app:app /srv /opt/hf-cache 2>/dev/null || chown -R app:app /srv
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/health', timeout=4)"
CMD ["sh", "-c", "uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT} --workers ${WEB_CONCURRENCY:-1} --proxy-headers"]
