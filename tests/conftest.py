"""Test configuration: everything runs in-process — embedded Qdrant, lite embedders, SQLite
in a temp dir, no LLM key, no Redis. That's why CI needs no services and runs in seconds."""
import os
import tempfile

os.environ.update({
    "QDRANT_URL": ":memory:",
    "EMBEDDING_BACKEND": "lite",
    "ANTHROPIC_API_KEY": "",
    "REDIS_URL": "",
    "DATABASE_URL": f"sqlite:///{tempfile.mkdtemp()}/test.db",
    "LOG_LEVEL": "WARNING",
    "RATE_LIMIT_PER_MINUTE": "10000",
})

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def engine():
    from app.core.container import get_engine
    return get_engine()


@pytest.fixture(scope="session")
def settings():
    from app.core.config import get_settings
    return get_settings()
