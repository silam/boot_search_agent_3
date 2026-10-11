import base64

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from app.api.main import app
    with TestClient(app) as c:
        yield c


def test_health_ready(client):
    assert client.get("/health").json()["status"] == "ok"
    ready = client.get("/ready").json()
    assert ready["boots"] == 48 and ready["images"] == 48


def test_search_text_and_cache(client):
    body = {"query": "black chelsea boots", "session_id": "s1"}
    r1 = client.post("/search", json=body).json()
    assert r1["results"][0]["boot"]["category"] == "chelsea" and not r1["cache_hit"]
    r2 = client.post("/search", json=body).json()
    assert r2["cache_hit"] and r2["request_id"] != r1["request_id"]


def test_search_image_and_prefix(client, settings):
    img = base64.b64encode((settings.raw_dir / "images" / "BOOT-033.png").read_bytes()).decode()
    r = client.post("/api/search", json={"image_base64": img, "mode": "image", "top_k": 3}).json()
    assert r["results"][0]["boot"]["id"] == "BOOT-033"
    assert client.get(r["results"][0]["image_url"]).status_code == 200


def test_validation_errors(client):
    assert client.post("/search", json={}).status_code == 422
    assert client.post("/search", json={"mode": "image", "query": "x"}).status_code == 422
    assert client.post("/search", json={"query": "x", "image_base64": "%%%"}).status_code == 422


def test_feedback_loop(client):
    rid = client.post("/search", json={"query": "logger boots", "use_cache": False}).json()["request_id"]
    assert client.post("/feedback", json={"request_id": rid, "rating": -1, "comment": "too pricey"}).status_code == 200
    assert any(row["request_id"] == rid for row in client.get("/feedback/review").json())
    assert client.post("/feedback", json={"request_id": rid, "rating": 5}).status_code == 422


def test_metrics_exposed(client):
    text = client.get("/metrics").text
    assert "searches_total" in text and "http_request_duration_seconds" in text
