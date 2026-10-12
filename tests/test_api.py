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
    assert ready["boots"] == 48 and ready["images"] >= 48 * 8


def test_search_text_and_cache(client):
    body = {"query": "black chelsea boots", "session_id": "s1"}
    r1 = client.post("/search", json=body).json()
    assert r1["results"][0]["boot"]["category"] == "chelsea" and not r1["cache_hit"]
    r2 = client.post("/search", json=body).json()
    assert r2["cache_hit"] and r2["request_id"] != r1["request_id"]


def _b64(engine_boot, settings, view):
    im = next(i for i in engine_boot.images if i.view == view)
    return base64.b64encode((settings.raw_dir / im.path).read_bytes()).decode()


def test_search_image_and_prefix(client, settings):
    from app.core.container import get_engine
    boot = get_engine().catalog["BOOT-033"]
    r = client.post("/api/search", json={"image_base64": _b64(boot, settings, "sole"), "mode": "image",
                                         "top_k": 3}).json()
    top = r["results"][0]
    assert top["boot"]["id"] == "BOOT-033" and top["matched_view"] == "sole"
    assert top["image_url"].endswith(f"/images/{[i.view for i in boot.images].index('sole')}")
    assert client.get(top["image_url"]).status_code == 200
    assert len(top["image_urls"]) == len(boot.images)


def test_multi_photo_search(client, settings):
    from app.core.container import get_engine
    boot = get_engine().catalog["BOOT-031"]
    r = client.post("/search", json={"image_base64": _b64(boot, settings, "side_right"),
                                     "images_base64": [_b64(boot, settings, "sole")], "mode": "image"}).json()
    assert r["results"][0]["boot"]["id"] == "BOOT-031"
    too_many = [_b64(boot, settings, "front")] * 6
    assert client.post("/search", json={"images_base64": too_many, "mode": "image"}).status_code == 422


def test_gallery_endpoints(client):
    gallery = client.get("/boots/BOOT-001/images").json()
    assert len(gallery) >= 8 and gallery[0]["view"] == "side_right"
    assert client.get(gallery[-1]["url"]).status_code == 200
    assert client.get("/boots/BOOT-001/images/99").status_code == 404


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
