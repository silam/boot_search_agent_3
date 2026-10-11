from app.core.models import Filters, RetrievedItem
from app.retrieval.bm25_index import tokenize
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.search_engine import RetrievalPlan


def test_tokenize_stems_and_splits_hyphens():
    toks = tokenize("Slip-resistant boots with laces")
    assert "slip-resistant" in toks and "slip" in toks and "lace" in toks and "boot" not in toks


def test_rrf_rewards_agreement():
    a = [RetrievedItem(boot_id=x, score=1, source="bm25") for x in ["A", "B", "C"]]
    b = [RetrievedItem(boot_id=x, score=1, source="text_vector") for x in ["B", "D", "A"]]
    fused = reciprocal_rank_fusion({"keyword": a, "semantic": b})
    assert fused[0][0] in {"A", "B"} and {fused[0][0], fused[1][0]} == {"A", "B"}
    assert fused[0][2] == ["keyword", "semantic"]


def test_rrf_weights():
    a = [RetrievedItem(boot_id="A", score=1, source="bm25")]
    b = [RetrievedItem(boot_id="B", score=1, source="text_vector")]
    assert reciprocal_rank_fusion({"keyword": a, "semantic": b}, {"semantic": 2.0})[0][0] == "B"


def test_keyword_finds_sku_and_jargon(engine):
    assert engine.keyword("BOOT-042")[0].boot_id == "BOOT-042"
    assert engine.catalog[engine.keyword("kiltie lace guard")[0].boot_id].category == "logger"


def test_filters_pushed_down_to_qdrant(engine):
    f = Filters(category="hiking", waterproof=True)
    hits = engine.semantic("comfortable boots", f)
    assert hits and all(engine.catalog[h.boot_id].category == "hiking" and engine.catalog[h.boot_id].waterproof
                        for h in hits)


def test_image_search_returns_itself_first(engine, settings):
    from PIL import Image
    img = Image.open(settings.raw_dir / "images" / "BOOT-020.png")
    hits = engine.image(img)
    assert hits[0].boot_id == "BOOT-020"
    assert engine.catalog[hits[1].boot_id].category == engine.catalog["BOOT-020"].category


def test_combined_image_and_text_respects_filters(engine, settings):
    from PIL import Image
    img = Image.open(settings.raw_dir / "images" / "BOOT-010.png")
    plan = RetrievalPlan(query="waterproof", image=img, tools=["image", "keyword", "semantic"],
                         filters=Filters(waterproof=True, max_price=250), top_k=5)
    results = engine.run(plan)
    assert results and all(r.boot.waterproof and r.boot.price <= 250 for r in results)
