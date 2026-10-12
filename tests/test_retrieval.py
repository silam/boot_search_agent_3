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


def _photo(engine, settings, boot_id, view):
    from PIL import Image
    im = next(i for i in engine.catalog[boot_id].images if i.view == view)
    return Image.open(settings.raw_dir / im.path), im.path


def test_every_boot_has_a_gallery_and_every_photo_is_indexed(engine):
    counts = [len(b.images) for b in engine.catalog.values()]
    assert min(counts) >= 8
    assert engine.store.count(engine.store.image_collection) == sum(counts)


def test_image_search_groups_photos_by_boot(engine, settings):
    img, _ = _photo(engine, settings, "BOOT-020", "side_right")
    hits = engine.image(img)
    assert hits[0].boot_id == "BOOT-020" and hits[0].matched_view == "side_right"
    assert len({h.boot_id for h in hits}) == len(hits)          # one entry per boot, not per photo
    assert engine.catalog[hits[1].boot_id].category == engine.catalog["BOOT-020"].category


def test_sole_photo_matches_the_sole_view(engine, settings):
    img, _ = _photo(engine, settings, "BOOT-012", "sole")
    hit = engine.image(img)[0]
    assert hit.boot_id == "BOOT-012" and hit.matched_view == "sole"


def test_exclude_paths_removes_the_query_photo(engine, settings):
    img, path = _photo(engine, settings, "BOOT-020", "back")
    hits = engine.image(img, exclude_paths=[path])
    assert all(h.matched_image != path for h in hits)


def test_multiple_query_photos_are_combined(engine, settings):
    from app.core.models import RetrievedItem
    from app.retrieval.search_engine import combine_query_images
    a = [RetrievedItem(boot_id="X", score=0.9, source="image_vector", matched_image="x1", matched_view="side"),
         RetrievedItem(boot_id="Y", score=0.8, source="image_vector", matched_image="y1", matched_view="side")]
    b = [RetrievedItem(boot_id="Y", score=0.95, source="image_vector", matched_image="y2", matched_view="sole"),
         RetrievedItem(boot_id="Z", score=0.5, source="image_vector", matched_image="z2", matched_view="sole")]
    combined = combine_query_images([a, b])
    assert combined[0].boot_id == "Y"                     # good from both angles beats great from one
    assert combined[0].matched_view == "sole"             # best single match is reported
    side, sole = _photo(engine, settings, "BOOT-031", "side_right")[0], _photo(engine, settings, "BOOT-031", "sole")[0]
    assert engine.image([side, sole])[0].boot_id == "BOOT-031"


def test_combined_image_and_text_respects_filters(engine, settings):
    img, _ = _photo(engine, settings, "BOOT-010", "side_right")
    plan = RetrievalPlan(query="waterproof", images=[img], tools=["image", "keyword", "semantic"],
                         filters=Filters(waterproof=True, max_price=250), top_k=5)
    results = engine.run(plan)
    assert results and all(r.boot.waterproof and r.boot.price <= 250 for r in results)
    assert any(r.matched_image for r in results)
