from app.core.models import Boot
from app.ingestion.chunking import chunk_boot, recursive_split
from app.ingestion.cleaning import clean_catalog, clean_text


def test_clean_text_strips_html_whitespace_and_shouting():
    assert clean_text("  GREAT BOOT. VERY WARM <br> ") == "Great boot. Very warm"
    assert clean_text("a&amp;b") == "a&b"


def test_clean_catalog_dedups_and_rejects_invalid():
    good = {"id": "B1", "name": "X", "brand": "Y", "category": "work", "color": "Black", "material": "leather",
            "price": "$100.00", "waterproof": "yes", "description": "d"}
    boots, rejected = clean_catalog([good, dict(good), {"id": "B2", "name": "", "price": "n/a"}])
    assert [b.id for b in boots] == ["B1"]
    assert boots[0].price == 100.0 and boots[0].waterproof is True and boots[0].color == "black"
    assert {r["reason"] for r in rejected} >= {"duplicate id"}
    assert all(r["kind"] == "record" for r in rejected)
    assert len(rejected) == 2


def test_recursive_split_respects_size_and_overlap():
    text = ". ".join(f"Sentence number {i} about boots" for i in range(40))
    chunks = recursive_split(text, chunk_size=120, overlap=30)
    assert len(chunks) > 1
    assert all(len(c) <= 120 + 30 for c in chunks)
    assert all(not c.startswith(" ") for c in chunks)


def test_chunk_boot_sections():
    b = Boot(id="B1", name="Test", brand="T", category="work", color="black", material="leather", price=10,
             waterproof=True, description="A boot.", reviews=["Great", "Comfy"], features=["steel shank"])
    chunks = chunk_boot(b)
    assert {c.section for c in chunks} == {"profile", "description", "reviews"}
    profile = next(c for c in chunks if c.section == "profile").text
    assert "SKU B1" in profile and "waterproof" in profile


def test_normalise_images_accepts_every_shape_and_blocks_unsafe_paths():
    from app.ingestion.cleaning import normalise_images
    imgs, problems = normalise_images({"images": [
        {"path": "images/B1/01_side_right.png", "view": "side_right"},
        "images/B1/02_sole.png",                       # plain path -> view inferred from the name
        "images/B1/02_sole.png",                       # duplicate -> dropped
        "../../etc/passwd",                            # unsafe -> reported
        "/abs/x.png",                                  # unsafe -> reported
    ]})
    assert [i["view"] for i in imgs] == ["side_right", "sole"]
    assert len(problems) == 2
    legacy, _ = normalise_images({"image": "images/B1.png"})
    assert legacy == [{"path": "images/B1.png", "view": "unknown"}]
    piped, _ = normalise_images({"images": "a/01_front.png|a/02_back.png"})
    assert [i["view"] for i in piped] == ["front", "back"]


def test_missing_photo_files_are_dropped_and_reported(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "01_side_right.png").write_bytes(b"x")
    rec = {"id": "B1", "name": "X", "brand": "Y", "category": "work", "color": "black", "material": "m",
           "price": 10, "waterproof": True, "description": "d",
           "images": ["images/01_side_right.png", "images/02_sole.png"]}
    boots, issues = clean_catalog([rec], image_root=tmp_path)
    assert [i.path for i in boots[0].images] == ["images/01_side_right.png"]
    assert issues == [{"id": "B1", "kind": "image", "reason": "file not found: images/02_sole.png"}]
