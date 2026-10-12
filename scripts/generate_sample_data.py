"""
Generate a synthetic boot catalog (JSON + PNG images) so the project runs end to end
without any private data.

Every boot gets 8-10 photos from different views (side, mirrored side, three-quarter,
front, back, top, sole, close-up detail, and sometimes an on-the-floor "lifestyle" shot
and a heel close-up), like a real e-commerce product gallery:

    data/raw/images/BOOT-001/01_side_right.png
    data/raw/images/BOOT-001/02_side_left.png
    ...
    catalog.json -> "images": [{"path": "images/BOOT-001/01_side_right.png", "view": "side_right"}, ...]

Each boot also gets its own small style details (lace/stitch accent colour, sole colour,
pull tab) so two boots of the same category and colour are still visually different.

INTERVIEW NOTE:
  In production this step is replaced by a real "Data Collection" job (catalog API,
  PIM export, web scrape). Keeping a deterministic generator (fixed seed) means the
  eval set in data/eval/golden_set.json always matches the data — reproducibility is
  what makes offline evaluation trustworthy.

Run:  python scripts/generate_sample_data.py
"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
IMG_DIR = RAW / "images"
EVAL = ROOT / "data" / "eval"

SEED = 42
BRANDS = ["Northline", "Ironpeak", "Prairie & Co.", "Granite Forge", "Lakeshore", "Timber Trail"]

COLORS = {
    "black": (25, 25, 28),
    "brown": (112, 66, 35),
    "tan": (196, 150, 98),
    "oxblood": (110, 25, 35),
    "olive": (95, 105, 55),
    "gray": (130, 132, 135),
    "honey": (214, 160, 60),
    "navy": (30, 45, 90),
}

# Each category = a product archetype with its own silhouette and typical features
CATEGORIES = {
    "work": {
        "shape": "work",
        "materials": ["full-grain leather", "oil-tanned leather", "nubuck leather"],
        "features": ["slip-resistant outsole", "Goodyear welt construction", "electrical hazard rated",
                     "cushioned footbed", "oil-resistant sole", "puncture-resistant midsole"],
        "uses": ["construction sites", "warehouse shifts", "factory floors", "long days on concrete"],
        "price": (150, 340),
    },
    "hiking": {
        "shape": "hiking",
        "materials": ["suede and mesh", "waterproof nubuck", "synthetic ripstop"],
        "features": ["Vibram-style lug outsole", "breathable membrane", "rock plate", "padded collar",
                     "lightweight EVA midsole", "speed-lace hooks"],
        "uses": ["mountain trails", "day hikes", "backpacking trips", "muddy forest paths"],
        "price": (110, 260),
    },
    "western": {
        "shape": "western",
        "materials": ["exotic-look embossed leather", "full-grain leather", "distressed leather"],
        "features": ["pointed toe", "stacked leather heel", "decorative stitching", "pull-on design",
                     "steel shank", "leather sole"],
        "uses": ["ranch work", "country concerts", "riding", "everyday western style"],
        "price": (170, 420),
    },
    "chelsea": {
        "shape": "chelsea",
        "materials": ["suede", "polished leather", "waxed leather"],
        "features": ["elastic side panels", "pull tab", "slim profile", "crepe sole", "leather lining"],
        "uses": ["the office", "weekend errands", "date nights", "city walking"],
        "price": (130, 300),
    },
    "rain": {
        "shape": "rain",
        "materials": ["natural rubber", "neoprene-lined rubber", "PVC"],
        "features": ["fully waterproof", "removable insole", "chemical-resistant", "wide calf fit",
                     "self-cleaning tread"],
        "uses": ["rainy days", "gardening", "farm chores", "muddy festivals"],
        "price": (60, 180),
    },
    "logger": {
        "shape": "logger",
        "materials": ["heavy-duty leather", "oil-tanned leather"],
        "features": ["raised logger heel", "aggressive lug sole", "kiltie lace guard", "steel shank",
                     "triple-stitched seams"],
        "uses": ["forestry", "linemen work", "rough terrain", "lumber yards"],
        "price": (230, 450),
    },
}

ADJ = ["rugged", "classic", "versatile", "heritage", "lightweight", "all-day", "premium", "dependable"]
NOUNS = {"work": ["Foreman", "Shift", "Rivet", "Anvil", "Ironworker"],
         "hiking": ["Summit", "Ridge", "Switchback", "Trailhead", "Alpine"],
         "western": ["Maverick", "Ranger", "Outlaw", "Mesa", "Stockyard"],
         "chelsea": ["Metro", "Harbor", "Avenue", "Mercer", "Loft"],
         "rain": ["Puddle", "Monsoon", "Drizzle", "Marsh", "Tide"],
         "logger": ["Timber", "Cascade", "Sawmill", "Highline", "Douglas"]}
REVIEW_TEMPLATES = [
    "Wore these for {use} and my feet felt great after 10 hours.",
    "Took about two weeks to break in, now they are the most comfortable boots I own.",
    "Runs half a size large, order down. Quality of the {material} is excellent.",
    "Kept my feet dry through a full day of {use}.",
    "Solid boot for the price, the sole has held up for a year.",
    "Looks even better in person, the {color} color is rich and deep.",
    "A bit heavy, but very supportive and stable on uneven ground.",
]


# ---------------------------------------------------------------- image drawing
BG = (245, 245, 242)
ACCENTS = [(200, 60, 40), (230, 220, 200), (240, 190, 60), (60, 120, 200), (40, 40, 40), (90, 160, 90)]
SOLES = {"black": (35, 30, 28), "brown": (80, 52, 30), "gum": (170, 120, 60), "gray": (95, 95, 98),
         "white": (225, 222, 215)}


def make_style(boot: dict, rng: random.Random) -> dict:
    """Per-boot visual details. Uses its OWN rng so the catalog attributes stay stable."""
    shape = CATEGORIES[boot["category"]]["shape"]
    color = COLORS[boot["color"]]
    if shape == "rain":
        sole = tuple(max(0, c - 40) for c in color)
    elif shape == "chelsea":
        sole = rng.choice([(205, 185, 140), SOLES["black"], SOLES["brown"]])
    elif shape == "western":
        sole = rng.choice([(150, 105, 60), SOLES["black"], SOLES["brown"]])
    else:
        sole = SOLES[rng.choice(list(SOLES))]
    return {"shape": shape, "color": color, "sole": sole, "accent": rng.choice(ACCENTS),
            "pull_tab": rng.random() < 0.5}


def draw_boot(shape: str, color: tuple[int, int, int], rng: random.Random, style: dict | None = None) -> Image.Image:
    """Side view (the primary product photo). Shape encodes category, fill encodes colour,
    accent/sole encode the individual boot."""
    style = style or {"sole": (35, 30, 28), "accent": (200, 60, 40), "pull_tab": False}
    img = Image.new("RGB", (256, 256), BG)
    d = ImageDraw.Draw(img)
    sole = style["sole"]
    accent = style["accent"]
    jitter = lambda v: v + rng.randint(-4, 4)  # noqa: E731

    if shape == "western":
        shaft = [(jitter(90), 30), (jitter(150), 30), (150, 170), (225, 190), (240, 205), (85, 205)]
        d.polygon(shaft, fill=color)
        d.polygon([(85, 205), (240, 205), (235, 215), (85, 215)], fill=sole)
        d.rectangle([85, 205, 110, 230], fill=sole)  # heel
        d.line([(100, 60), (140, 100)], fill=accent, width=3)
    elif shape == "rain":
        d.rectangle([jitter(80), 25, 150, 190], fill=color)
        d.polygon([(80, 150), (150, 150), (215, 175), (220, 200), (80, 200)], fill=color)
        d.rectangle([75, 200, 225, 222], fill=sole)
    elif shape == "chelsea":
        d.rectangle([jitter(85), 85, 150, 185], fill=color)
        d.polygon([(85, 160), (150, 160), (225, 185), (228, 205), (85, 205)], fill=color)
        d.rectangle([110, 95, 128, 160], fill=tuple(int(c * 0.55) for c in color))  # elastic gore
        d.rectangle([82, 205, 232, 220], fill=sole)  # crepe / rubber sole
    elif shape == "hiking":
        d.rectangle([jitter(85), 110, 155, 190], fill=color)
        d.polygon([(85, 160), (155, 160), (230, 180), (235, 205), (85, 205)], fill=color)
        for x in range(95, 150, 12):
            d.ellipse([x, 120, x + 6, 126], fill=accent)  # lace hooks
        d.rectangle([80, 205, 238, 228], fill=sole)
        for x in range(82, 236, 14):
            d.rectangle([x, 228, x + 7, 234], fill=sole)  # lugs
    elif shape == "logger":
        d.rectangle([jitter(85), 30, 150, 185], fill=color)
        d.polygon([(85, 160), (150, 160), (225, 185), (230, 205), (85, 205)], fill=color)
        d.rectangle([80, 205, 232, 222], fill=sole)
        d.rectangle([80, 205, 112, 240], fill=sole)  # tall logger heel
    else:  # work boot, 6 inch
        d.rectangle([jitter(85), 70, 150, 185], fill=color)
        d.polygon([(85, 160), (150, 160), (228, 182), (232, 205), (85, 205)], fill=color)
        d.rectangle([80, 205, 236, 224], fill=sole)
        for y in range(80, 160, 14):
            d.line([(140, y), (152, y + 6)], fill=accent, width=2)
    if style.get("pull_tab"):
        top = {"western": 30, "rain": 25, "chelsea": 85, "hiking": 110, "logger": 30}.get(shape, 70)
        d.rectangle([92, top - 12, 104, top + 4], fill=accent)
    return img.filter(ImageFilter.GaussianBlur(0.6))


# ---------------------------------------------------------------- other views
SHAFT_TOP = {"western": 30, "rain": 25, "chelsea": 85, "hiking": 110, "logger": 30, "work": 70}
LACED = {"work", "hiking", "logger"}


def draw_front(style: dict) -> Image.Image:
    shape, color, sole, accent = style["shape"], style["color"], style["sole"], style["accent"]
    img = Image.new("RGB", (256, 256), BG)
    d = ImageDraw.Draw(img)
    top = SHAFT_TOP[shape]
    d.rectangle([93, top, 163, 190], fill=color)
    if shape == "western":
        d.polygon([(88, 175), (168, 175), (140, 218), (116, 218)], fill=color)   # pointed toe
    else:
        d.ellipse([80, 168, 176, 216], fill=color)
    d.rectangle([78, 210, 178, 224], fill=sole)
    if shape in LACED:
        for y in range(top + 10, 185, 12):
            d.line([(118, y), (138, y + 6)], fill=accent, width=2)
            d.line([(138, y), (118, y + 6)], fill=accent, width=2)
    if shape == "chelsea":
        d.rectangle([93, top + 8, 101, 165], fill=tuple(int(c * 0.55) for c in color))
        d.rectangle([155, top + 8, 163, 165], fill=tuple(int(c * 0.55) for c in color))
    return img


def draw_back(style: dict) -> Image.Image:
    shape, color, sole, accent = style["shape"], style["color"], style["sole"], style["accent"]
    img = Image.new("RGB", (256, 256), BG)
    d = ImageDraw.Draw(img)
    top = SHAFT_TOP[shape]
    d.rectangle([93, top, 163, 205], fill=color)
    d.rectangle([97, 150, 159, 205], fill=tuple(int(c * 0.8) for c in color))   # heel counter
    heel_bottom = 240 if shape in {"logger", "western"} else 226
    d.rectangle([100, 205, 156, heel_bottom], fill=sole)
    if style["pull_tab"] or shape in {"chelsea", "western"}:
        d.rectangle([120, top - 14, 136, top + 6], fill=accent)
    return img


def draw_top(style: dict) -> Image.Image:
    shape, color, accent = style["shape"], style["color"], style["accent"]
    img = Image.new("RGB", (256, 256), BG)
    d = ImageDraw.Draw(img)
    if shape == "western":
        d.polygon([(40, 105), (150, 95), (228, 130), (150, 165), (40, 155)], fill=color)
    else:
        d.ellipse([36, 92, 224, 168], fill=color)
    d.ellipse([50, 106, 104, 154], fill=tuple(int(c * 0.35) for c in color))   # shaft opening
    if shape in LACED:
        for x in range(112, 176, 12):
            d.line([(x, 112), (x + 8, 148)], fill=accent, width=2)
    return img


def draw_sole(style: dict) -> Image.Image:
    shape, color, sole = style["shape"], style["color"], style["sole"]
    img = Image.new("RGB", (256, 256), BG)
    d = ImageDraw.Draw(img)
    d.ellipse([34, 90, 226, 170], fill=color)                 # upper rim visible around the sole
    d.ellipse([40, 95, 220, 165], fill=sole)
    tread = tuple(min(255, c + 45) for c in sole)
    if shape in {"hiking", "logger"}:
        for x in range(60, 205, 16):
            for y in (110, 128, 146):
                d.rectangle([x, y, x + 8, y + 8], fill=tread)  # lugs
    elif shape == "work":
        for x in range(55, 210, 9):
            d.line([(x, 104), (x, 156)], fill=tread, width=2)
    elif shape == "rain":
        for x in range(55, 205, 14):
            d.line([(x, 130), (x + 7, 110), (x + 14, 130), (x + 7, 150), (x, 130)], fill=tread, width=2)
    d.rectangle([40, 112, 78, 148], outline=tread, width=3)     # heel block
    return img


def photo_jitter(img: Image.Image, rng: random.Random, max_rot: float = 4.0) -> Image.Image:
    """Make each 'photo' a little different: rotation, scale, brightness, contrast."""
    img = img.rotate(rng.uniform(-max_rot, max_rot), resample=Image.BICUBIC, fillcolor=BG)
    scale = rng.uniform(0.92, 1.06)
    w = int(256 * scale)
    img = img.resize((w, w), Image.BICUBIC)
    canvas = Image.new("RGB", (256, 256), BG)
    canvas.paste(img, ((256 - w) // 2, (256 - w) // 2)) if w <= 256 else canvas.paste(
        img.crop(((w - 256) // 2, (w - 256) // 2, (w - 256) // 2 + 256, (w - 256) // 2 + 256)), (0, 0))
    canvas = ImageEnhance.Brightness(canvas).enhance(rng.uniform(0.93, 1.05))
    canvas = ImageEnhance.Contrast(canvas).enhance(rng.uniform(0.95, 1.08))
    return canvas.filter(ImageFilter.GaussianBlur(0.5))


def base_views(side: Image.Image, style: dict, rng: random.Random) -> list[tuple[str, Image.Image]]:
    """The clean, un-jittered drawing of every view: 8 standard + up to 2 optional."""
    three_q = side.transform((256, 256), Image.AFFINE, (1, 0.22, -26, 0.06, 1, -6),
                             resample=Image.BICUBIC, fillcolor=BG)
    x0, y0 = rng.randint(110, 135), rng.randint(120, 140)
    detail = side.crop((x0, y0, x0 + 120, y0 + 120)).resize((256, 256), Image.BICUBIC)
    views = [
        ("side_right", side),
        ("side_left", ImageOps.mirror(side)),
        ("three_quarter", three_q),
        ("front", draw_front(style)),
        ("back", draw_back(style)),
        ("top", draw_top(style)),
        ("sole", draw_sole(style)),
        ("detail_toe", detail),
    ]
    if rng.random() < 0.6:   # on a wooden floor, a harder real-world shot
        floor = Image.new("RGB", (256, 256), (222, 205, 180))
        mask = Image.eval(side.convert("L"), lambda v: 255 if v < 225 else 0)
        floor.paste(side, (0, 0), mask)
        views.append(("lifestyle", floor))
    if rng.random() < 0.5:
        views.append(("detail_heel", side.crop((60, 150, 180, 250)).resize((256, 256), Image.BICUBIC)))
    return views


def catalog_photos(views: list[tuple[str, Image.Image]], rng: random.Random) -> list[tuple[str, Image.Image]]:
    """Catalog photos: the primary side shot as drawn, every other view lightly jittered."""
    return [views[0]] + [(name, photo_jitter(img, rng)) for name, img in views[1:]]


def shopper_photos(views: list[tuple[str, Image.Image]], rng: random.Random, n: int = 4
                   ) -> list[tuple[str, Image.Image]]:
    """Held-out 'shopper' photos for evaluation: NEW shots of some of the boot's views, with
    stronger rotation/scale/lighting changes and sometimes a different background. They are
    never indexed, so the eval measures what happens when someone photographs a boot."""
    out = []
    for name, img in rng.sample(views, k=min(n, len(views))):
        shot = photo_jitter(img, rng, max_rot=9.0)
        if rng.random() < 0.3:      # pure-white studio background instead of off-white
            mask = Image.eval(shot.convert("L"), lambda v: 255 if v < 225 else 0)
            white = Image.new("RGB", (256, 256), (255, 255, 255))
            white.paste(shot, (0, 0), mask)
            shot = white
        out.append((name, shot))
    return out


# ---------------------------------------------------------------- catalog
def make_boot(i: int, category: str, rng: random.Random) -> dict:
    spec = CATEGORIES[category]
    color = rng.choice(list(COLORS))
    material = rng.choice(spec["materials"])
    features = rng.sample(spec["features"], k=3)
    waterproof = category in {"rain"} or rng.random() < 0.45
    safety_toe = rng.choice(["steel", "composite", "none"]) if category in {"work", "logger"} else "none"
    if waterproof and "waterproof" not in " ".join(features):
        features.append("waterproof lining")
    if safety_toe != "none":
        features.append(f"{safety_toe} safety toe")
    lo, hi = spec["price"]
    price = round(rng.uniform(lo, hi) / 5) * 5 - 0.01
    use = rng.choice(spec["uses"])
    brand = rng.choice(BRANDS)
    suffix = rng.choice(["", "Pro", "II", "Classic", "Lite"] + (["6-inch"] if category == "work" else []))
    name = f"{brand} {rng.choice(NOUNS[category])} {suffix}".strip()

    description = (
        f"The {name} is a {rng.choice(ADJ)} {category} boot made from {color} {material}. "
        f"Built for {use}, it features {', '.join(features[:-1])} and {features[-1]}. "
        + ("Fully waterproof to keep you dry. " if waterproof else "")
        + (f"Meets ASTM safety standards with a {safety_toe} toe cap. " if safety_toe != "none" else "")
    )
    reviews = [
        rng.choice(REVIEW_TEMPLATES).format(use=rng.choice(spec["uses"]), material=material, color=color)
        for _ in range(rng.randint(2, 4))
    ]
    return {
        "id": f"BOOT-{i:03d}",
        "name": name,
        "brand": brand,
        "category": category,
        "color": color,
        "material": material,
        "price": price,
        "waterproof": waterproof,
        "safety_toe": safety_toe,
        "features": features,
        "description": description,
        "reviews": reviews,
        "images": [],          # filled in by main() once the photos are rendered
    }


def main(n_per_category: int = 8) -> None:
    rng = random.Random(SEED)
    if IMG_DIR.exists():
        shutil.rmtree(IMG_DIR)
    IMG_DIR.mkdir(parents=True)
    if (EVAL / "query_photos").exists():
        shutil.rmtree(EVAL / "query_photos")
    (EVAL / "query_photos").mkdir(parents=True)
    boots, queries, i, n_images = [], [], 1, 0
    for category in CATEGORIES:
        for _ in range(n_per_category):
            boot = make_boot(i, category, rng)
            # separate, per-boot rng for visuals: changing the photos never changes the catalog text
            vis_rng = random.Random(f"{SEED}-{boot['id']}")
            style = make_style(boot, vis_rng)
            side = draw_boot(CATEGORIES[category]["shape"], COLORS[boot["color"]], rng, style)
            folder = IMG_DIR / boot["id"]
            folder.mkdir()
            views = base_views(side, style, vis_rng)
            for n, (view, img) in enumerate(catalog_photos(views, vis_rng), start=1):
                rel = f"images/{boot['id']}/{n:02d}_{view}.png"
                img.save(RAW / rel)
                boot["images"].append({"path": rel, "view": view})
            q_rng = random.Random(f"{SEED}-{boot['id']}-shopper")
            for view, img in shopper_photos(views, q_rng):
                rel = f"query_photos/{boot['id']}_{view}.png"
                img.save(EVAL / rel)
                queries.append({"path": rel, "boot_id": boot["id"], "view": view})
            n_images += len(boot["images"])
            boots.append(boot)
            i += 1

    # Inject a few "dirty" records on purpose so the cleaning step has real work to do
    dirty = dict(boots[0])
    dirty["description"] = "  " + dirty["description"].upper() + "  <br> "
    boots.append(dirty)                                    # duplicate id -> dedup
    boots.append({"id": "BOOT-BAD", "name": "", "price": "n/a"})  # invalid -> rejected

    (RAW / "catalog.json").write_text(json.dumps(boots, indent=2))
    (EVAL / "image_queries.json").write_text(json.dumps({
        "_doc": "Held-out shopper photos (not in the index) for app/evaluation/image_eval.py. "
                "Each one is a new shot of one view of boot_id.",
        "queries": queries}, indent=2))
    print(f"Wrote {len(boots)} raw records, {i - 1} boots, {n_images} images to {RAW}")
    print(f"Wrote {len(queries)} held-out shopper photos to {EVAL / 'query_photos'}")


if __name__ == "__main__":
    main()
