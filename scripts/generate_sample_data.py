"""
Generate a synthetic boot catalog (JSON + PNG images) so the project runs end to end
without any private data.

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
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
IMG_DIR = RAW / "images"

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
def draw_boot(shape: str, color: tuple[int, int, int], rng: random.Random) -> Image.Image:
    """Draw a simple boot silhouette. Shape encodes category, fill encodes color —
    so image similarity is meaningful even with a non-neural embedder."""
    img = Image.new("RGB", (256, 256), (245, 245, 242))
    d = ImageDraw.Draw(img)
    sole = (35, 30, 28) if shape != "rain" else tuple(max(0, c - 40) for c in color)
    jitter = lambda v: v + rng.randint(-4, 4)  # noqa: E731

    if shape == "western":
        shaft = [(jitter(90), 30), (jitter(150), 30), (150, 170), (225, 190), (240, 205), (85, 205)]
        d.polygon(shaft, fill=color)
        d.polygon([(85, 205), (240, 205), (235, 215), (85, 215)], fill=sole)
        d.rectangle([85, 205, 110, 230], fill=sole)  # heel
        d.line([(100, 60), (140, 100)], fill=tuple(min(255, c + 60) for c in color), width=3)
    elif shape == "rain":
        d.rectangle([jitter(80), 25, 150, 190], fill=color)
        d.polygon([(80, 150), (150, 150), (215, 175), (220, 200), (80, 200)], fill=color)
        d.rectangle([75, 200, 225, 222], fill=sole)
    elif shape == "chelsea":
        d.rectangle([jitter(85), 85, 150, 185], fill=color)
        d.polygon([(85, 160), (150, 160), (225, 185), (228, 205), (85, 205)], fill=color)
        d.rectangle([110, 95, 128, 160], fill=tuple(int(c * 0.55) for c in color))  # elastic gore
        d.rectangle([82, 205, 232, 220], fill=(205, 185, 140))  # crepe sole
    elif shape == "hiking":
        d.rectangle([jitter(85), 110, 155, 190], fill=color)
        d.polygon([(85, 160), (155, 160), (230, 180), (235, 205), (85, 205)], fill=color)
        for x in range(95, 150, 12):
            d.ellipse([x, 120, x + 6, 126], fill=(200, 60, 40))  # red lace hooks
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
            d.line([(140, y), (152, y + 6)], fill=(230, 220, 200), width=2)
    return img.filter(ImageFilter.GaussianBlur(0.6))


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
        "image": f"images/BOOT-{i:03d}.png",
    }


def main(n_per_category: int = 8) -> None:
    rng = random.Random(SEED)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    boots, i = [], 1
    for category in CATEGORIES:
        for _ in range(n_per_category):
            boot = make_boot(i, category, rng)
            draw_boot(CATEGORIES[category]["shape"], COLORS[boot["color"]], rng).save(IMG_DIR / f"{boot['id']}.png")
            boots.append(boot)
            i += 1

    # Inject a few "dirty" records on purpose so the cleaning step has real work to do
    dirty = dict(boots[0])
    dirty["description"] = "  " + dirty["description"].upper() + "  <br> "
    boots.append(dirty)                                    # duplicate id -> dedup
    boots.append({"id": "BOOT-BAD", "name": "", "price": "n/a"})  # invalid -> rejected

    (RAW / "catalog.json").write_text(json.dumps(boots, indent=2))
    print(f"Wrote {len(boots)} raw records and {i - 1} images to {RAW}")


if __name__ == "__main__":
    main()
