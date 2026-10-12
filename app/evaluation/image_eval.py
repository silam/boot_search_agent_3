"""
Offline evaluation of IMAGE search: does indexing every photo of a boot (and scoring
with MaxSim) beat indexing one photo per boot?

Two protocols:

  shopper (the main one)
    192 held-out "shopper photos" (data/eval/query_photos/, written by the generator): NEW
    shots of a boot from one of its angles, with stronger rotation, scale, lighting and
    background changes. They are never indexed. This is the real use case: someone
    photographs a boot from whatever angle and wants that boot.

  unseen_angle (a stress test)
    Every non-primary catalog photo is used as a query and that photo is removed from the
    index for that query, so the target boot has NO photo from the query's angle. This
    shows the limit of view-matching: if the catalog lacks the angle, the other boots'
    photos from that angle compete with it.

Three index strategies are compared on the same queries:
      primary_only   one vector per boot (the side photo) -> what the project had before
      all_max        one vector per photo, boot score = best photo (MaxSim)
      all_mean_top3  one vector per photo, boot score = mean of best 3 photos

Metrics (averaged over queries):
  product@1 / product@5 : the exact same boot is ranked 1st / in the top 5
  MRR                   : 1 / rank of the exact same boot
  lookalike@5           : share of the top 5 with the same category AND colour
                          (what "visually similar" means to a shopper)

Run:  python -m app.evaluation.image_eval
"""
from __future__ import annotations

import json
import statistics
import time

from PIL import Image

from app.core.config import get_settings
from app.core.container import get_engine

STRATEGIES = {
    "primary_only": {"views": ["__primary__"], "aggregation": "max"},
    "all_max": {"views": None, "aggregation": "max"},
    "all_mean_top3": {"views": None, "aggregation": "mean_top3"},
}


def _run(engine, queries, vectors, exclude_self: bool) -> tuple[dict, dict]:
    catalog = engine.catalog
    primary_views = sorted({b.images[0].view for b in catalog.values() if b.images})
    strategies, by_view = {}, {}
    for name, cfg in STRATEGIES.items():
        p1, p5, rr, look, lat = [], [], [], [], []
        for (boot_id, path, view), vec in zip(queries, vectors):
            views = primary_views if cfg["views"] == ["__primary__"] else None
            t0 = time.perf_counter()
            hits = engine.store.search_images(vec, None, limit=10, photos_per_boot=3, aggregation=cfg["aggregation"],
                                              views=views, exclude_paths=[path] if exclude_self else None)
            lat.append((time.perf_counter() - t0) * 1000)
            ranked = [h.boot_id for h in hits]
            rank = ranked.index(boot_id) + 1 if boot_id in ranked else None
            p1.append(1.0 if rank == 1 else 0.0)
            p5.append(1.0 if rank and rank <= 5 else 0.0)
            rr.append(1 / rank if rank else 0.0)
            target = catalog[boot_id]
            top5 = [catalog[b] for b in ranked[:5]]
            look.append(sum(b.category == target.category and b.color == target.color for b in top5) / max(len(top5), 1))
            by_view.setdefault(view, {}).setdefault(name, []).append(rr[-1])
        strategies[name] = {
            "product@1": round(statistics.mean(p1), 3), "product@5": round(statistics.mean(p5), 3),
            "MRR": round(statistics.mean(rr), 3), "lookalike@5": round(statistics.mean(look), 3),
            "p50_ms": round(sorted(lat)[len(lat) // 2], 2),
        }
    return strategies, {v: {k: round(statistics.mean(x), 3) for k, x in d.items()} for v, d in sorted(by_view.items())}


def _embed(engine, files):
    imgs = []
    for f in files:
        with Image.open(f) as im:
            imgs.append(im.convert("RGB"))
    return engine.image_embedder.embed_images(imgs)


def evaluate(protocols: tuple[str, ...] = ("shopper", "unseen_angle")) -> dict:
    s = get_settings()
    engine = get_engine()
    report = {"embedder": engine.image_embedder.name,
              "n_photos": sum(len(b.images) for b in engine.catalog.values()), "protocols": {}}
    if "shopper" in protocols:
        qs = json.loads((s.data_dir / "eval" / "image_queries.json").read_text())["queries"]
        queries = [(q["boot_id"], q["path"], q["view"]) for q in qs if q["boot_id"] in engine.catalog]
        vectors = _embed(engine, [s.data_dir / "eval" / q[1] for q in queries])
        strat, views = _run(engine, queries, vectors, exclude_self=False)
        report["protocols"]["shopper"] = {"n_queries": len(queries), "strategies": strat, "by_view": views}
    if "unseen_angle" in protocols:
        queries = [(b.id, im.path, im.view) for b in engine.catalog.values() for im in b.images[1:]]
        vectors = _embed(engine, [s.raw_dir / q[1] for q in queries])
        strat, views = _run(engine, queries, vectors, exclude_self=True)
        report["protocols"]["unseen_angle"] = {"n_queries": len(queries), "strategies": strat, "by_view": views}
    return report


def to_markdown(report: dict) -> str:
    cols = ["product@1", "product@5", "MRR", "lookalike@5", "p50_ms"]
    lines = [f"Image embedder `{report['embedder']}` — {report['n_photos']} indexed photos"]
    for proto, data in report["protocols"].items():
        lines += ["", f"### {proto} ({data['n_queries']} query photos)", "",
                  "| strategy | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for name, row in data["strategies"].items():
            lines.append(f"| {name} | " + " | ".join(str(row[c]) for c in cols) + " |")
        lines += ["", "MRR by query view:", "", "| view | " + " | ".join(STRATEGIES) + " |",
                  "|---" * (len(STRATEGIES) + 1) + "|"]
        for view, row in data["by_view"].items():
            lines.append(f"| {view} | " + " | ".join(str(row[k]) for k in STRATEGIES) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    rep = evaluate()
    out = get_settings().data_dir / "eval" / "image_report.json"
    out.write_text(json.dumps(rep, indent=2))
    print(to_markdown(rep))
    print(f"\nFull report: {out}")
