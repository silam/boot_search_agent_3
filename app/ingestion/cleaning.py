"""
TRANSFORM step: cleaning, normalisation, validation, de-duplication.

DISCUSSION NOTE:
  "Garbage in, garbage out" is truer for RAG than for classic ML: a dirty description
  becomes a dirty embedding AND a dirty LLM context. This module:
    1. normalises text (HTML stripping, unicode, whitespace, casing)
    2. coerces types (price "$189.99" -> 189.99, "yes" -> True)
    3. validates against the Boot schema and REJECTS bad rows (with a reason)
    4. de-duplicates by id, keeping the first clean record
    5. normalises the image gallery: accepts a list of {path, view} dicts, a list of
       paths, or a legacy single "image" field; de-duplicates photos; blocks unsafe paths
       (absolute or containing ".."); infers the view from the file name
       ("03_three_quarter.png" -> "three_quarter"); and, when an image folder is given,
       drops photos whose file is missing and reports them.
  The pipeline reports how many rows were rejected — a data-quality metric you can alert on.
"""
from __future__ import annotations

import html
import re
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import ValidationError

from app.core.models import Boot

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = _TAG.sub(" ", text)
    text = unicodedata.normalize("NFKC", text)
    text = _WS.sub(" ", text).strip()
    # Fix SHOUTING descriptions (common in supplier feeds) -> sentence case
    letters = [c for c in text if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.8:
        text = ". ".join(s.strip().capitalize() for s in text.lower().split(". "))
    return text


def to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"true", "yes", "y", "1"}


def to_price(v: Any) -> float | None:
    try:
        return float(re.sub(r"[^\d.]", "", str(v)))
    except ValueError:
        return None


_VIEW_FROM_NAME = re.compile(r"^\d+[_-](.+)$")


def view_from_path(path: str) -> str:
    stem = PurePosixPath(path).stem.lower()
    m = _VIEW_FROM_NAME.match(stem)
    return m.group(1) if m else "unknown"


def normalise_images(rec: dict[str, Any]) -> tuple[list[dict[str, str]], list[str]]:
    """Return (clean image list, problems). Order is kept: the first photo is the primary one."""
    raw = rec.get("images") or ([rec["image"]] if rec.get("image") else [])
    if isinstance(raw, str):
        raw = [p for p in raw.split("|") if p.strip()]
    images, problems, seen = [], [], set()
    for item in raw:
        path = item.get("path", "") if isinstance(item, dict) else str(item)
        path = path.strip().replace("\\", "/")
        pure = PurePosixPath(path)
        if not path or pure.is_absolute() or ".." in pure.parts:
            problems.append(f"unsafe or empty image path: {path!r}")
            continue
        if path in seen:
            continue                                     # same photo listed twice
        seen.add(path)
        view = (item.get("view") if isinstance(item, dict) else None) or view_from_path(path)
        images.append({"path": path, "view": str(view).strip().lower()})
    return images, problems


def normalise_record(raw: dict[str, Any]) -> dict[str, Any]:
    rec = dict(raw)
    for key in ("name", "brand", "material", "description"):
        rec[key] = clean_text(rec.get(key))
    for key in ("category", "color", "safety_toe"):
        if rec.get(key):
            rec[key] = str(rec[key]).strip().lower()
    rec["price"] = to_price(rec.get("price"))
    rec["waterproof"] = to_bool(rec.get("waterproof", False))
    rec["features"] = [clean_text(f) for f in rec.get("features", []) if clean_text(f)]
    rec["reviews"] = [clean_text(r) for r in rec.get("reviews", []) if clean_text(r)]
    rec["images"], _ = normalise_images(rec)
    rec.pop("image", None)
    return rec


def clean_catalog(raw_records: list[dict[str, Any]], image_root: Path | None = None
                  ) -> tuple[list[Boot], list[dict[str, str]]]:
    """Returns (clean boots, issues). Issues with kind "record" are rejected rows; issues with
    kind "image" are photos dropped from an otherwise valid boot."""
    boots: list[Boot] = []
    rejected: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in raw_records:
        rid = str(raw.get("id", "?"))
        if rid in seen:
            rejected.append({"id": rid, "kind": "record", "reason": "duplicate id"})
            continue
        _, image_problems = normalise_images(raw)
        try:
            boot = Boot(**normalise_record(raw))
        except ValidationError as e:
            rejected.append({"id": rid, "kind": "record", "reason": e.errors()[0]["msg"]})
            continue
        rejected += [{"id": rid, "kind": "image", "reason": p} for p in image_problems]
        if image_root is not None:
            missing = [im for im in boot.images if not (Path(image_root) / im.path).is_file()]
            rejected += [{"id": rid, "kind": "image", "reason": f"file not found: {im.path}"} for im in missing]
            boot.images = [im for im in boot.images if im not in missing]
        seen.add(boot.id)
        boots.append(boot)
    return boots, rejected
