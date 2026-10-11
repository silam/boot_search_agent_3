"""
TRANSFORM step: cleaning, normalisation, validation, de-duplication.

INTERVIEW NOTE:
  "Garbage in, garbage out" is truer for RAG than for classic ML: a dirty description
  becomes a dirty embedding AND a dirty LLM context. This module:
    1. normalises text (HTML stripping, unicode, whitespace, casing)
    2. coerces types (price "$189.99" -> 189.99, "yes" -> True)
    3. validates against the Boot schema and REJECTS bad rows (with a reason)
    4. de-duplicates by id, keeping the first clean record
  The pipeline reports how many rows were rejected — a data-quality metric you can alert on.
"""
from __future__ import annotations

import html
import re
import unicodedata
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
    return rec


def clean_catalog(raw_records: list[dict[str, Any]]) -> tuple[list[Boot], list[dict[str, str]]]:
    """Returns (clean boots, rejected rows with reasons)."""
    boots: list[Boot] = []
    rejected: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in raw_records:
        rid = str(raw.get("id", "?"))
        if rid in seen:
            rejected.append({"id": rid, "reason": "duplicate id"})
            continue
        try:
            boot = Boot(**normalise_record(raw))
        except ValidationError as e:
            rejected.append({"id": rid, "reason": e.errors()[0]["msg"]})
            continue
        seen.add(boot.id)
        boots.append(boot)
    return boots, rejected
