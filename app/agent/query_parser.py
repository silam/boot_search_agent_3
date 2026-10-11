"""
Rule-based query understanding — the deterministic fallback for the LLM.

Extracts structured filters from free text and expands synonyms (query expansion).

INTERVIEW NOTE — why keep rules when you have an LLM?
  1. Availability: if the LLM times out, search still works.
  2. Cost/latency: zero tokens, ~0.1 ms.
  3. Testability: unit tests pin the behaviour (tests/test_query_parser.py).
  4. Evaluation baseline: you can measure how much the LLM actually improves things
     over the rules (if it doesn't, don't pay for it!).
"""
from __future__ import annotations

import re

from app.core.models import Filters

CATEGORY_SYNONYMS = {
    "western": ["western", "cowboy", "cowgirl", "ranch", "rodeo", "riding"],
    "rain": ["rain boot", "rain boots", "rubber", "wellies", "wellington", "galoshes", "gardening"],
    "chelsea": ["chelsea", "slip-on", "slip on", "elastic side", "pull-on dress"],
    "hiking": ["hiking", "hiker", "trail", "backpacking", "trekking", "mountain"],
    "logger": ["logger", "lineman", "linemen", "forestry", "logging", "lumber"],
    "work": ["work boot", "work boots", "construction", "job site", "jobsite", "warehouse", "factory"],
}
COLORS = {
    "black": ["black"], "brown": ["brown", "chocolate"], "tan": ["tan", "beige", "camel", "wheat"],
    "oxblood": ["oxblood", "burgundy", "red", "maroon"], "olive": ["olive", "green"],
    "gray": ["gray", "grey"], "honey": ["honey", "yellow", "gold"], "navy": ["navy", "blue"],
}
# Query expansion: add catalog vocabulary for common shopper phrasings
EXPANSIONS = {
    "keep my feet dry": "waterproof", "dry feet": "waterproof", "wet": "waterproof",
    "cowboy": "western pointed toe pull-on", "slip-on": "elastic side panels pull tab",
    "lineman": "logger raised heel climbing", "comfortable": "cushioned footbed comfortable",
    "grip": "slip-resistant lug outsole", "safety": "safety toe ASTM",
}
SKU = re.compile(r"\bboot-\d{3}\b", re.I)
PRICE_MAX = re.compile(r"(?:under|below|less than|max(?:imum)?|up to|<)\s*\$?\s*(\d+)", re.I)
PRICE_MIN = re.compile(r"(?:over|above|more than|at least|min(?:imum)?|>)\s*\$?\s*(\d+)", re.I)
VISUAL_WORDS = {"look", "looks", "looking like", "color", "colour", "style", "photo", "picture", "similar"}


def _contains(text: str, phrase: str) -> bool:
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def parse_query(query: str) -> dict:
    """Returns {"search_text", "filters": Filters, "intent", "needs_clarification", "clarifying_question"}."""
    q = query.lower().strip()
    f = Filters()

    for cat, words in CATEGORY_SYNONYMS.items():
        if any(_contains(q, w) for w in words):
            f.category = cat
            break
    for color, words in COLORS.items():
        if any(_contains(q, w) for w in words):
            f.color = color
            break

    if re.search(r"\b(not|no|non)[- ]waterproof\b", q):
        f.waterproof = False
    elif re.search(r"\bwaterproof\b|keep (my )?feet dry|dry feet", q):
        f.waterproof = True

    if re.search(r"\b(no|without|soft)[- ](steel|safety|composite)?[- ]?toe\b", q):
        f.safety_toe = "none"
    elif re.search(r"\bsteel[- ]toe", q):
        f.safety_toe = "steel"
    elif re.search(r"\bcomposite[- ]toe|composite safety", q):
        f.safety_toe = "composite"
    elif re.search(r"\bsafety[- ]toe|safety boots?\b", q):
        f.safety_toe = "any"

    if m := PRICE_MAX.search(q):
        f.max_price = float(m.group(1))
    if m := PRICE_MIN.search(q):
        f.min_price = float(m.group(1))

    expanded = query
    for phrase, extra in EXPANSIONS.items():
        if phrase in q:
            expanded += " " + extra

    if SKU.search(q) or (q.startswith('"') and q.endswith('"')):
        intent = "keyword_lookup"
    elif any(w in q for w in VISUAL_WORDS):
        intent = "visual"
    else:
        intent = "describe"

    words = re.findall(r"[a-z]+", q)
    vague = len([w for w in words if w not in {"boot", "boots", "a", "some", "the", "pair", "of", "show", "me"}]) == 0
    return {
        "search_text": expanded,
        "filters": f,
        "intent": intent,
        "needs_clarification": vague,
        "clarifying_question": "What will you mainly wear them for — work, hiking, riding, the city, or rain?"
        if vague else None,
    }
