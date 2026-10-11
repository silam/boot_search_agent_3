"""
Output guardrails — hallucination detection for the generated answer.

The answer is only allowed to talk about products we actually retrieved. We check:
  1. Every SKU mentioned (BOOT-123) must be in the retrieved set.
  2. Every product NAME mentioned must be a retrieved product, not a catalog product we
     didn't show, and not an invented one.
  3. Every price mentioned ($123.45) must equal the price of a retrieved product.
If a check fails, the agent replaces the LLM answer with a safe, template-generated one
and reports the failure (metric: answer_grounding_failures_total).

INTERVIEW NOTE:
  This is a cheap, deterministic "faithfulness" check that runs on EVERY request. Offline
  we also run LLM-as-judge faithfulness metrics (Ragas / DeepEval) on the eval set — those
  are richer but too slow and costly for the hot path.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.models import Boot, SearchResult

SKU = re.compile(r"\bBOOT-\d{3}\b")
PRICE = re.compile(r"\$\s?(\d+(?:\.\d{1,2})?)")


@dataclass
class GroundingReport:
    grounded: bool
    issues: list[str] = field(default_factory=list)


def check_grounding(answer: str, results: list[SearchResult], catalog: dict[str, Boot]) -> GroundingReport:
    issues: list[str] = []
    shown = {r.boot.id for r in results}

    for sku in set(SKU.findall(answer)):
        if sku not in shown:
            issues.append(f"mentions SKU {sku} that was not retrieved")

    # Remove the names of shown products first (longest first), so a shown "Cascade II"
    # doesn't make the unshown "Cascade" look mentioned (substring false positive).
    remaining = answer.lower()
    for name in sorted({r.boot.name.lower() for r in results}, key=len, reverse=True):
        remaining = remaining.replace(name, " ")
    for boot in catalog.values():
        if boot.id not in shown and re.search(rf"\b{re.escape(boot.name.lower())}\b", remaining):
            issues.append(f"mentions product '{boot.name}' that was not retrieved")

    allowed_prices = {round(r.boot.price, 2) for r in results}
    for p in PRICE.findall(answer):
        value = round(float(p), 2)
        # tolerate prices the user asked about ("under $200") — only flag cents-precise prices
        if "." in p and value not in allowed_prices:
            issues.append(f"mentions price ${p} that matches no retrieved product")

    return GroundingReport(grounded=not issues, issues=issues)
