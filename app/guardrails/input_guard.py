"""
Input guardrails — run BEFORE anything reaches the LLM or the logs.

1. Validation      : empty / too long / bad image (format, size, decompression bomb)
2. Prompt injection: heuristic detector for "ignore previous instructions"-style attacks
3. PII redaction   : emails, phone numbers, card numbers (Luhn-checked), SSNs

INTERVIEW NOTES
  * Defence in depth. This regex layer is fast and catches the obvious; the real
    protections are architectural:
      - the LLM never has tools that can do damage (read-only search, no SQL, no shell),
      - retrieved catalog text is wrapped in <catalog> tags and the prompt says it is
        DATA, not instructions (indirect prompt injection via product reviews!),
      - the output guard verifies the answer against retrieved data.
    For stronger detection add a classifier (e.g. a fine-tuned DeBERTa prompt-injection
    model from Hugging Face, or Llama Guard / Prompt Guard) behind the same interface.
  * Privacy / GDPR: data minimisation — we redact PII BEFORE logging, caching or sending
    text to a third-party LLM. Search doesn't need someone's phone number.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

from PIL import Image

Image.MAX_IMAGE_PIXELS = 40_000_000  # decompression-bomb protection

INJECTION_PATTERNS = [
    (r"ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|prompts?|rules)", 0.9),
    (r"disregard (all |the )?(previous|prior|above|system)", 0.9),
    (r"(reveal|print|show|repeat|output) (me )?(your|the) (system|hidden|initial) (prompt|instructions)", 0.9),
    (r"you are now|act as (an? )?(unfiltered|jailbroken|dan)|developer mode|do anything now", 0.8),
    (r"</?(system|assistant|instructions?)>", 0.7),
    (r"\b(jailbreak|prompt injection)\b", 0.5),
    (r"(api[_ ]?key|password|secret|credentials)", 0.4),
    (r"(rm -rf|drop table|;--|<script)", 0.6),
]
BLOCK_THRESHOLD = 0.8

PII_PATTERNS = {
    "EMAIL": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "PHONE": re.compile(r"(?<!\d)(\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)"),
}
CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")


def _luhn_ok(number: str) -> bool:
    digits = [int(d) for d in re.sub(r"\D", "", number)][::-1]
    total = sum(d if i % 2 == 0 else (d * 2 - 9 if d * 2 > 9 else d * 2) for i, d in enumerate(digits))
    return len(digits) >= 13 and total % 10 == 0


@dataclass
class InputCheck:
    allowed: bool
    text: str                       # redacted text, safe to log and send to the LLM
    risk_score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    pii_found: list[str] = field(default_factory=list)


def redact_pii(text: str) -> tuple[str, list[str]]:
    found: list[str] = []
    for m in CARD.finditer(text):
        if _luhn_ok(m.group()):
            text = text.replace(m.group(), "[REDACTED_CARD]")
            found.append("CARD")
    for label, pattern in PII_PATTERNS.items():
        if pattern.search(text):
            found.append(label)
            text = pattern.sub(f"[REDACTED_{label}]", text)
    return text, found


def injection_risk(text: str) -> tuple[float, list[str]]:
    lowered = text.lower()
    score, reasons = 0.0, []
    for pattern, weight in INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            score = max(score, weight) + (0.1 if score else 0.0)   # several signals stack a little
            reasons.append(f"matched: {pattern[:40]}")
    return min(score, 1.0), reasons


def check_query(text: str, max_chars: int = 500) -> InputCheck:
    text = (text or "").strip()
    if len(text) > max_chars:
        return InputCheck(False, text[:max_chars], 0.0, [f"query longer than {max_chars} characters"])
    redacted, pii = redact_pii(text)
    risk, reasons = injection_risk(redacted)
    return InputCheck(allowed=risk < BLOCK_THRESHOLD, text=redacted, risk_score=risk, reasons=reasons, pii_found=pii)


def load_image(data: bytes, max_bytes: int) -> Image.Image:
    """Validate and decode an uploaded image. Raises ValueError with a user-safe message."""
    if len(data) > max_bytes:
        raise ValueError(f"Image too large ({len(data) // 1024} KB > {max_bytes // 1024} KB)")
    try:
        img = Image.open(io.BytesIO(data))
        if img.format not in {"PNG", "JPEG", "WEBP"}:
            raise ValueError(f"Unsupported image format: {img.format}")
        img.load()
    except (Image.DecompressionBombError, OSError) as e:
        raise ValueError(f"Invalid image: {e}") from e
    return img.convert("RGB")
