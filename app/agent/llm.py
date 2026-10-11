"""
Thin LLM client around the Anthropic SDK, with the production concerns built in:
  * timeouts + retries (SDK `max_retries`) — an LLM is a flaky network dependency
  * token usage and $ cost tracking per call (Cost & Latency Optimisation)
  * graceful degradation: no API key or an error -> returns None, and the agent falls
    back to deterministic rules/templates. Search must work even if the LLM is down.

INTERVIEW NOTE — cost/latency levers:
  * Small fast model (Haiku class) for query understanding: it's a classification/
    extraction task. Use a bigger model only where quality matters (answer writing).
  * Cache: identical queries hit Redis and never call the LLM (app/cache).
  * Short prompts: compact catalog context, max_tokens caps.
  * The LLM is OFF the critical path for explicit modes (keyword/semantic/image):
    those skip query understanding entirely.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from app.core.config import Settings
from app.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class LLMUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    errors: list[str] = field(default_factory=list)


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = settings.llm_model
        self.client = None
        if settings.anthropic_api_key:
            from anthropic import Anthropic

            self.client = Anthropic(api_key=settings.anthropic_api_key,
                                    timeout=settings.llm_timeout_s, max_retries=2)

    @property
    def available(self) -> bool:
        return self.client is not None

    def complete(self, system: str, user: str, usage: LLMUsage, max_tokens: int | None = None) -> str | None:
        if not self.client:
            return None
        t0 = time.perf_counter()
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens or self.settings.llm_max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as e:  # network, rate limit, bad key ... never crash the search
            usage.errors.append(type(e).__name__)
            log.warning(f"LLM call failed: {e}")
            return None
        finally:
            usage.latency_ms += (time.perf_counter() - t0) * 1000
        usage.calls += 1
        usage.input_tokens += resp.usage.input_tokens
        usage.output_tokens += resp.usage.output_tokens
        usage.cost_usd += (resp.usage.input_tokens * self.settings.llm_input_cost_per_mtok
                           + resp.usage.output_tokens * self.settings.llm_output_cost_per_mtok) / 1e6
        return "".join(block.text for block in resp.content if block.type == "text")

    def complete_json(self, system: str, user: str, usage: LLMUsage) -> dict | None:
        text = self.complete(system, user, usage, max_tokens=300)
        if not text:
            return None
        match = re.search(r"\{.*\}", text, re.S)   # tolerate stray prose / code fences
        try:
            return json.loads(match.group()) if match else None
        except json.JSONDecodeError:
            usage.errors.append("JSONDecodeError")
            return None
