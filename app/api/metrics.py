"""
Prometheus metrics (scraped at GET /metrics, visualised in Grafana).

The four golden signals + AI-specific ones:
  latency      : http_request_duration_seconds, agent_node_duration_seconds
  traffic      : http_requests_total, searches_total{mode,understood_by}
  errors       : http_requests_total{status="5xx"}, llm_errors_total
  saturation   : (from the container runtime / k8s)
  AI-specific  : llm_tokens_total, llm_cost_usd_total, cache_hits/misses,
                 zero_result_searches_total, answer_grounding_failures_total,
                 guardrail_blocks_total, feedback_total{rating}

INTERVIEW NOTE: "zero_result_searches" and "grounding_failures" are the metrics a product
team actually alerts on — they measure quality, not just uptime.
"""
from prometheus_client import Counter, Histogram

HTTP_REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "path", "status"])
HTTP_LATENCY = Histogram("http_request_duration_seconds", "HTTP latency", ["method", "path"],
                         buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10))
SEARCHES = Counter("searches_total", "Searches", ["mode", "understood_by"])
NODE_LATENCY = Histogram("agent_node_duration_seconds", "Agent node latency", ["node"],
                         buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5))
CACHE_HITS = Counter("cache_hits_total", "Cache hits")
CACHE_MISSES = Counter("cache_misses_total", "Cache misses")
ZERO_RESULTS = Counter("zero_result_searches_total", "Searches that returned nothing")
GROUNDING_FAILURES = Counter("answer_grounding_failures_total", "LLM answers rejected by output guard")
GUARDRAIL_BLOCKS = Counter("guardrail_blocks_total", "Requests blocked by input guard")
LLM_TOKENS = Counter("llm_tokens_total", "LLM tokens", ["direction"])
LLM_COST = Counter("llm_cost_usd_total", "Estimated LLM spend (USD)")
LLM_ERRORS = Counter("llm_errors_total", "LLM call errors", ["type"])
FEEDBACK = Counter("feedback_total", "User feedback", ["rating"])
