# Architecture & Design Decisions

## 1. Request lifecycle (what happens on one search)

1. **UI → API.** Streamlit posts `{query, image_base64?, mode, filters, top_k, session_id}` to `POST /search`.
2. **Middleware** assigns a request id (in every log line and the `x-request-id` header) and starts timing.
3. **Security dependency:** optional API key, per-IP rate limit.
4. **Validation:** image size/format/decompression-bomb check, "query or image required".
5. **Cache lookup:** key = hash(query, mode, filters, top_k, prompt variant, image hash, prompt version,
   catalog size). Hit → return in ~1 ms.
6. **Agent graph** (`app/agent/graph.py`):
   - `guard_input` — prompt-injection score, PII redaction, length limit. Blocked → done, no LLM call.
   - `understand` — Claude returns strict JSON (filters, rewritten query, intent, clarification);
     on no key / timeout / bad JSON the rule parser does it. Then **tool selection and weights**.
   - `retrieve` — selected retrievers run in parallel threads; results fused with weighted RRF;
     hard filters enforced on every result.
   - `reflect` — zero results → relax the most restrictive *inferred* filter (never a user-set one),
     go back to `retrieve`. Bounded to 3 retries.
   - `generate` — Claude writes an answer from the top-5 products only (variant A or B);
     template fallback.
   - `guard_output` — every SKU, product name and cents-precise price in the answer must come from
     the retrieved set; otherwise the answer is replaced by the template.
7. **Metrics + logs:** node latencies, zero-result count, grounding failures, tokens, cost; the redacted
   query and outcome are written to the SQL `search_logs` table.
8. **Cache write** (not for blocked requests), response returned with the full trace.

## 2. Indexing pipeline

```
catalog.json / .csv / REST  →  loaders  →  clean + validate + dedup  →  chunk (section-aware, recursive)
      →  boots.jsonl, chunks.jsonl, rejected.json, manifest.json
      →  embed in batches  →  Qdrant: boot_text_chunks (1 point per chunk), boot_images (1 point per image)
```

* **Chunk-level retrieval, product-level results.** A boot has a profile chunk (attributes as a
  sentence, incl. SKU), description chunk(s) and review chunk(s). Vector search over chunks, then
  `MAX` score per boot. This lets a review sentence ("kept my feet dry all day") surface a product,
  and gives the user an *evidence* snippet explaining the match.
* **Idempotency.** Point ids are `uuid5(chunk_id)`, so re-running ingestion upserts.
* **Staleness detection.** `manifest.json` stores the raw-data SHA-256, chunk params and
  `PIPELINE_VERSION`; the API re-runs ETL and re-indexes when any of them change, and
  `ensure_collections` recreates a collection when the embedding dimension changes.

## 3. Key decisions and trade-offs

| Decision | Why | Trade-off / alternative |
|---|---|---|
| Hybrid BM25 + dense + image | Each covers the others' blind spots: exact tokens vs. paraphrase vs. visuals | More moving parts; fusion must be tuned |
| RRF fusion | Score scales differ wildly (BM25 unbounded, cosine ∈ [-1,1]); RRF uses ranks only | Ignores score margins; a cross-encoder re-ranker on top-50 is better but slower |
| Filters pushed into Qdrant | Filterable HNSW returns k *valid* results; post-filtering can return 0 | BM25 is in-process, so it post-filters (fine at this scale) |
| Separate text and image collections | Different dims, different models, independent re-indexing | A single collection with **named vectors** is the alternative; better when one point = one product with several vectors |
| MiniLM for text, CLIP for images and text→image | Use each model for what it was trained on (CLIP's text tower is caption-length) | Two models to serve |
| LangGraph state machine, not a free ReAct loop | Bounded latency and cost, testable nodes, explainable trace | Less flexible than an open-ended agent |
| Rule-based fallback for every LLM step | Availability, cost, tests, and a baseline to prove the LLM's value | Rules need maintenance (synonym lists) |
| Deterministic grounding check on every answer | Cheap, catches the most damaging hallucinations (wrong product/price) | Doesn't verify free-text claims; offline LLM-judge covers that |
| Redis response cache with versioned keys | Shared across replicas; invalidation by key versioning | Exact-match only; semantic caching would raise hit rate |
| SQLAlchemy for logs/feedback | SQLite locally, Postgres in prod, same code | — |
| `lite` embedding backend | Tests/CI/serverless with no torch and no downloads | Lexical only, not semantic — clearly labelled |

## 4. Scaling to a real catalog (100k–10M products)

* **Qdrant:** server/cluster with sharding and replication; scalar or binary **quantization** to cut
  memory 4–32×; tune `hnsw_ef` for the recall/latency target; payload indexes on filter fields (already
  created).
* **Keyword search:** move BM25 into the engine — Qdrant **sparse vectors** with the IDF modifier (one
  query, server-side fusion via the Query API) or OpenSearch/Elasticsearch.
* **Embeddings:** GPU inference service (Triton / TEI) with dynamic batching; embed at ingestion, not
  per request; cache query embeddings.
* **Ingestion:** catalog changes as events (Kafka/Kinesis) → consumer embeds and upserts one product;
  nightly batch reconciliation.
* **LLM:** stream the answer (SSE) so search results render first; route extraction to a small model;
  self-host with **vLLM** if volume makes per-token pricing worse than GPUs.
* **Re-ranking:** cross-encoder (e.g. `bge-reranker`) over the fused top-50, or learning-to-rank on
  click/feedback data.
* **API:** stateless replicas behind a load balancer, HPA on CPU/latency, rate limiting at the gateway.

## 5. Failure modes and what happens

| Failure | Behaviour |
|---|---|
| LLM down / slow / bad JSON | Rule parser + template answer; `llm_errors_total` increments |
| LLM hallucinates a product or price | Output guard replaces the answer; `answer_grounding_failures_total` |
| Redis down | In-memory cache fallback (startup) or cache miss (runtime) — search still works |
| Qdrant empty / new embedding model | Startup re-index; dimension mismatch → collection recreated |
| No results | Agent relaxes inferred filters and says so; `zero_result_searches_total` |
| Malicious query | Input guard blocks before any LLM call; `guardrail_blocks_total` |
| Malicious catalog text (indirect injection) | Prompt marks `<catalog>` as data; UI HTML-escapes all catalog text |
| Oversized / malformed image | 422 with a clear message; decompression-bomb limit |
