"""
Offline retrieval evaluation — compare keyword vs semantic vs hybrid vs the full agent.

Metrics (all @k, averaged over the golden queries):
  Precision@5 : of the top 5, how many are relevant?           (what the user SEES)
  Recall@10   : of all relevant boots, how many are in the top 10?
  MRR         : 1 / rank of the first relevant result          (how fast the user finds one)
  nDCG@10     : rank-aware quality; rewards relevant items near the top
  p50 / p95   : latency percentiles in ms

INTERVIEW NOTES
  * "How do you know your RAG system is good?" -> you build a golden set and measure,
    BEFORE and AFTER every change (chunk size, embedding model, fusion weights, prompts).
  * Retrieval quality bounds answer quality: if the right product isn't retrieved, no
    prompt can fix it. So evaluate retrieval separately from generation (llm_eval.py).
  * This runs in CI as a quality gate (tests/test_eval_gate.py): a PR that drops hybrid
    nDCG below the threshold fails the build.

Run:  python -m app.evaluation.retrieval_eval
"""
from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path

from app.core.config import get_settings
from app.core.container import get_engine
from app.retrieval.search_engine import RetrievalPlan


def relevant_ids(rule: dict, catalog) -> set[str]:
    return {b.id for b in catalog.values() if all(getattr(b, k) == v for k, v in rule.items())}


def precision_at(ranked, rel, k):
    return sum(1 for x in ranked[:k] if x in rel) / k


def recall_at(ranked, rel, k):
    return sum(1 for x in ranked[:k] if x in rel) / max(len(rel), 1)


def mrr(ranked, rel):
    return next((1 / i for i, x in enumerate(ranked, 1) if x in rel), 0.0)


def ndcg_at(ranked, rel, k):
    dcg = sum(1 / math.log2(i + 1) for i, x in enumerate(ranked[:k], 1) if x in rel)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(rel), k) + 1))
    return dcg / ideal if ideal else 0.0


def evaluate(systems: list[str] | None = None, golden_path: Path | None = None) -> dict:
    s = get_settings()
    engine = get_engine()
    golden = json.loads((golden_path or s.data_dir / "eval" / "golden_set.json").read_text())["queries"]

    from app.agent.graph import BootSearchAgent
    from app.agent.llm import LLMClient
    agent = BootSearchAgent(engine, LLMClient(s), s)

    def run_system(name: str, query: str) -> list[str]:
        if name == "agent":    # full pipeline incl. filter extraction + reflection
            return [r.boot.id for r in agent.run(query=query, top_k=10).get("results", [])]
        tools = {"keyword": ["keyword"], "semantic": ["semantic"], "hybrid": ["keyword", "semantic"]}[name]
        return [r.boot.id for r in engine.run(RetrievalPlan(query=query, tools=tools, top_k=10))]

    systems = systems or ["keyword", "semantic", "hybrid", "agent"]
    report: dict = {"embedder": engine.text_embedder.name, "n_queries": len(golden), "systems": {}, "per_query": []}
    per_query_rows = {q["query"]: {"query": q["query"]} for q in golden}

    for name in systems:
        scores = {"P@5": [], "R@10": [], "MRR": [], "nDCG@10": []}
        latencies = []
        for q in golden:
            rel = relevant_ids(q["relevant"], engine.catalog)
            t0 = time.perf_counter()
            ranked = run_system(name, q["query"])
            latencies.append((time.perf_counter() - t0) * 1000)
            scores["P@5"].append(precision_at(ranked, rel, 5))
            scores["R@10"].append(recall_at(ranked, rel, 10))
            scores["MRR"].append(mrr(ranked, rel))
            scores["nDCG@10"].append(ndcg_at(ranked, rel, 10))
            per_query_rows[q["query"]][name] = round(scores["nDCG@10"][-1], 3)
        lat = sorted(latencies)
        report["systems"][name] = {
            **{k: round(statistics.mean(v), 3) for k, v in scores.items()},
            "p50_ms": round(lat[len(lat) // 2], 2),
            "p95_ms": round(lat[min(len(lat) - 1, int(len(lat) * 0.95))], 2),
        }
    report["per_query"] = list(per_query_rows.values())
    return report


def to_markdown(report: dict) -> str:
    cols = ["P@5", "R@10", "MRR", "nDCG@10", "p50_ms", "p95_ms"]
    lines = [f"Embedder: `{report['embedder']}` — {report['n_queries']} golden queries\n",
             "| system | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, row in report["systems"].items():
        lines.append(f"| {name} | " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    rep = evaluate()
    out = get_settings().data_dir / "eval" / "report.json"
    out.write_text(json.dumps(rep, indent=2))
    print(to_markdown(rep))
    print(f"\nPer-query nDCG@10 and full report: {out}")
