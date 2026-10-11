"""
Offline GENERATION evaluation + prompt A/B test.

For each golden query and each prompt variant (A concise, B consultative) we generate an
answer from the retrieved products and score it on:
  * faithfulness      : are all claims supported by the retrieved catalog context?
  * answer_relevancy  : does it address the shopper's need?
  * grounding_check   : our deterministic output guard (SKUs, names, prices) — always on

Scoring backends, best available wins:
  1. Ragas  (pip install ragas)       -> ragas.metrics faithfulness / answer_relevancy
  2. LLM-as-judge with Claude          -> rubric prompt below, JSON scores 1-5
  3. No LLM key                        -> only the deterministic grounding check

INTERVIEW NOTES
  * LLM-as-judge pitfalls: position bias, verbosity bias, self-preference. Mitigations: a
    rubric with anchored scores, temperature-free short JSON output, judge model different
    from (or at least blind to) the generator, and spot-checking judge labels by a human.
  * Tools in this space: Ragas, DeepEval (pytest-style asserts), TruLens (feedback
    functions + tracing). The scoring here is shaped so any of them can be dropped in.

Run:  ANTHROPIC_API_KEY=... python -m app.evaluation.llm_eval
"""
from __future__ import annotations

import json
import statistics

from app.agent.graph import BootSearchAgent
from app.agent.llm import LLMClient, LLMUsage
from app.agent.prompts import format_catalog_context
from app.core.config import get_settings
from app.core.container import get_engine

JUDGE_SYSTEM = """You are a strict evaluator of a shopping assistant. Score the ANSWER using only the CONTEXT.
Return JSON only: {"faithfulness": 1-5, "relevancy": 1-5, "reason": "<one sentence>"}
faithfulness: 5 = every product, price and feature claim appears in CONTEXT; 1 = invented facts.
relevancy:    5 = directly answers the shopper's request with fitting products; 1 = off-topic."""


def judge(llm: LLMClient, query: str, context: str, answer: str, usage: LLMUsage) -> dict | None:
    return llm.complete_json(JUDGE_SYSTEM, f"QUERY: {query}\nCONTEXT:\n{context}\nANSWER:\n{answer}", usage)


def try_ragas(rows: list[dict]) -> dict | None:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, faithfulness
    except ImportError:
        return None
    ds = Dataset.from_list([{"question": r["query"], "answer": r["answer"], "contexts": [r["context"]]} for r in rows])
    return evaluate(ds, metrics=[faithfulness, answer_relevancy]).to_pandas().mean(numeric_only=True).to_dict()


def main() -> None:
    s = get_settings()
    engine = get_engine()
    llm = LLMClient(s)
    agent = BootSearchAgent(engine, llm, s)
    golden = json.loads((s.data_dir / "eval" / "golden_set.json").read_text())["queries"]
    usage = LLMUsage()

    report = {}
    for variant in ("A", "B"):
        rows = []
        for q in golden:
            out = agent.run(query=q["query"], variant=variant, top_k=5)
            ctx = format_catalog_context(out.get("results", [])[:5])
            row = {"query": q["query"], "answer": out["answer"], "context": ctx, "grounded": out.get("grounded", True)}
            if llm.available:
                row["judge"] = judge(llm, q["query"], ctx, out["answer"], usage)
            rows.append(row)
        summary = {"grounding_pass_rate": round(sum(r["grounded"] for r in rows) / len(rows), 3)}
        judged = [r["judge"] for r in rows if r.get("judge")]
        if judged:
            summary["faithfulness_avg"] = round(statistics.mean(j["faithfulness"] for j in judged), 2)
            summary["relevancy_avg"] = round(statistics.mean(j["relevancy"] for j in judged), 2)
        if (ragas := try_ragas(rows)) is not None:
            summary["ragas"] = ragas
        report[variant] = {"summary": summary, "rows": rows}
        print(f"Variant {variant}: {summary}")

    if not llm.available:
        print("No ANTHROPIC_API_KEY: answers are template-based, so only the grounding check ran.")
    print(f"Judge cost: ${usage.cost_usd:.4f} over {usage.calls} calls")
    (s.data_dir / "eval" / "llm_report.json").write_text(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
