"""
The agent — a LangGraph state machine that plans, retrieves, reflects, and answers.

                  ┌──────────────┐
      request ──► │ guard_input  │── blocked ──────────────────────────────► END
                  └──────┬───────┘
                         ▼
                  ┌──────────────┐   LLM (JSON) or rule parser:
                  │  understand  │   filters, rewritten query, intent, which tools
                  └──────┬───────┘
                         ▼
                  ┌──────────────┐   keyword | semantic | image | text_to_image
          ┌─────► │   retrieve   │   (run in parallel) + weighted RRF fusion
          │       └──────┬───────┘
          │              ▼
          │       ┌──────────────┐   0 results?  -> relax the most restrictive filter
          └─retry─│   reflect    │   (max 2 retries) and search again
                  └──────┬───────┘
                         ▼
                  ┌──────────────┐   grounded answer from retrieved products only
                  │   generate   │   (LLM, prompt variant A/B; template fallback)
                  └──────┬───────┘
                         ▼
                  ┌──────────────┐   hallucination check: SKUs, names, prices
                  │ guard_output │   must come from retrieved products
                  └──────┬───────┘
                         ▼
                        END

DISCUSSION NOTES — what makes this "agentic" rather than a fixed RAG chain?
  * Tool selection: the agent decides which retrievers to call and how to weight them
    from the query (SKU -> keyword-heavy; image uploaded -> image-heavy; visual words ->
    add CLIP text->image).
  * Self-correction loop: `reflect` inspects the results and re-plans (relaxes filters)
    when the search comes back empty, and tells the user what it relaxed.
  * Human-in-the-loop: vague queries produce a clarifying question; thumbs-down feedback
    goes to a review queue (/feedback/review).
  * Why a graph (LangGraph) instead of a free-form ReAct loop? Predictable latency and
    cost (bounded steps), every node is testable, and the trace is explainable. The LLM
    is used where it adds value (understanding, writing), not for control flow.
"""
from __future__ import annotations

import time
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from app.agent.llm import LLMClient, LLMUsage
from app.agent.prompts import ANSWER_SYSTEM, PROMPT_VERSION, QUERY_UNDERSTANDING_SYSTEM, format_catalog_context
from app.agent.query_parser import parse_query
from app.core.config import Settings
from app.core.models import Filters, SearchResult
from app.guardrails.input_guard import check_query
from app.guardrails.output_guard import check_grounding
from app.retrieval.search_engine import RetrievalPlan, SearchEngine

RELAX_ORDER = ["color", "max_price", "min_price", "waterproof", "safety_toe", "category"]
MAX_RETRIES = 3


class AgentState(TypedDict, total=False):
    # inputs
    query: str
    images: list[Any]             # 0..N query photos (PIL.Image)
    mode: str                     # auto | keyword | semantic | image | hybrid
    user_filters: Filters         # explicit filters from the UI (always win)
    top_k: int
    variant: str                  # prompt A/B variant
    # working state
    safe_query: str
    filters: Filters
    search_text: str
    intent: str
    tools: list[str]
    weights: dict[str, float]
    results: list[SearchResult]
    retriever_counts: dict[str, int]
    attempts: int
    relaxed: list[str]
    # outputs
    blocked: bool
    answer: str
    clarifying_question: str | None
    grounded: bool
    grounding_issues: list[str]
    understood_by: str            # "llm" | "rules" | "explicit-mode"
    trace: list[dict]
    usage: LLMUsage


def _as_list(images) -> list:
    """Accept None, one PIL image, or a list of them."""
    if images is None:
        return []
    return list(images) if isinstance(images, (list, tuple)) else [images]


def _trace(state: AgentState, node: str, t0: float, **detail) -> list[dict]:
    return [*state.get("trace", []), {"node": node, "ms": round((time.perf_counter() - t0) * 1000, 2), **detail}]


class BootSearchAgent:
    def __init__(self, engine: SearchEngine, llm: LLMClient, settings: Settings):
        self.engine, self.llm, self.settings = engine, llm, settings
        self.graph = self._build()

    # ================================================================ nodes
    def guard_input(self, state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        check = check_query(state.get("query", ""), self.settings.max_query_chars)
        if not check.allowed:
            return {"blocked": True, "safe_query": check.text, "results": [],
                    "answer": "Sorry, I can only help with finding boots. Please describe what you're looking for.",
                    "trace": _trace(state, "guard_input", t0, allowed=False, risk=check.risk_score, reasons=check.reasons)}
        if not check.text and not state.get("images"):
            return {"blocked": True, "safe_query": "", "results": [],
                    "answer": "Type a description or upload a boot photo to start searching.",
                    "trace": _trace(state, "guard_input", t0, allowed=False, reasons=["empty request"])}
        return {"blocked": False, "safe_query": check.text,
                "trace": _trace(state, "guard_input", t0, allowed=True, risk=check.risk_score, pii_redacted=check.pii_found)}

    def understand(self, state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        query, images, mode = state["safe_query"], state.get("images") or [], state.get("mode", "auto")
        usage = state.get("usage") or LLMUsage()
        available = self.engine.available_tools()

        parsed = parse_query(query) if query else {"search_text": "", "filters": Filters(), "intent": "visual",
                                                   "needs_clarification": False, "clarifying_question": None}
        understood_by = "rules"
        # Only spend LLM tokens when the agent is choosing the strategy (auto mode) and there is text
        if mode == "auto" and query and self.llm.available:
            data = self.llm.complete_json(QUERY_UNDERSTANDING_SYSTEM, f"<query>{query}</query>", usage)
            if data:
                try:
                    parsed = {
                        "search_text": data.get("search_text") or query,
                        "filters": Filters(**{k: v for k, v in (data.get("filters") or {}).items() if v is not None}),
                        "intent": data.get("intent", "describe"),
                        "needs_clarification": bool(data.get("needs_clarification")),
                        "clarifying_question": data.get("clarifying_question"),
                    }
                    understood_by = "llm"
                except Exception:   # schema violation -> keep the rule-based parse
                    usage.errors.append("LLMSchemaError")

        # Explicit UI filters always override inferred ones
        filters = parsed["filters"].model_copy(update={
            k: v for k, v in (state.get("user_filters") or Filters()).model_dump().items() if v is not None})

        # ---- tool selection + weights (the "planning" step)
        if mode == "keyword":
            tools, weights = ["keyword"], {}
        elif mode == "semantic":
            tools, weights = ["semantic"] + (["text_to_image"] if "text_to_image" in available else []), {"text_to_image": 0.5}
        elif mode == "image":
            tools, weights = ["image"], {}
        else:  # auto / hybrid
            tools, weights = [], {}
            if images:
                tools.append("image")
                weights["image"] = 1.5 if query else 1.0      # the photos are the strongest signal
            if query:
                tools += ["keyword", "semantic"]
                if parsed["intent"] == "keyword_lookup":
                    weights.update(keyword=2.0, semantic=0.5)  # SKU / exact term: trust BM25
                if "text_to_image" in available and (parsed["intent"] == "visual" or filters.color):
                    tools.append("text_to_image")
                    weights["text_to_image"] = 0.7
        if mode != "auto":
            understood_by = f"explicit-mode:{mode}" if mode != "hybrid" else understood_by

        return {
            "filters": filters, "search_text": parsed["search_text"] or query, "intent": parsed["intent"],
            "tools": tools, "weights": weights, "usage": usage, "attempts": 0, "relaxed": [],
            "clarifying_question": parsed["clarifying_question"] if parsed["needs_clarification"] and not images else None,
            "understood_by": understood_by,
            "trace": _trace(state, "understand", t0, by=understood_by, intent=parsed["intent"], tools=tools,
                            query_images=len(images),
                            weights=weights, filters=filters.model_dump(exclude_none=True),
                            search_text=parsed["search_text"]),
        }

    def retrieve(self, state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        plan = RetrievalPlan(query=state["search_text"], images=state.get("images") or [], tools=state["tools"],
                             weights=state["weights"], filters=state["filters"], top_k=state.get("top_k", 8))
        raw, latencies = self.engine.retrieve(plan)
        results = self.engine.fuse(raw, plan)
        counts = {k: len(v) for k, v in raw.items()}
        return {"results": results, "retriever_counts": counts, "attempts": state.get("attempts", 0) + 1,
                "trace": _trace(state, "retrieve", t0, attempt=state.get("attempts", 0) + 1, per_tool_hits=counts,
                                per_tool_ms={k: round(v, 2) for k, v in latencies.items()}, fused=len(results))}

    def reflect(self, state: AgentState) -> AgentState:
        """If nothing matched, relax the most restrictive remaining filter and retry."""
        t0 = time.perf_counter()
        if state["results"] or state["attempts"] > MAX_RETRIES:
            return {"trace": _trace(state, "reflect", t0, decision="accept", results=len(state["results"]))}
        filters, relaxed = state["filters"], list(state.get("relaxed", []))
        user_set = {k for k, v in (state.get("user_filters") or Filters()).model_dump().items() if v is not None}
        for field in RELAX_ORDER:
            if getattr(filters, field) is not None and field not in user_set:
                relaxed.append(field)
                return {"filters": filters.model_copy(update={field: None}), "relaxed": relaxed,
                        "trace": _trace(state, "reflect", t0, decision="retry", relaxed=field)}
        return {"trace": _trace(state, "reflect", t0, decision="give_up")}

    def generate(self, state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        results, usage = state["results"], state["usage"]
        variant = state.get("variant", "A")
        answer = None
        if results and self.llm.available and state["safe_query"]:
            n_photos = len(state.get("images") or [])
            photo_note = f"The shopper also uploaded {n_photos} reference photo(s)." if n_photos else ""
            user_msg = (f"Shopper request: <query>{state['safe_query']}</query>\n"
                        f"{photo_note}\n"
                        f"{format_catalog_context(results[:5])}")
            answer = self.llm.complete(ANSWER_SYSTEM.get(variant, ANSWER_SYSTEM["A"]), user_msg, usage)
        source = "llm" if answer else "template"
        if not answer:
            answer = self.template_answer(state)
        return {"answer": answer.strip(),
                "trace": _trace(state, "generate", t0, source=source, variant=variant, prompt_version=PROMPT_VERSION)}

    def guard_output(self, state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        report = check_grounding(state["answer"], state["results"], self.engine.catalog)
        answer = state["answer"] if report.grounded else self.template_answer(state)
        return {"answer": answer, "grounded": report.grounded, "grounding_issues": report.issues,
                "trace": _trace(state, "guard_output", t0, grounded=report.grounded, issues=report.issues)}

    # ================================================================ helpers
    @staticmethod
    def template_answer(state: AgentState) -> str:
        """Deterministic, always-grounded answer used when the LLM is off or fails checks."""
        results = state.get("results", [])
        relaxed = state.get("relaxed", [])
        if not results:
            return "I couldn't find boots matching that. Try fewer constraints or a different description."
        top = results[:3]
        parts = [f"{r.boot.name} ({r.boot.id}, ${r.boot.price:.2f})" for r in top]
        msg = f"Top matches: {'; '.join(parts)}."
        if relaxed:
            msg += f" No exact match, so I relaxed: {', '.join(relaxed)}."
        return msg

    # ================================================================ graph
    def _build(self):
        g = StateGraph(AgentState)
        for name in ("guard_input", "understand", "retrieve", "reflect", "generate", "guard_output"):
            g.add_node(name, getattr(self, name))
        g.set_entry_point("guard_input")
        g.add_conditional_edges("guard_input", lambda s: "end" if s.get("blocked") else "go",
                                {"end": END, "go": "understand"})
        g.add_edge("understand", "retrieve")
        g.add_edge("retrieve", "reflect")
        g.add_conditional_edges(
            "reflect",
            lambda s: "retry" if (not s["results"] and s["attempts"] <= MAX_RETRIES
                                  and s["trace"][-1].get("decision") == "retry") else "done",
            {"retry": "retrieve", "done": "generate"},
        )
        g.add_edge("generate", "guard_output")
        g.add_edge("guard_output", END)
        return g.compile()

    def run(self, query: str = "", images=None, mode: str = "auto", user_filters: Filters | None = None,
            top_k: int = 8, variant: str = "A") -> AgentState:
        return self.graph.invoke({
            "query": query, "images": _as_list(images), "mode": mode, "user_filters": user_filters or Filters(),
            "top_k": top_k, "variant": variant, "trace": [], "usage": LLMUsage(),
        })

    def mermaid(self) -> str:
        """Diagram of the compiled graph — handy to paste into the README or show in the UI."""
        return self.graph.get_graph().draw_mermaid()
