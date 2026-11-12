"""
SQL store for search logs and user feedback — the "Continuous Improvement Loop".

  search_logs : one row per search (redacted query, tools, results, latency, cost, variant)
  feedback    : thumbs up/down (+ optional comment, optional boot) linked by request_id

How it closes the loop (discussion talking points):
  1. Human-in-the-loop review: /feedback/review lists thumbs-down searches for a
     merchandiser to inspect; fixes go into synonyms, chunking, or the golden eval set.
  2. A/B testing: /feedback/stats gives the thumbs-up rate per prompt variant.
  3. New eval data: reviewed failures become new rows in data/eval/golden_set.json,
     so every fixed bug becomes a regression test.
  4. Learning to rank (future): clicks/likes become training labels for a re-ranker.

SQLAlchemy Core keeps it database-agnostic: SQLite locally, Postgres in production
(set DATABASE_URL=postgresql+psycopg://...). Note: serverless containers have no
persistent disk, so on Vercel use a hosted Postgres (Neon, Supabase, RDS).

Privacy: we store the REDACTED query only, and no user identifiers beyond a random
session id. Retention can be enforced with a scheduled DELETE (GDPR storage limitation).
"""
from __future__ import annotations

import json
import time

from sqlalchemy import Column, Float, Integer, MetaData, String, Table, Text, create_engine, func, insert, select
from sqlalchemy.engine import Engine

metadata = MetaData()

search_logs = Table(
    "search_logs", metadata,
    Column("request_id", String(64), primary_key=True),
    Column("ts", Float, nullable=False),
    Column("session_id", String(64)),
    Column("query", Text),
    Column("mode", String(16)),
    Column("has_image", Integer),
    Column("variant", String(4)),
    Column("tools", String(128)),
    Column("filters", Text),
    Column("result_ids", Text),
    Column("n_results", Integer),
    Column("latency_ms", Float),
    Column("cost_usd", Float),
    Column("grounded", Integer),
    Column("understood_by", String(32)),
    Column("cache_hit", Integer),
)

feedback = Table(
    "feedback", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", Float, nullable=False),
    Column("request_id", String(64), index=True),
    Column("rating", Integer, nullable=False),     # +1 / -1
    Column("boot_id", String(32)),
    Column("comment", Text),
)


class FeedbackStore:
    def __init__(self, url: str):
        kwargs = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {"pool_pre_ping": True}
        self.engine: Engine = create_engine(url, **kwargs)
        metadata.create_all(self.engine)

    def log_search(self, **row) -> None:
        row.setdefault("ts", time.time())
        for key in ("filters", "result_ids", "tools"):
            if key in row and not isinstance(row[key], str):
                row[key] = json.dumps(row[key])
        with self.engine.begin() as conn:
            conn.execute(insert(search_logs).values(**row))

    def add_feedback(self, request_id: str, rating: int, boot_id: str | None, comment: str | None) -> None:
        with self.engine.begin() as conn:
            conn.execute(insert(feedback).values(ts=time.time(), request_id=request_id, rating=rating,
                                                 boot_id=boot_id, comment=comment))

    def review_queue(self, limit: int = 50) -> list[dict]:
        """Negative feedback joined with what the system did — the human review queue."""
        stmt = (
            select(feedback.c.ts, feedback.c.request_id, feedback.c.comment, feedback.c.boot_id,
                   search_logs.c.query, search_logs.c.mode, search_logs.c.tools, search_logs.c.filters,
                   search_logs.c.result_ids, search_logs.c.variant)
            .join(search_logs, search_logs.c.request_id == feedback.c.request_id, isouter=True)
            .where(feedback.c.rating < 0)
            .order_by(feedback.c.ts.desc())
            .limit(limit)
        )
        with self.engine.connect() as conn:
            return [dict(r._mapping) for r in conn.execute(stmt)]

    def ab_stats(self) -> list[dict]:
        """Thumbs-up rate, latency and cost per prompt variant."""
        stmt = (
            select(
                search_logs.c.variant,
                func.count(func.distinct(search_logs.c.request_id)).label("searches"),
                func.count(feedback.c.id).label("feedback_count"),
                func.avg(feedback.c.rating).label("avg_rating"),
                func.avg(search_logs.c.latency_ms).label("avg_latency_ms"),
                func.sum(search_logs.c.cost_usd).label("total_cost_usd"),
            )
            .join(feedback, feedback.c.request_id == search_logs.c.request_id, isouter=True)
            .group_by(search_logs.c.variant)
        )
        with self.engine.connect() as conn:
            rows = [dict(r._mapping) for r in conn.execute(stmt)]
        for r in rows:
            r["thumbs_up_rate"] = None if r["avg_rating"] is None else round((r["avg_rating"] + 1) / 2, 3)
        return rows
