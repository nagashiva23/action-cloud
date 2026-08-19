"""
Database layer.

Deliberately thin — raw SQL through psycopg rather than an ORM. The queries in
Phase 2 (vector similarity, graph joins) and Phase 3 (metric aggregations) are
ones an ORM would only get in the way of, and being able to paste a query
straight into psql while debugging is worth more here than model classes.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import Any, Iterator, Optional

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .config import settings
from .schema import Experience, MemoryTier

_pool: Optional[ConnectionPool] = None


def get_pool() -> ConnectionPool:
    """Lazily create the pool so importing this module never opens a socket."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            settings.dsn,
            min_size=1,
            max_size=10,
            kwargs={"row_factory": dict_row},
            open=True,
        )
    return _pool


@contextmanager
def get_conn() -> Iterator[psycopg.Connection]:
    with get_pool().connection() as conn:
        yield conn


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


# --------------------------------------------------------------------------
# Writes
# --------------------------------------------------------------------------

INSERT_SQL = """
INSERT INTO experiences (
    id, schema_version, agent_id, agent_role,
    task, problem, action, solution, result,
    tools_used, technologies, success,
    tokens_input, tokens_output, tool_calls, execution_time_ms, cost_usd,
    run_id, system, task_key, retrieved_experience_ids,
    tier, confidence
) VALUES (
    %(id)s, %(schema_version)s, %(agent_id)s, %(agent_role)s,
    %(task)s, %(problem)s, %(action)s, %(solution)s, %(result)s,
    %(tools_used)s, %(technologies)s, %(success)s,
    %(tokens_input)s, %(tokens_output)s, %(tool_calls)s, %(execution_time_ms)s, %(cost_usd)s,
    %(run_id)s, %(system)s, %(task_key)s, %(retrieved_experience_ids)s,
    %(tier)s, %(confidence)s
)
-- The queue guarantees at-least-once delivery, so the same event can legitimately
-- arrive twice. Making the insert idempotent on the primary key is what turns
-- that guarantee from a data-corruption risk into a non-event.
ON CONFLICT (id) DO NOTHING
RETURNING id;
"""


def insert_experience(exp: Experience) -> uuid.UUID:
    """Persist an experience. Safe to call twice with the same id."""
    params = {
        "id": exp.id,
        "schema_version": exp.schema_version,
        "agent_id": exp.agent_id,
        "agent_role": exp.agent_role.value,
        "task": exp.task,
        "problem": exp.problem,
        "action": exp.action,
        "solution": exp.solution,
        "result": exp.result,
        "tools_used": exp.tools_used,
        "technologies": exp.technologies,
        "success": exp.success,
        "tokens_input": exp.tokens_input,
        "tokens_output": exp.tokens_output,
        "tool_calls": exp.tool_calls,
        "execution_time_ms": exp.execution_time_ms,
        "cost_usd": exp.cost_usd,
        "run_id": exp.run_id,
        "system": exp.system.value,
        "task_key": exp.task_key,
        "retrieved_experience_ids": [str(i) for i in exp.retrieved_experience_ids],
        "tier": exp.tier.value,
        "confidence": exp.confidence,
    }
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(INSERT_SQL, params)
            row = cur.fetchone()
        conn.commit()
    return exp.id if row is None else row["id"]


def record_tier_transition(
    experience_id: uuid.UUID,
    to_tier: MemoryTier,
    reason: str,
    from_tier: MemoryTier | None = None,
    decided_by: str = "memory_judge",
) -> None:
    """Append to the governance audit log. Phase 2 uses this heavily."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO tier_transitions
                    (experience_id, from_tier, to_tier, reason, decided_by)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    experience_id,
                    from_tier.value if from_tier else None,
                    to_tier.value,
                    reason,
                    decided_by,
                ),
            )
        conn.commit()


# --------------------------------------------------------------------------
# Reads
# --------------------------------------------------------------------------

def get_experience(experience_id: uuid.UUID) -> Optional[dict[str, Any]]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM experiences WHERE id = %s", (experience_id,))
        return cur.fetchone()


# Phase 1 retrieval: Postgres full-text search, ranked by ts_rank.
#
# This is intentionally the dumb version. It gives the pipeline something real
# to return so the end-to-end loop is testable, and it doubles as a control in
# Phase 2 — you can compare hybrid retrieval against it to show the graph and
# vector layers actually earn their complexity.
SEARCH_SQL = """
SELECT *,
       ts_rank(
           to_tsvector('english',
               coalesce(task, '') || ' ' || coalesce(problem, '') || ' ' ||
               coalesce(solution, '') || ' ' || coalesce(result, '')),
           plainto_tsquery('english', %(query)s)
       ) AS relevance
FROM experiences
WHERE to_tsvector('english',
          coalesce(task, '') || ' ' || coalesce(problem, '') || ' ' ||
          coalesce(solution, '') || ' ' || coalesce(result, ''))
      @@ plainto_tsquery('english', %(query)s)
  -- Never surface an experience that has been disproven and replaced.
  AND superseded_by IS NULL
  -- Tier gate. In Phase 1 the default floor is 'private' (everything visible)
  -- so the loop is easy to verify; Phase 2 raises it to 'shared' for
  -- cross-agent reads, which is where governance starts to bite.
  AND tier >= %(min_tier)s::memory_tier
  AND (%(agent_id)s::text IS NULL OR tier > 'private'::memory_tier OR agent_id = %(agent_id)s)
ORDER BY relevance DESC, created_at DESC
LIMIT %(limit)s;
"""


def search_experiences(
    query: str,
    limit: int = 5,
    min_tier: MemoryTier = MemoryTier.PRIVATE,
    agent_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Naive keyword search over stored experiences (Phase 1).

    `agent_id`, when given, lets an agent see its own private rows while still
    excluding other agents' private ones.
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            SEARCH_SQL,
            {
                "query": query,
                "limit": limit,
                "min_tier": min_tier.value,
                "agent_id": agent_id,
            },
        )
        return cur.fetchall()


def count_experiences(run_id: str | None = None) -> int:
    with get_conn() as conn, conn.cursor() as cur:
        if run_id:
            cur.execute("SELECT count(*) AS n FROM experiences WHERE run_id = %s", (run_id,))
        else:
            cur.execute("SELECT count(*) AS n FROM experiences")
        return cur.fetchone()["n"]


def health_check() -> bool:
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
            return cur.fetchone() is not None
    except Exception:
        return False
