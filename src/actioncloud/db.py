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


def update_experience_tier(
    experience_id: uuid.UUID,
    tier: MemoryTier,
    confidence: float,
) -> None:
    """Update tier and confidence assigned by the Memory Judge."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE experiences
                SET tier = %s::memory_tier, confidence = %s, updated_at = now()
                WHERE id = %s
                """,
                (tier.value, confidence, experience_id),
            )
        conn.commit()


def record_experience_reuse(
    experience_id: uuid.UUID,
    success: bool,
) -> Optional[dict[str, Any]]:
    """Increment reuse_count and (if success) reuse_success_count. Returns updated record."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            if success:
                cur.execute(
                    """
                    UPDATE experiences
                    SET reuse_count = reuse_count + 1,
                        reuse_success_count = reuse_success_count + 1,
                        updated_at = now()
                    WHERE id = %s
                    RETURNING *
                    """,
                    (experience_id,),
                )
            else:
                cur.execute(
                    """
                    UPDATE experiences
                    SET reuse_count = reuse_count + 1,
                        updated_at = now()
                    WHERE id = %s
                    RETURNING *
                    """,
                    (experience_id,),
                )
            row = cur.fetchone()
        conn.commit()
    return row


def update_experience_extractions(
    experience_id: uuid.UUID,
    workflow: dict | None,
    knowledge_triples: list[dict],
    embedding: list[float] | None = None,
    embedded: bool = True,
) -> None:
    """Save extracted workflow, knowledge triples, and embedding vector."""
    import json
    with get_conn() as conn:
        with conn.cursor() as cur:
            if embedding:
                cur.execute(
                    """
                    UPDATE experiences
                    SET workflow = %s::jsonb,
                        knowledge_triples = %s::jsonb,
                        embedding = %s::vector,
                        embedded = %s,
                        updated_at = now()
                    WHERE id = %s
                    """,
                    (
                        json.dumps(workflow) if workflow else None,
                        json.dumps(knowledge_triples),
                        str(embedding),
                        embedded,
                        experience_id,
                    ),
                )
            else:
                cur.execute(
                    """
                    UPDATE experiences
                    SET workflow = %s::jsonb,
                        knowledge_triples = %s::jsonb,
                        embedded = %s,
                        updated_at = now()
                    WHERE id = %s
                    """,
                    (
                        json.dumps(workflow) if workflow else None,
                        json.dumps(knowledge_triples),
                        embedded,
                        experience_id,
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
  AND superseded_by IS NULL
  AND tier >= %(min_tier)s::memory_tier
  AND (%(agent_id)s::text IS NULL OR tier > 'private'::memory_tier OR agent_id = %(agent_id)s)
ORDER BY relevance DESC, created_at DESC
LIMIT %(limit)s;
"""


HYBRID_SEARCH_SQL = """
WITH fts_results AS (
    SELECT id,
           ts_rank(
               to_tsvector('english',
                   coalesce(task, '') || ' ' || coalesce(problem, '') || ' ' ||
                   coalesce(solution, '') || ' ' || coalesce(result, '')),
               plainto_tsquery('english', %(query)s)
           ) AS fts_score
    FROM experiences
    WHERE to_tsvector('english',
              coalesce(task, '') || ' ' || coalesce(problem, '') || ' ' ||
              coalesce(solution, '') || ' ' || coalesce(result, ''))
          @@ plainto_tsquery('english', %(query)s)
),
vector_results AS (
    SELECT id,
           (1 - (embedding <=> %(vector)s::vector)) AS vec_score
    FROM experiences
    WHERE embedding IS NOT NULL
)
SELECT e.*,
       COALESCE(f.fts_score, 0.0) AS fts_score,
       COALESCE(v.vec_score, 0.0) AS vec_score,
       (COALESCE(f.fts_score, 0.0) + 0.5 * COALESCE(v.vec_score, 0.0)) AS relevance
FROM experiences e
LEFT JOIN fts_results f ON e.id = f.id
LEFT JOIN vector_results v ON e.id = v.id
WHERE e.superseded_by IS NULL
  AND e.tier >= %(min_tier)s::memory_tier
  AND (%(agent_id)s::text IS NULL OR e.tier > 'private'::memory_tier OR e.agent_id = %(agent_id)s)
  AND (f.id IS NOT NULL OR v.id IS NOT NULL)
ORDER BY relevance DESC, e.created_at DESC
LIMIT %(limit)s;
"""


def search_experiences(
    query: str,
    limit: int = 5,
    min_tier: MemoryTier = MemoryTier.PRIVATE,
    agent_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Naive keyword search over stored experiences.
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


def hybrid_search_experiences(
    query: str,
    query_vector: list[float] | None,
    limit: int = 5,
    min_tier: MemoryTier = MemoryTier.PRIVATE,
    agent_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Phase 2 hybrid retrieval: combines vector similarity (pgvector) and full-text ranking.
    Falls back to text search if query_vector is None.
    """
    if not query_vector:
        return search_experiences(query, limit=limit, min_tier=min_tier, agent_id=agent_id)

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            HYBRID_SEARCH_SQL,
            {
                "query": query,
                "vector": str(query_vector),
                "limit": limit,
                "min_tier": min_tier.value,
                "agent_id": agent_id,
            },
        )
        results = cur.fetchall()
        if not results:  # Fallback to plain keyword search if hybrid returns empty
            return search_experiences(query, limit=limit, min_tier=min_tier, agent_id=agent_id)
        return results


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

