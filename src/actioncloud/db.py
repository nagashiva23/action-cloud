from __future__ import annotations

import atexit
import uuid
from contextlib import contextmanager
import json
from typing import Any, Callable, Iterator, Optional

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .config import settings
from .schema import Experience, MemoryTier

_pool: Optional[ConnectionPool] = None


def _vec(v: list[float]) -> str:
    """pgvector text literal."""
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def get_pool() -> ConnectionPool:
    """Lazily create the pool so importing this module never opens a socket."""
    global _pool
    if _pool is None:
        # Close on interpreter exit; otherwise the pool's worker threads make
        # every script and test run hang ~5 s at shutdown.
        atexit.register(close_pool)
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
    tier, confidence, prior
) VALUES (
    %(id)s, %(schema_version)s, %(agent_id)s, %(agent_role)s,
    %(task)s, %(problem)s, %(action)s, %(solution)s, %(result)s,
    %(tools_used)s, %(technologies)s, %(success)s,
    %(tokens_input)s, %(tokens_output)s, %(tool_calls)s, %(execution_time_ms)s, %(cost_usd)s,
    %(run_id)s, %(system)s, %(task_key)s, %(retrieved_experience_ids)s,
    %(tier)s, %(confidence)s, %(prior)s
)
-- The queue guarantees at-least-once delivery, so the same event can legitimately
-- arrive twice. Making the insert idempotent on the primary key is what turns
-- that guarantee from a data-corruption risk into a non-event.
ON CONFLICT (id) DO NOTHING
RETURNING id;
"""


def _insert_params(exp: Experience) -> dict[str, Any]:
    return {
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
        "prior": exp.prior,
    }


def insert_experience(exp: Experience) -> uuid.UUID:
    """Persist an experience. Safe to call twice with the same id."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(INSERT_SQL, _insert_params(exp))
            row = cur.fetchone()
        conn.commit()
    return exp.id if row is None else row["id"]


def insert_enriched_experience(
    exp: Experience,
    workflow: dict | None,
    knowledge_triples: list[dict],
    embedding: list[float] | None,
    tier_reason: str,
    decided_by: str,
) -> bool:
    """
    Insert an experience together with its extraction, embedding and the
    initial-tier audit row, in ONE transaction.

    Returns False when the id already existed (an SQS redelivery): nothing is
    written, so a duplicate message cannot duplicate the audit trail either.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(INSERT_SQL, _insert_params(exp))
            if cur.fetchone() is None:
                conn.rollback()
                return False
            cur.execute(
                """
                UPDATE experiences
                SET workflow = %s::jsonb,
                    knowledge_triples = %s::jsonb,
                    embedding = %s::vector,
                    embedded = %s
                WHERE id = %s
                """,
                (
                    json.dumps(workflow) if workflow else None,
                    json.dumps(knowledge_triples),
                    _vec(embedding) if embedding else None,
                    embedding is not None,
                    exp.id,
                ),
            )
            cur.execute(
                """
                INSERT INTO tier_transitions
                    (experience_id, from_tier, to_tier, reason, decided_by)
                VALUES (%s, NULL, %s, %s, %s)
                """,
                (exp.id, exp.tier.value, tier_reason, decided_by),
            )
        conn.commit()
    return True


def author_evidence(agent_id: str) -> tuple[int, int]:
    """(counted successes, counted reports) over every memory this agent wrote."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT coalesce(sum(reuse_success_count), 0) AS s, coalesce(sum(reuse_count), 0) AS n "
            "FROM experiences WHERE agent_id = %s",
            (agent_id,),
        )
        row = cur.fetchone()
    return int(row["s"]), int(row["n"])


def record_injections(experience_ids: list[uuid.UUID | str], agent_id: str) -> None:
    """Ledger entry: these memories were put in front of this agent."""
    if not experience_ids:
        return
    with get_conn() as conn, conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO injections (experience_id, agent_id) VALUES (%s, %s)",
            [(str(e), agent_id) for e in experience_ids],
        )
        conn.commit()


def _reuse_is_countable(cur, row: dict, reporter: str, success: bool) -> tuple[bool, str]:
    """
    Decide whether a reuse report counts, consuming an injection if it does.

    * Another agent's report counts only against an unreported injection of
      this memory to that agent — one injection, at most one counted report.
    * The author's positive report never counts (no vouching for yourself).
    * The author's negative report counts once ("my solution turned out
      wrong"); repeats are ignored so an author cannot bury a rival version.
    """
    exp_id = row["id"]
    if reporter == row["agent_id"]:
        if success:
            return False, "positive self-report ignored"
        cur.execute(
            "SELECT 1 FROM reuse_events WHERE experience_id = %s AND reporter_agent_id = %s "
            "AND counted AND NOT success LIMIT 1",
            (exp_id, reporter),
        )
        if cur.fetchone():
            return False, "author already reported this memory as failed"
        return True, "author reported own memory as failed"

    cur.execute(
        """
        UPDATE injections SET reported_at = now()
        WHERE id = (
            SELECT id FROM injections
            WHERE experience_id = %s AND agent_id = %s AND reported_at IS NULL
            ORDER BY created_at LIMIT 1
            FOR UPDATE SKIP LOCKED
        )
        RETURNING id
        """,
        (exp_id, reporter),
    )
    if cur.fetchone() is None:
        return False, "no unreported injection of this memory to this agent"
    return True, "counted against injection"


def apply_reuse(
    experience_id: uuid.UUID,
    success: bool,
    reporter_agent_id: str,
    decide: Callable[[MemoryTier, float, int, int], Optional[tuple]],
    decided_by: str,
    confidence_fn: Optional[Callable[[int, int, float], float]] = None,
) -> Optional[dict[str, Any]]:
    """
    Record one reuse outcome and apply the Judge's decision atomically.

    The row is locked (SELECT ... FOR UPDATE) for the whole read-decide-write,
    so concurrent reports serialise instead of racing on stale counts.
    Every report is logged in reuse_events; whether it COUNTS toward the
    memory's governance state is decided by _reuse_is_countable.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM experiences WHERE id = %s FOR UPDATE", (experience_id,)
            )
            row = cur.fetchone()
            if row is None:
                conn.rollback()
                return None

            counted, why = _reuse_is_countable(cur, row, reporter_agent_id, success)
            cur.execute(
                """
                INSERT INTO reuse_events (experience_id, reporter_agent_id, success, counted)
                VALUES (%s, %s, %s, %s)
                """,
                (experience_id, reporter_agent_id, success, counted),
            )
            if counted:
                cur.execute(
                    """
                    UPDATE experiences
                    SET reuse_count = reuse_count + 1,
                        reuse_success_count = reuse_success_count + %s
                    WHERE id = %s
                    RETURNING *
                    """,
                    (1 if success else 0, experience_id),
                )
                row = cur.fetchone()

            transition = None
            if counted and confidence_fn is not None:
                conf = confidence_fn(int(row["reuse_success_count"]), int(row["reuse_count"]),
                                     float(row.get("prior", 0.6)))
                cur.execute(
                    "UPDATE experiences SET confidence = %s WHERE id = %s",
                    (conf, experience_id),
                )
                row["confidence"] = conf
            if counted:
                decision = decide(
                    MemoryTier(row["tier"]),
                    float(row["confidence"]),
                    int(row["reuse_count"]),
                    int(row["reuse_success_count"]),
                )
                if decision is not None:
                    new_tier, new_conf, reason = decision
                    cur.execute(
                        """
                        UPDATE experiences SET tier = %s::memory_tier, confidence = %s
                        WHERE id = %s
                        """,
                        (new_tier.value, new_conf, experience_id),
                    )
                    cur.execute(
                        """
                        INSERT INTO tier_transitions
                            (experience_id, from_tier, to_tier, reason, decided_by)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (experience_id, row["tier"], new_tier.value, reason, decided_by),
                    )
                    transition = {"from": row["tier"], "to": new_tier.value, "reason": reason}
                    row["tier"] = new_tier.value
                    row["confidence"] = new_conf
        conn.commit()

    row.pop("embedding", None)
    return {"row": row, "counted": counted, "why": why, "transition": transition}


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
                        _vec(embedding),
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


_DOC = """to_tsvector('english',
        coalesce(e.task, '') || ' ' || coalesce(e.problem, '') || ' ' ||
        coalesce(e.solution, '') || ' ' || coalesce(e.result, ''))"""

# Who may read what. Mirrors MemoryTier.is_visible_to():
#   PRIVATE -> author only; AGENT -> author + same role; SHARED+ -> everyone.
# A caller with no identity sees fleet-visible (SHARED+) memories only.
VISIBILITY_SQL = """
    (
        e.tier >= 'shared'::memory_tier
        OR (%(agent_id)s::text IS NOT NULL AND e.agent_id = %(agent_id)s::text)
        OR (e.tier = 'agent'::memory_tier
            AND %(agent_role)s::text IS NOT NULL
            AND e.agent_role::text = %(agent_role)s::text)
    )
"""

_COMMON_FILTERS = f"""
      e.superseded_by IS NULL
  AND e.tier >= %(min_tier)s::memory_tier
  AND (%(scope_run_id)s::text IS NULL OR e.run_id = %(scope_run_id)s::text)
  AND {VISIBILITY_SQL}
"""

# Phase 1 retrieval: Postgres full-text search, ranked by ts_rank.
SEARCH_SQL = f"""
SELECT e.*,
       0.0::float8 AS vec_score,
       ts_rank({_DOC}, plainto_tsquery('english', %(query)s), 32)::float8 AS fts_score,
       ts_rank({_DOC}, plainto_tsquery('english', %(query)s), 32)::float8 AS relevance
FROM experiences e
WHERE {_DOC} @@ plainto_tsquery('english', %(query)s)
  AND {_COMMON_FILTERS}
ORDER BY relevance DESC, e.created_at DESC
LIMIT %(limit)s;
"""

# Hybrid retrieval. Cosine similarity is the primary signal; normalised
# ts_rank (flag 32 => rank/(rank+1), in [0,1)) adds a bounded keyword bonus.
# Both terms are on comparable scales, so `relevance` can be thresholded.
FTS_WEIGHT = 0.15

HYBRID_SEARCH_SQL = f"""
WITH scored AS (
    SELECT e.*,
           CASE WHEN e.embedding IS NULL THEN 0.0
                ELSE 1 - (e.embedding <=> %(vector)s::vector) END::float8 AS vec_score,
           ts_rank({_DOC}, plainto_tsquery('english', %(query)s), 32)::float8 AS fts_score
    FROM experiences e
    WHERE {_COMMON_FILTERS}
)
SELECT *, (vec_score + {FTS_WEIGHT} * fts_score) AS relevance
FROM scored
WHERE vec_score > 0 OR fts_score > 0
ORDER BY relevance DESC, created_at DESC
LIMIT %(limit)s;
"""


# Index-backed variant for large stores. The HNSW index returns the ann_k
# nearest memories by cosine; visibility, scope and the keyword bonus are then
# applied to those candidates only. pgvector < 0.8 filters after the index
# scan, so ann_k must comfortably exceed the number of memories a caller
# cannot see among the true neighbours.
HYBRID_ANN_SQL = f"""
WITH nn AS (
    SELECT id FROM experiences
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> %(vector)s::vector
    LIMIT %(ann_k)s
), scored AS (
    SELECT e.*,
           (1 - (e.embedding <=> %(vector)s::vector))::float8 AS vec_score,
           ts_rank({_DOC}, plainto_tsquery('english', %(query)s), 32)::float8 AS fts_score
    FROM experiences e JOIN nn ON nn.id = e.id
    WHERE {_COMMON_FILTERS}
)
SELECT *, (vec_score + {FTS_WEIGHT} * fts_score) AS relevance
FROM scored
ORDER BY relevance DESC, created_at DESC
LIMIT %(limit)s;
"""


def _params(query, limit, min_tier, agent_id, agent_role, scope_run_id, vector=None):
    return {
        "query": query,
        "vector": _vec(vector) if vector else None,
        "limit": limit,
        "min_tier": min_tier.value,
        "agent_id": agent_id,
        "agent_role": agent_role.value if hasattr(agent_role, "value") else agent_role,
        "scope_run_id": scope_run_id,
    }


def _strip(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for r in rows:
        r.pop("embedding", None)
    return rows


def search_experiences(
    query: str,
    limit: int = 5,
    min_tier: MemoryTier = MemoryTier.PRIVATE,
    agent_id: str | None = None,
    agent_role: Any = None,
    scope_run_id: str | None = None,
) -> list[dict[str, Any]]:
    """Keyword-only search (used when no query vector is available)."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            SEARCH_SQL, _params(query, limit, min_tier, agent_id, agent_role, scope_run_id)
        )
        return _strip(cur.fetchall())


def hybrid_search_experiences(
    query: str,
    query_vector: list[float] | None,
    limit: int = 5,
    min_tier: MemoryTier = MemoryTier.PRIVATE,
    agent_id: str | None = None,
    agent_role: Any = None,
    scope_run_id: str | None = None,
    ann_candidates: int = 0,
) -> list[dict[str, Any]]:
    """
    Hybrid vector + keyword retrieval under the visibility rules above.

    ann_candidates = 0 scores every visible row exactly (default). A positive
    value uses the HNSW index to pre-select that many nearest neighbours.
    """
    if not query_vector:
        return search_experiences(query, limit, min_tier, agent_id, agent_role, scope_run_id)
    params = _params(query, limit, min_tier, agent_id, agent_role, scope_run_id, query_vector)
    with get_conn() as conn, conn.cursor() as cur:
        if ann_candidates > 0:
            params["ann_k"] = ann_candidates
            cur.execute(f"SET LOCAL hnsw.ef_search = {min(1000, max(40, int(ann_candidates)))}")
            cur.execute(HYBRID_ANN_SQL, params)
        else:
            cur.execute(HYBRID_SEARCH_SQL, params)
        rows = cur.fetchall()
        conn.commit()
        return _strip(rows)


def count_experiences(run_id: str | None = None) -> int:
    with get_conn() as conn, conn.cursor() as cur:
        if run_id:
            cur.execute("SELECT count(*) AS n FROM experiences WHERE run_id = %s", (run_id,))
        else:
            cur.execute("SELECT count(*) AS n FROM experiences")
        return cur.fetchone()["n"]


def health_check(timeout_s: int = 3) -> bool:
    """
    Direct connection with a short timeout — deliberately NOT via the pool,
    whose 30 s wait made every "is Postgres up?" check hang when it wasn't.
    """
    try:
        with psycopg.connect(settings.dsn, connect_timeout=timeout_s) as conn:
            return conn.execute("SELECT 1").fetchone() is not None
    except Exception:
        return False


def require_database() -> None:
    """Exit with a clear message instead of a 30 s pool timeout traceback."""
    if not health_check():
        import sys  # noqa: PLC0415
        target = settings.dsn.rsplit("@", 1)[-1]
        sys.exit(
            f"Postgres is not reachable at {target}.\n"
            "Start Docker Desktop, then run `make up` (and `make migrate` for an old volume)."
        )
