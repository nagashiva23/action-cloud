"""
Agent Memory API — the single door into ActionCloud.

Agents never touch storage directly. Everything goes through here, which is
what makes governance enforceable: if an agent could write to Postgres itself,
the Memory Judge would be advisory rather than binding.

Phase 1 endpoints:
    GET  /health          liveness + dependency check
    POST /experiences     store an experience (async via queue -> 202)
    GET  /search          naive keyword retrieval
    GET  /experiences/{id}
    GET  /stats           row counts, for eyeballing an experiment run

Phase 2 adds /recommend (ranked, governed) and the MCP wrapper.
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel

from . import db, queue
from .config import settings
from .schema import (
    Experience,
    ExperienceCreate,
    ExperienceResult,
    MemoryTier,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(
    title="ActionCloud Agent Memory API",
    version="0.1.0",
    description="Phase 1 — plumbing only. No extraction, graph, or ranking yet.",
)


class StoreResponse(BaseModel):
    id: uuid.UUID
    queued: bool
    message: str


class SearchResponse(BaseModel):
    query: str
    count: int
    results: list[ExperienceResult]


@app.get("/health")
def health() -> dict:
    """
    Report on dependencies rather than just returning 200.

    A health check that only proves the process is alive tells you nothing when
    the interesting failure is 'API is up, Postgres is not'.
    """
    db_ok = db.health_check()
    try:
        queue.get_queue_url()
        queue_ok = True
    except Exception as e:  # noqa: BLE001 - report, don't crash the probe
        log.warning("queue health check failed: %s", e)
        queue_ok = False

    healthy = db_ok and queue_ok
    return {
        "status": "healthy" if healthy else "degraded",
        "database": db_ok,
        "queue": queue_ok,
        "mode": "local" if settings.is_local else "aws",
    }


@app.post(
    "/experiences",
    response_model=StoreResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def store_experience(payload: ExperienceCreate) -> StoreResponse:
    """
    Accept a completed experience.

    Returns 202 (not 201) on purpose: the row does not exist yet. It is queued,
    and a worker will persist it. Saying 201 Created would be a lie, and would
    also invite an agent to immediately read back something that is not there.

    The id is generated here and returned so the agent has a handle for the
    record before it is written.
    """
    exp = Experience(**payload.model_dump())

    # Escape hatch for tests only. In the experiment this stays off, since
    # writing synchronously would fold worker latency into the agent's measured
    # task time and quietly corrupt H3.
    if settings.sync_write:
        db.insert_experience(exp)
        return StoreResponse(id=exp.id, queued=False, message="written synchronously")

    try:
        queue.publish("experience.created", exp.model_dump(mode="json"))
    except Exception as e:  # noqa: BLE001
        log.exception("failed to publish experience")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"could not queue experience: {e}",
        ) from e

    return StoreResponse(id=exp.id, queued=True, message="queued for processing")


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1, description="Free-text task or problem description"),
    limit: int = Query(5, ge=1, le=50),
    min_tier: MemoryTier = Query(
        MemoryTier.PRIVATE,
        description="Trust floor. Phase 2 cross-agent reads should use 'shared'.",
    ),
    agent_id: Optional[str] = Query(
        None, description="If set, this agent's own private rows are also visible"
    ),
) -> SearchResponse:
    """
    Phase 1 retrieval: Postgres full-text ranking. No embeddings, no graph.

    Kept deliberately simple so that when the end-to-end loop misbehaves you
    are debugging one thing (the plumbing) rather than two (plumbing plus
    ranking).
    """
    rows = db.search_experiences(
        query=q, limit=limit, min_tier=min_tier, agent_id=agent_id
    )
    results = [
        ExperienceResult(
            id=r["id"],
            task=r["task"],
            problem=r["problem"],
            solution=r["solution"],
            result=r["result"],
            technologies=r["technologies"],
            tools_used=r["tools_used"],
            success=r["success"],
            tier=MemoryTier(r["tier"]),
            confidence=r["confidence"],
            workflow=r.get("workflow"),
            relevance=float(r["relevance"]) if r.get("relevance") is not None else None,
        )
        for r in rows
    ]
    return SearchResponse(query=q, count=len(results), results=results)


@app.get("/experiences/{experience_id}")
def get_experience(experience_id: uuid.UUID) -> dict:
    """Full record including governance fields — for debugging and auditing."""
    row = db.get_experience(experience_id)
    if row is None:
        raise HTTPException(status_code=404, detail="experience not found")
    row.pop("embedding", None)  # a 1536-float array is noise in a JSON response
    return row


@app.get("/stats")
def stats(run_id: Optional[str] = None) -> dict:
    """Row counts. Useful for confirming a workload actually landed."""
    return {"run_id": run_id, "experiences": db.count_experiences(run_id)}


@app.on_event("shutdown")
def _shutdown() -> None:
    db.close_pool()
