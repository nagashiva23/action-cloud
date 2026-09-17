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


from .embeddings import get_embedding_provider
from .judge import MemoryJudge

class StoreResponse(BaseModel):
    id: uuid.UUID
    queued: bool
    message: str


class SearchResponse(BaseModel):
    query: str
    count: int
    results: list[ExperienceResult]


class ReuseReportRequest(BaseModel):
    success: bool
    agent_id: Optional[str] = None


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
    """
    exp = Experience(**payload.model_dump())

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
    Phase 2 hybrid retrieval: vector similarity + Postgres full-text ranking.
    """
    embedder = get_embedding_provider()
    query_vector = embedder.embed(q)

    rows = db.hybrid_search_experiences(
        query=q, query_vector=query_vector, limit=limit, min_tier=min_tier, agent_id=agent_id
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


@app.post("/experiences/{experience_id}/reuse")
def report_experience_reuse(
    experience_id: uuid.UUID,
    payload: ReuseReportRequest,
) -> dict:
    """
    Record an experience reuse event and trigger MemoryJudge evaluation.
    """
    row = db.record_experience_reuse(experience_id, success=payload.success)
    if row is None:
        raise HTTPException(status_code=404, detail="experience not found")

    transition = MemoryJudge.evaluate_reuse(row)
    current_tier = transition[0].value if transition else row["tier"]
    return {
        "id": str(experience_id),
        "reuse_count": row["reuse_count"],
        "reuse_success_count": row["reuse_success_count"],
        "tier": current_tier,
        "transition": transition[0].value if transition else None,
    }


@app.get("/experiences/{experience_id}")
def get_experience(experience_id: uuid.UUID) -> dict:
    """Full record including governance fields — for debugging and auditing."""
    row = db.get_experience(experience_id)
    if row is None:
        raise HTTPException(status_code=404, detail="experience not found")
    row.pop("embedding", None)  # a 1536-float array is noise in a JSON response
    return row


from .metrics import MetricCalculator


@app.get("/stats")
def stats(run_id: Optional[str] = None) -> dict:
    """Row counts. Useful for confirming a workload actually landed."""
    return {"run_id": run_id, "experiences": db.count_experiences(run_id)}


@app.get("/metrics")
def get_metrics(run_id: Optional[str] = None) -> dict:
    """
    Calculate full evaluation metrics (KRR, Redundancy Index, Token Savings, Latency, Cost).
    """
    return MetricCalculator.calculate_run_metrics(run_id)


@app.on_event("shutdown")
def _shutdown() -> None:
    db.close_pool()

