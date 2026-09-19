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

from .policy import MemorySelectionPolicy
from .service import memory_service


class StoreResponse(BaseModel):
    id: uuid.UUID
    queued: bool
    message: str


class SearchResponse(BaseModel):
    query: str
    count: int
    results: list[ExperienceResult]


class ContextRequest(BaseModel):
    query: str
    agent_id: Optional[str] = None
    role: Optional[AgentRole] = None
    candidate_k: Optional[int] = None
    max_context_memories: Optional[int] = None
    similarity_threshold: Optional[float] = None
    context_token_budget: Optional[int] = None
    redundancy_threshold: Optional[float] = None


class ReuseReportRequest(BaseModel):
    success: bool
    agent_id: Optional[str] = None


@app.get("/health")
def health() -> dict:
    """
    Report on dependencies rather than just returning 200.
    """
    db_ok = db.health_check()
    try:
        queue.get_queue_url()
        queue_ok = True
    except Exception as e:  # noqa: BLE001
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
    try:
        res = memory_service.store_experience(payload)
        return StoreResponse(**res)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"could not queue experience: {e}",
        ) from e


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
    results = memory_service.search_memory(
        query=q, limit=limit, min_tier=min_tier, agent_id=agent_id
    )
    return SearchResponse(query=q, count=len(results), results=results)


@app.post("/context")
def prepare_context(req: ContextRequest) -> dict:
    """
    Adaptive context preparation endpoint (Search -> Governance -> Ranking -> Redundancy -> Token Budget).
    """
    pol = MemorySelectionPolicy.from_env()
    if req.candidate_k is not None:
        pol.candidate_k = req.candidate_k
    if req.max_context_memories is not None:
        pol.max_context_memories = req.max_context_memories
    if req.similarity_threshold is not None:
        pol.similarity_threshold = req.similarity_threshold
    if req.context_token_budget is not None:
        pol.context_token_budget = req.context_token_budget
    if req.redundancy_threshold is not None:
        pol.redundancy_threshold = req.redundancy_threshold

    return memory_service.prepare_context(
        query=req.query, agent_id=req.agent_id, role=req.role, policy=pol
    )


@app.post("/experiences/{experience_id}/reuse")
def report_experience_reuse(
    experience_id: uuid.UUID,
    payload: ReuseReportRequest,
) -> dict:
    """
    Record an experience reuse event and trigger MemoryJudge evaluation.
    """
    try:
        return memory_service.record_reuse(
            experience_id=experience_id, success=payload.success, agent_id=payload.agent_id
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="experience not found")


@app.get("/experiences/{experience_id}")
def get_experience(experience_id: uuid.UUID) -> dict:
    """Full record including governance fields — for debugging and auditing."""
    row = memory_service.get_experience(experience_id)
    if row is None:
        raise HTTPException(status_code=404, detail="experience not found")
    return row


@app.get("/stats")
def stats(run_id: Optional[str] = None) -> dict:
    """Row counts. Useful for confirming a workload actually landed."""
    return {"run_id": run_id, "experiences": db.count_experiences(run_id)}


@app.get("/metrics")
def get_metrics(run_id: Optional[str] = None) -> dict:
    """
    Calculate full evaluation metrics (KRR, Redundancy Index, Token Savings, Latency, Cost).
    """
    return memory_service.get_metrics(run_id)


@app.on_event("shutdown")
def _shutdown() -> None:
    db.close_pool()

