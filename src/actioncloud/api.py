from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from pydantic import BaseModel

from . import auth, db, queue
from .auth import Principal
from .config import settings
from .policy import MemorySelectionPolicy
from .schema import AgentRole, ExperienceCreate, ExperienceResult, MemoryTier
from .service import memory_service

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

@asynccontextmanager
async def _lifespan(_app):
    yield
    db.close_pool()


app = FastAPI(
    lifespan=_lifespan,
    title="ActionCloud Agent Memory API",
    version="0.5.0",
    description=(
        "Governed adaptive experience memory for agent fleets. Every endpoint except "
        "/health requires an agent API key in the X-API-Key header; the caller's "
        "agent_id and role are taken from the key."
    ),
)


# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------

def authenticate(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None),
) -> Optional[Principal]:
    """
    Resolve the caller. Returns None only when AUTH_REQUIRED=false (legacy
    mode, identity taken from the request as before).
    """
    key = x_api_key
    if not key and authorization and authorization.lower().startswith("bearer "):
        key = authorization[7:].strip()
    if not settings.auth_required and not key:
        return None
    principal = auth.resolve_key(key, settings.admin_key)
    if principal is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing, unknown or revoked API key (send it as X-API-Key)",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return principal


def require_admin(principal: Optional[Principal] = Depends(authenticate)) -> Principal:
    if principal is None or not principal.is_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "admin key required")
    return principal


def resolve_identity(
    principal: Optional[Principal],
    claimed_id: Optional[str],
    claimed_role: Optional[AgentRole | str],
) -> tuple[Optional[str], Optional[AgentRole]]:
    """
    The identity a request acts as.

    With an agent key, it is the key's agent — a body/query naming anyone else
    is refused (403) rather than silently corrected, so a misconfigured client
    fails loudly. The admin key may act as any named agent. With auth off,
    the claimed identity is used as-is.
    """
    role = AgentRole(claimed_role) if claimed_role else None
    if principal is None or principal.is_admin:
        return claimed_id, role
    if claimed_id and claimed_id != principal.agent_id:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            f"this key belongs to agent {principal.agent_id!r}, not {claimed_id!r}",
        )
    if role and role != principal.role:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            f"agent {principal.agent_id!r} has role {principal.role.value!r}, not {role.value!r}",
        )
    return principal.agent_id, principal.role


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

class StoreRequest(ExperienceCreate):
    """ExperienceCreate where identity may be omitted (it comes from the key)."""

    agent_id: Optional[str] = None  # type: ignore[assignment]
    agent_role: Optional[AgentRole] = None  # type: ignore[assignment]


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
    technologies: list[str] = []
    scope_run_id: Optional[str] = None
    candidate_k: Optional[int] = None
    max_context_memories: Optional[int] = None
    similarity_threshold: Optional[float] = None
    context_token_budget: Optional[int] = None
    redundancy_threshold: Optional[float] = None


class ReuseReportRequest(BaseModel):
    success: bool
    agent_id: Optional[str] = None  # reporter; taken from the key when authenticated


class AgentCreateRequest(BaseModel):
    agent_id: str
    agent_role: AgentRole
    rotate: bool = False


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    """Dependency status. Unauthenticated so load balancers can probe it."""
    db_ok = db.health_check()
    if memory_service.sync_write:
        queue_ok = None  # not used: writes are processed in-request
    else:
        try:
            queue.get_queue_url()
            queue_ok = True
        except Exception as e:  # noqa: BLE001
            log.warning("queue health check failed: %s", e)
            queue_ok = False

    healthy = db_ok and queue_ok is not False
    return {
        "status": "healthy" if healthy else "degraded",
        "database": db_ok,
        "queue": queue_ok,
        "write_mode": "sync" if memory_service.sync_write else "queued",
        "auth_required": settings.auth_required,
        "mode": "local" if settings.is_local else "aws",
    }


@app.post("/experiences", response_model=StoreResponse, status_code=status.HTTP_202_ACCEPTED)
def store_experience(
    payload: StoreRequest, principal: Optional[Principal] = Depends(authenticate)
) -> StoreResponse:
    agent_id, role = resolve_identity(principal, payload.agent_id, payload.agent_role)
    if not agent_id or not role:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "agent_id and agent_role are required")
    data = payload.model_dump()
    data.update(agent_id=agent_id, agent_role=role)
    try:
        res = memory_service.store_experience(ExperienceCreate(**data))
        return StoreResponse(**res)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"could not store experience: {e}",
        ) from e


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1, description="Free-text task or problem description"),
    limit: int = Query(5, ge=1, le=50),
    min_tier: MemoryTier = Query(MemoryTier.PRIVATE, description="Trust floor"),
    agent_id: Optional[str] = Query(None, description="Only needed when auth is off"),
    agent_role: Optional[AgentRole] = Query(None, description="Only needed when auth is off"),
    technologies: list[str] = Query([], description="Tech tags added to the query embedding"),
    scope_run_id: Optional[str] = Query(None, description="Only search this run's memories"),
    principal: Optional[Principal] = Depends(authenticate),
) -> SearchResponse:
    """Governed hybrid search. Visibility is computed for the key's agent."""
    agent_id, role = resolve_identity(principal, agent_id, agent_role)
    results = memory_service.search_memory(
        query=q, limit=limit, min_tier=min_tier, agent_id=agent_id,
        agent_role=role, technologies=technologies, scope_run_id=scope_run_id,
    )
    return SearchResponse(query=q, count=len(results), results=results)


@app.post("/context")
def prepare_context(
    req: ContextRequest, principal: Optional[Principal] = Depends(authenticate)
) -> dict:
    """Compact governed context. Injected memories are recorded for this agent."""
    agent_id, role = resolve_identity(principal, req.agent_id, req.role)
    pol = MemorySelectionPolicy.from_env()
    for field in ("candidate_k", "max_context_memories", "similarity_threshold",
                  "context_token_budget", "redundancy_threshold"):
        value = getattr(req, field)
        if value is not None:
            setattr(pol, field, value)
    return memory_service.prepare_context(
        query=req.query, agent_id=agent_id, role=role, policy=pol,
        technologies=req.technologies, scope_run_id=req.scope_run_id,
    )


@app.post("/experiences/{experience_id}/reuse")
def report_experience_reuse(
    experience_id: uuid.UUID,
    payload: ReuseReportRequest,
    principal: Optional[Principal] = Depends(authenticate),
) -> dict:
    """
    Report whether a memory worked. Counts only against a prior injection of
    that memory to the reporting agent (or once, negatively, by its author).
    """
    reporter, _ = resolve_identity(principal, payload.agent_id, None)
    try:
        return memory_service.record_reuse(
            experience_id=experience_id, success=payload.success, agent_id=reporter
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="experience not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/experiences/{experience_id}")
def get_experience(
    experience_id: uuid.UUID, principal: Optional[Principal] = Depends(authenticate)
) -> dict:
    """Full record. Agents only see records they could retrieve; admin sees all."""
    row = memory_service.get_experience(experience_id)
    if row is None:
        raise HTTPException(status_code=404, detail="experience not found")
    if principal is not None and not principal.is_admin:
        visible = MemoryTier(row["tier"]).is_visible_to(
            author_agent_id=row["agent_id"], author_role=row["agent_role"],
            requester_agent_id=principal.agent_id, requester_role=principal.role.value,
        )
        if not visible:
            # 404, not 403: do not confirm that a hidden memory exists.
            raise HTTPException(status_code=404, detail="experience not found")
    return row


@app.get("/stats")
def stats(run_id: Optional[str] = None, _: Optional[Principal] = Depends(authenticate)) -> dict:
    return {"run_id": run_id, "experiences": db.count_experiences(run_id)}


@app.get("/metrics")
def get_metrics(run_id: Optional[str] = None, _: Optional[Principal] = Depends(authenticate)) -> dict:
    return memory_service.get_metrics(run_id)


# --- admin -----------------------------------------------------------------

@app.post("/agents", status_code=status.HTTP_201_CREATED)
def create_agent(req: AgentCreateRequest, _: Principal = Depends(require_admin)) -> dict:
    """Register an agent. The API key is returned once and never stored in clear."""
    try:
        key = auth.register_agent(req.agent_id, req.agent_role, rotate=req.rotate)
    except Exception as e:  # noqa: BLE001 — unique violation etc.
        raise HTTPException(status.HTTP_409_CONFLICT, f"could not register agent: {e}") from e
    return {"agent_id": req.agent_id, "agent_role": req.agent_role.value, "api_key": key}


@app.delete("/agents/{agent_id}")
def delete_agent(agent_id: str, _: Principal = Depends(require_admin)) -> dict:
    if not auth.revoke_agent(agent_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no active agent with that id")
    return {"agent_id": agent_id, "revoked": True}


