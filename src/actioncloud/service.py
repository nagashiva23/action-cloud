from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from . import db, queue
from .builder import CompactContextBuilder, count_tokens
from .config import settings
from .embeddings import experience_embedding_text, get_embedding_provider
from .judge import DECIDED_BY, MemoryJudge
from .metrics import MetricCalculator
from .pipeline import process_experience
from .policy import MemorySelectionPolicy
from .schema import (
    AgentRole,
    Experience,
    ExperienceCreate,
    ExperienceResult,
    MemoryTier,
)

log = logging.getLogger(__name__)


def _role_value(role: AgentRole | str | None) -> Optional[str]:
    if role is None:
        return None
    return role.value if isinstance(role, AgentRole) else AgentRole(role).value


class MemoryService:
    """
    Central MemoryService: storage, governed hybrid retrieval, adaptive
    context preparation, governance feedback and metrics.

    Shared by the REST API, the MCP server and the in-process benchmark
    client, so all three behave identically.

    Identity matters for every read: `agent_id` and `agent_role` decide which
    PRIVATE / AGENT tier memories are visible (see db.VISIBILITY_SQL). A caller
    that gives neither sees only SHARED-and-above memories.
    """

    def __init__(self, sync_write: bool | None = None, governance: bool = True,
                 author_prior: bool | None = None) -> None:
        self.sync_write = settings.sync_write if sync_write is None else sync_write
        # False only for the flat-memory ablation arm (see pipeline.py).
        self.governance = governance
        # Seed new memories' confidence from their author's reputation.
        import os  # noqa: PLC0415
        self.author_prior = (
            os.environ.get("MEMORY_AUTHOR_PRIOR", "false").strip().lower() in {"1", "true", "yes"}
            if author_prior is None else author_prior
        )

    # --- Writes -----------------------------------------------------------

    def store_experience(self, payload: ExperienceCreate) -> Dict[str, Any]:
        """Accept a completed experience (HTTP 202 semantics when queued)."""
        exp = Experience(**payload.model_dump())

        if self.sync_write:
            process_experience(exp, governance=self.governance, author_prior=self.author_prior)
            return {"id": exp.id, "queued": False, "message": "processed synchronously"}

        try:
            queue.publish("experience.created", exp.model_dump(mode="json"))
        except Exception as e:
            log.exception("failed to publish experience to SQS queue")
            raise RuntimeError(f"could not queue experience: {e}") from e

        return {"id": exp.id, "queued": True, "message": "queued for processing"}

    # --- Reads & Search ---------------------------------------------------

    def _query_vector(self, query: str, technologies: list[str] | None) -> list[float]:
        return get_embedding_provider().embed(experience_embedding_text(query, technologies))

    def search_memory(
        self,
        query: str,
        limit: int = 5,
        min_tier: MemoryTier = MemoryTier.PRIVATE,
        agent_id: str | None = None,
        agent_role: AgentRole | str | None = None,
        technologies: list[str] | None = None,
        scope_run_id: str | None = None,
    ) -> List[ExperienceResult]:
        """Governed hybrid vector + full-text search."""
        rows = db.hybrid_search_experiences(
            query=query,
            query_vector=self._query_vector(query, technologies),
            limit=limit,
            min_tier=min_tier,
            agent_id=agent_id,
            agent_role=_role_value(agent_role),
            scope_run_id=scope_run_id,
        )
        return [
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

    def get_experience(self, experience_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        row = db.get_experience(experience_id)
        if row is not None:
            row.pop("embedding", None)
        return row

    # --- Adaptive Context Preparation -------------------------------------

    def prepare_context(
        self,
        query: str,
        agent_id: str | None = None,
        role: AgentRole | str | None = None,
        policy: MemorySelectionPolicy | None = None,
        technologies: list[str] | None = None,
        scope_run_id: str | None = None,
    ) -> Dict[str, Any]:
        """
        1. Retrieve candidate_k governed candidates (hybrid search).
        2. Drop candidates below similarity_threshold (none pass -> inject nothing).
        3. Drop near-duplicates (same task_key or text similarity >= redundancy_threshold).
        4. Format up to max_context_memories within context_token_budget.
        """
        pol = policy or MemorySelectionPolicy.from_env()
        candidates = db.hybrid_search_experiences(
            query=query,
            query_vector=self._query_vector(query, technologies),
            limit=pol.candidate_k,
            min_tier=pol.min_tier,
            agent_id=agent_id,
            agent_role=_role_value(role),
            scope_run_id=scope_run_id,
            ann_candidates=pol.ann_candidates,
        )
        context_str, injected_ids, candidate_count, final_count = (
            CompactContextBuilder.build_context(candidates, pol)
        )
        # Ledger for governance: reuse credit is only granted against these.
        if agent_id and injected_ids:
            db.record_injections(injected_ids, agent_id)
        injected = {str(i) for i in injected_ids}
        return {
            "query": query,
            "agent_id": agent_id,
            "role": _role_value(role),
            "context": context_str,
            "injected_experience_ids": [str(i) for i in injected_ids],
            "injected_task_keys": [
                c.get("task_key") for c in candidates if str(c["id"]) in injected
            ],
            "candidate_count": candidate_count,
            "final_count": final_count,
            "context_tokens": count_tokens(context_str),
        }

    # --- Governance & Reuse Feedback -------------------------------------

    def record_reuse(
        self,
        experience_id: uuid.UUID,
        success: bool,
        agent_id: str | None = None,
    ) -> Dict[str, Any]:
        """
        Record a reuse outcome and let the Memory Judge re-tier the memory.

        `agent_id` (the reporter) is required: an anonymous report cannot be
        checked against the author, so it would let anyone promote anything.
        """
        if not agent_id:
            raise ValueError("agent_id of the reporting agent is required")

        res = db.apply_reuse(
            experience_id,
            success=success,
            reporter_agent_id=agent_id,
            decide=MemoryJudge.decide,
            decided_by=DECIDED_BY,
            confidence_fn=MemoryJudge.posterior_confidence,
        )
        if res is None:
            raise KeyError(f"experience {experience_id} not found")

        row, transition = res["row"], res["transition"]
        return {
            "id": str(experience_id),
            "counted": res["counted"],
            "reuse_count": row["reuse_count"],
            "reuse_success_count": row["reuse_success_count"],
            "tier": row["tier"],
            "confidence": row["confidence"],
            "transition": transition["to"] if transition else None,
            "reason": (
                transition["reason"] if transition
                else f"Not counted: {res['why']}" if not res["counted"]
                else "Tier maintained"
            ),
        }

    # --- Metrics ----------------------------------------------------------

    def get_metrics(self, run_id: str | None = None) -> Dict[str, Any]:
        return MetricCalculator.calculate_run_metrics(run_id)


memory_service = MemoryService()
