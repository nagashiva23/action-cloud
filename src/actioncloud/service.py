from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from . import db, queue
from .builder import CompactContextBuilder, count_tokens
from .config import settings
from .embeddings import get_embedding_provider
from .judge import MemoryJudge
from .metrics import MetricCalculator
from .policy import MemorySelectionPolicy
from .schema import (
    AgentRole,
    Experience,
    ExperienceCreate,
    ExperienceResult,
    MemoryTier,
)

log = logging.getLogger(__name__)


class MemoryService:
    """
    Central unified MemoryService encapsulating storage, hybrid retrieval,
    adaptive context preparation, governance evaluation, and system metrics.

    Shared by both REST API and Model Context Protocol (MCP) Server.
    """

    def __init__(self, sync_write: bool | None = None) -> None:
        self.sync_write = settings.sync_write if sync_write is None else sync_write

    # --- Writes -----------------------------------------------------------

    def store_experience(self, payload: ExperienceCreate) -> Dict[str, Any]:
        """
        Accept and process a completed experience.
        Returns immediately (HTTP 202 Accepted semantics).
        """
        exp = Experience(**payload.model_dump())

        if self.sync_write:
            db.insert_experience(exp)
            return {
                "id": exp.id,
                "queued": False,
                "message": "written synchronously",
            }

        try:
            queue.publish("experience.created", exp.model_dump(mode="json"))
        except Exception as e:
            log.exception("failed to publish experience to SQS queue")
            raise RuntimeError(f"could not queue experience: {e}") from e

        return {
            "id": exp.id,
            "queued": True,
            "message": "queued for processing",
        }

    # --- Reads & Search ---------------------------------------------------

    def search_memory(
        self,
        query: str,
        limit: int = 5,
        min_tier: MemoryTier = MemoryTier.PRIVATE,
        agent_id: str | None = None,
    ) -> List[ExperienceResult]:
        """
        Perform governed hybrid vector + full-text search.
        """
        embedder = get_embedding_provider()
        query_vector = embedder.embed(query)

        rows = db.hybrid_search_experiences(
            query=query,
            query_vector=query_vector,
            limit=limit,
            min_tier=min_tier,
            agent_id=agent_id,
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
        return results

    def get_experience(self, experience_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        """Retrieve full experience record including governance details."""
        row = db.get_experience(experience_id)
        if row is not None:
            row.pop("embedding", None)
        return row

    # --- Adaptive Context Preparation -------------------------------------

    def prepare_context(
        self,
        query: str,
        agent_id: str | None = None,
        role: AgentRole | None = None,
        policy: MemorySelectionPolicy | None = None,
    ) -> Dict[str, Any]:
        """
        Task-aware adaptive context builder:
          1. Candidate search retrieval (depth: candidate_k = 10)
          2. Relevance threshold filtering (relevance >= similarity_threshold)
          3. Redundancy filtering (CosineSimilarity < redundancy_threshold)
          4. Token budget allocation & procedural memory formatting
        """
        pol = policy or MemorySelectionPolicy.from_env()

        # Step 1: Candidate Search Retrieval
        embedder = get_embedding_provider()
        query_vector = embedder.embed(query)

        candidates = db.hybrid_search_experiences(
            query=query,
            query_vector=query_vector,
            limit=pol.candidate_k,
            min_tier=pol.min_tier,
            agent_id=agent_id,
        )

        # Step 2-4: Context Building & Token Budgeting
        context_str, injected_ids, candidate_count, final_count = (
            CompactContextBuilder.build_context(candidates, pol)
        )

        return {
            "query": query,
            "agent_id": agent_id,
            "role": role.value if role else None,
            "context": context_str,
            "injected_experience_ids": [str(i) for i in injected_ids],
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
        Record experience reuse feedback and trigger MemoryJudge evaluation.
        """
        row = db.record_experience_reuse(experience_id, success=success)
        if row is None:
            raise KeyError(f"experience {experience_id} not found")

        transition = MemoryJudge.evaluate_reuse(row)
        current_tier = transition[0].value if transition else row["tier"]

        return {
            "id": str(experience_id),
            "reuse_count": row["reuse_count"],
            "reuse_success_count": row["reuse_success_count"],
            "tier": current_tier,
            "transition": transition[0].value if transition else None,
            "reason": transition[2] if transition else "Tier maintained",
        }

    # --- Metrics ----------------------------------------------------------

    def get_metrics(self, run_id: str | None = None) -> Dict[str, Any]:
        """Retrieve aggregated empirical evaluation metrics."""
        return MetricCalculator.calculate_run_metrics(run_id)


# Module-level singleton instance
memory_service = MemoryService()
