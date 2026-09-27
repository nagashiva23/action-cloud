"""
ActionCloud — ingest pipeline.

One function, used by BOTH the SQS worker and the synchronous write path, so
an experience is processed identically however it arrives:

  1. Memory Judge assigns the initial tier and confidence.
  2. Extractor derives a reusable workflow + knowledge triples.
  3. Embedding provider vectorises the task text.
  4. Everything, including the initial-tier audit row, is written in one
     idempotent transaction (a redelivered message is a no-op).

Previously the sync path skipped steps 1-3, so rows written with SYNC_WRITE
had no embedding, no workflow, and sat at the default PRIVATE tier.
"""

from __future__ import annotations

import logging
from typing import Any

from . import db
from .embeddings import experience_embedding_text, get_embedding_provider
from .extractor import ExperienceExtractor
from .judge import DECIDED_BY, MemoryJudge
from .schema import Experience, MemoryTier

log = logging.getLogger(__name__)


def process_experience(exp: Experience | dict[str, Any], governance: bool = True,
                       author_prior: bool = False) -> bool:
    """
    Enrich and persist one experience. Returns False if it already existed.

    governance=False is the "flat shared memory" ablation arm: every
    experience is fleet-visible from the moment it is written (failures
    included, labelled FAILED) and no reuse feedback is applied.
    """
    if not isinstance(exp, Experience):
        exp = Experience(**exp)

    if governance:
        tier, confidence, reason = MemoryJudge.assign_initial_tier(exp)
        if author_prior and exp.success:
            # Start from the author's track record instead of the global prior.
            s, n = db.author_evidence(exp.agent_id)
            exp.prior = MemoryJudge.author_reputation(s, n)
            confidence = exp.prior
            reason += f" (author reputation {exp.prior:.2f} over {n} reports)"
    else:
        tier, confidence, reason = MemoryTier.SHARED, 0.5, "Governance disabled (ablation arm)"
    exp.tier = tier
    exp.confidence = confidence

    workflow, triples = ExperienceExtractor().extract(exp)
    embedding = get_embedding_provider().embed(
        experience_embedding_text(exp.task, exp.technologies, exp.problem)
    )

    inserted = db.insert_enriched_experience(
        exp,
        workflow=workflow,
        knowledge_triples=triples,
        embedding=embedding,
        tier_reason=reason,
        decided_by=DECIDED_BY,
    )
    if inserted:
        log.info(
            "stored %s | %s/%s | success=%s | tier=%s | %d tokens",
            exp.id, exp.agent_role.value, exp.system.value, exp.success,
            tier.value, exp.total_tokens,
        )
    else:
        log.info("duplicate delivery for %s ignored", exp.id)
    return inserted
