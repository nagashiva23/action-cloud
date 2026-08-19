"""
ActionCloud — Experience Object schema.

This is the keystone of the whole system: every worker, every storage backend,
and every metric in the Phase 3 experiment reads from this shape. Treat changes
here as breaking changes and bump SCHEMA_VERSION.

Three models, deliberately separated:

  ExperienceCreate   what an agent submits when it finishes a task
  Experience         what ActionCloud stores (adds governance + provenance)
  ExperienceResult   what the retrieval path returns to an agent

Agents can never set governance fields (tier, confidence, reuse counts). Those
are assigned by the Memory Judge in Phase 2. Keeping them out of the create
model means an agent cannot promote its own memory, which is the entire point
of having governance at all.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

# Bump this whenever the stored shape changes. Persisted rows keep the version
# they were written with, so a later migration can find and upgrade them.
SCHEMA_VERSION = "1.0.0"


class AgentRole(str, Enum):
    """The 5-6 agent roles from the proposal (Section 6.1)."""

    RESEARCH = "research"
    CODING = "coding"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    DOCUMENTATION = "documentation"
    DATA_ANALYSIS = "data_analysis"


class MemoryTier(str, Enum):
    """
    The five-tier governance ladder (proposal Section 9.3).

    An experience is NOT trusted by default. It enters at PRIVATE and climbs
    only by being reused successfully. The Memory Judge (Phase 2) owns all
    transitions; nothing else may write this field.
    """

    PRIVATE = "private"                # visible only to the run that created it
    AGENT = "agent"                    # visible to the same agent across runs
    SHARED = "shared"                  # visible fleet-wide, not yet validated
    VALIDATED = "validated"            # proven by repeated successful reuse
    ORGANIZATIONAL = "organizational"  # canonical knowledge

    @property
    def rank(self) -> int:
        return _TIER_RANK[self]

    def is_visible_to_fleet(self) -> bool:
        """Whether an arbitrary agent may retrieve this experience."""
        return self.rank >= MemoryTier.SHARED.rank


_TIER_RANK = {
    MemoryTier.PRIVATE: 0,
    MemoryTier.AGENT: 1,
    MemoryTier.SHARED: 2,
    MemoryTier.VALIDATED: 3,
    MemoryTier.ORGANIZATIONAL: 4,
}


class SystemCondition(str, Enum):
    """
    Which arm of the Phase 3 experiment produced this record.

    Baseline runs still submit experiences (so we capture their token/latency
    cost) but the baseline agent never *retrieves*. Tagging every row with its
    condition is what makes the A/B comparison computable with a GROUP BY
    instead of a forensic reconstruction later.
    """

    BASELINE = "baseline"          # System A — stateless
    ACTIONCLOUD = "actioncloud"    # System B — memory-enabled


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ExperienceCreate(BaseModel):
    """
    What an agent submits after finishing a task.

    The task/problem/action/solution/result decomposition is what makes an
    experience *reusable* rather than a transcript: a future agent can match on
    `problem` and read `solution` without re-reading the whole episode.
    """

    # --- provenance -------------------------------------------------------
    agent_id: str = Field(..., min_length=1, description="Stable per-agent identity")
    agent_role: AgentRole

    # --- the experience itself -------------------------------------------
    task: str = Field(..., min_length=1, description="What the agent was asked to do")
    problem: Optional[str] = Field(None, description="Obstacle hit, if any")
    action: str = Field(..., min_length=1, description="What the agent actually did")
    solution: Optional[str] = Field(None, description="What resolved the problem")
    result: str = Field(..., min_length=1, description="Outcome in plain terms")

    tools_used: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(
        default_factory=list,
        description="Named tech (docker, neo4j, fastapi). Seeds the Phase 2 graph.",
    )

    success: bool

    # --- measurement (drives every Phase 3 metric) ------------------------
    tokens_input: int = Field(0, ge=0)
    tokens_output: int = Field(0, ge=0)
    tool_calls: int = Field(0, ge=0)
    execution_time_ms: int = Field(0, ge=0)
    cost_usd: float = Field(0.0, ge=0.0)

    # --- experiment bookkeeping ------------------------------------------
    run_id: str = Field(..., description="Groups all tasks in one experimental run")
    system: SystemCondition
    task_key: Optional[str] = Field(
        None,
        description=(
            "Stable identifier for the logical task, shared by repeated/related "
            "variants in the 100-task workload. This is what makes the Redundancy "
            "Index computable: same task_key solved twice from scratch = redundant work."
        ),
    )
    retrieved_experience_ids: list[uuid.UUID] = Field(
        default_factory=list,
        description=(
            "Experiences this agent was shown before starting. Non-empty only in "
            "the actioncloud arm. Combined with `success`, this is the raw material "
            "for Knowledge Reuse Rate and for the Memory Judge's success statistics."
        ),
    )

    @field_validator("tools_used", "technologies")
    @classmethod
    def _normalise_tags(cls, v: list[str]) -> list[str]:
        """Lowercase + dedupe so 'Docker' and 'docker' don't split the graph."""
        seen, out = set(), []
        for item in v:
            k = item.strip().lower()
            if k and k not in seen:
                seen.add(k)
                out.append(k)
        return out


class Experience(ExperienceCreate):
    """
    The stored record: an ExperienceCreate plus everything ActionCloud adds.

    Governance fields are separated below because they are written *only* by
    the Memory Judge, never by an agent.
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    schema_version: str = SCHEMA_VERSION

    # --- governance (Memory Judge owns these) -----------------------------
    tier: MemoryTier = MemoryTier.PRIVATE
    confidence: float = Field(0.5, ge=0.0, le=1.0)

    # Observed reuse, which is how an experience earns promotion.
    reuse_count: int = Field(0, ge=0)
    reuse_success_count: int = Field(0, ge=0)

    # --- versioning -------------------------------------------------------
    # A disproven experience is superseded, never overwritten, so a bad memory
    # leaves an audit trail instead of silently vanishing.
    version: int = Field(1, ge=1)
    superseded_by: Optional[uuid.UUID] = None

    # --- Phase 2 placeholders (nullable now, populated by workers later) ---
    workflow: Optional[dict] = Field(
        None, description="Generalized reusable step sequence (Phase 2 extraction)"
    )
    knowledge_triples: list[dict] = Field(
        default_factory=list, description="subject/predicate/object triples (Phase 2)"
    )
    embedded: bool = Field(
        False, description="Whether the Embedding Worker has processed this row"
    )

    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    @property
    def observed_success_rate(self) -> Optional[float]:
        """
        Success rate across reuses. None until reused at least once — an
        untested experience is genuinely unknown, and collapsing that to 0.0
        would let the Judge punish new memories for being new.
        """
        if self.reuse_count == 0:
            return None
        return self.reuse_success_count / self.reuse_count

    @property
    def total_tokens(self) -> int:
        return self.tokens_input + self.tokens_output


class ExperienceResult(BaseModel):
    """
    What retrieval hands back to an agent.

    Deliberately narrower than `Experience`: an agent needs the content and
    enough signal to judge trust, not the full audit trail.
    """

    id: uuid.UUID
    task: str
    problem: Optional[str]
    solution: Optional[str]
    result: str
    technologies: list[str]
    tools_used: list[str]
    success: bool
    tier: MemoryTier
    confidence: float
    workflow: Optional[dict] = None

    # Populated by the Phase 2 Retrieval Engine; None in Phase 1's naive search.
    relevance: Optional[float] = None
    utility_score: Optional[float] = None

    @classmethod
    def from_experience(cls, exp: Experience, **scores) -> "ExperienceResult":
        return cls(
            id=exp.id,
            task=exp.task,
            problem=exp.problem,
            solution=exp.solution,
            result=exp.result,
            technologies=exp.technologies,
            tools_used=exp.tools_used,
            success=exp.success,
            tier=exp.tier,
            confidence=exp.confidence,
            workflow=exp.workflow,
            **scores,
        )
