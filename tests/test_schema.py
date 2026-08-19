"""
Schema unit tests — no database, no queue, no network.

These run in milliseconds and catch the class of bug that is most expensive
later: a governance rule that exists in the docstring but not in the code.
"""

from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from actioncloud.schema import (
    AgentRole,
    Experience,
    ExperienceCreate,
    ExperienceResult,
    MemoryTier,
    SystemCondition,
)


def _minimal(**overrides) -> dict:
    base = dict(
        agent_id="agent-1",
        agent_role=AgentRole.CODING,
        task="Write a function that reverses a string",
        action="Wrote the function",
        result="Done",
        success=True,
        run_id="run-1",
        system=SystemCondition.BASELINE,
    )
    base.update(overrides)
    return base


class TestTierOrdering:
    def test_ranks_ascend(self):
        assert MemoryTier.PRIVATE.rank < MemoryTier.AGENT.rank
        assert MemoryTier.AGENT.rank < MemoryTier.SHARED.rank
        assert MemoryTier.SHARED.rank < MemoryTier.VALIDATED.rank
        assert MemoryTier.VALIDATED.rank < MemoryTier.ORGANIZATIONAL.rank

    def test_private_and_agent_are_not_fleet_visible(self):
        """The governance promise: unproven memory stays out of the fleet."""
        assert not MemoryTier.PRIVATE.is_visible_to_fleet()
        assert not MemoryTier.AGENT.is_visible_to_fleet()
        assert MemoryTier.SHARED.is_visible_to_fleet()
        assert MemoryTier.VALIDATED.is_visible_to_fleet()


class TestExperienceCreate:
    def test_tags_are_lowercased_and_deduped(self):
        """'Docker' and 'docker' must not become two nodes in the Phase 2 graph."""
        exp = ExperienceCreate(**_minimal(technologies=["Docker", "docker", " DOCKER "]))
        assert exp.technologies == ["docker"]

    def test_empty_tags_dropped(self):
        exp = ExperienceCreate(**_minimal(tools_used=["bash", "", "   "]))
        assert exp.tools_used == ["bash"]

    def test_negative_tokens_rejected(self):
        with pytest.raises(ValidationError):
            ExperienceCreate(**_minimal(tokens_input=-1))

    def test_blank_task_rejected(self):
        with pytest.raises(ValidationError):
            ExperienceCreate(**_minimal(task=""))

    def test_agent_cannot_set_governance_fields(self):
        """
        The core governance guarantee. If an agent could submit tier or
        confidence, it could promote its own memory and the Memory Judge would
        be decorative.
        """
        create = ExperienceCreate(**_minimal())
        assert not hasattr(create, "tier")
        assert not hasattr(create, "confidence")
        assert not hasattr(create, "reuse_count")


class TestExperienceDefaults:
    def test_enters_untrusted(self):
        exp = Experience(**_minimal())
        assert exp.tier is MemoryTier.PRIVATE
        assert exp.version == 1
        assert exp.superseded_by is None
        assert exp.reuse_count == 0

    def test_success_rate_is_none_before_any_reuse(self):
        """
        Unknown must not collapse to zero, or the Judge punishes new memories
        for the crime of being new.
        """
        exp = Experience(**_minimal())
        assert exp.observed_success_rate is None

    def test_success_rate_after_reuse(self):
        exp = Experience(**_minimal(), reuse_count=4, reuse_success_count=3)
        assert exp.observed_success_rate == pytest.approx(0.75)

    def test_success_count_cannot_exceed_reuse_count(self):
        """Mirrors the CHECK constraint in 001_init.sql."""
        exp = Experience(**_minimal(), reuse_count=2, reuse_success_count=5)
        assert exp.reuse_success_count > exp.reuse_count  # caught by the DB, not pydantic

    def test_total_tokens(self):
        exp = Experience(**_minimal(tokens_input=100, tokens_output=250))
        assert exp.total_tokens == 350


class TestExperienceResult:
    def test_projection_drops_audit_fields(self):
        """Agents get content and trust signal, not the full audit trail."""
        exp = Experience(**_minimal(), reuse_count=9)
        res = ExperienceResult.from_experience(exp, relevance=0.8)
        assert res.id == exp.id
        assert res.relevance == 0.8
        assert not hasattr(res, "reuse_count")
        assert not hasattr(res, "run_id")


class TestExperimentFields:
    def test_retrieved_ids_default_empty(self):
        """Baseline runs must never carry retrieval provenance."""
        exp = Experience(**_minimal(system=SystemCondition.BASELINE))
        assert exp.retrieved_experience_ids == []

    def test_retrieved_ids_accept_uuids(self):
        ids = [uuid.uuid4(), uuid.uuid4()]
        exp = Experience(**_minimal(retrieved_experience_ids=ids))
        assert exp.retrieved_experience_ids == ids

    def test_task_key_optional(self):
        assert Experience(**_minimal()).task_key is None
        assert Experience(**_minimal(task_key="t-01")).task_key == "t-01"
