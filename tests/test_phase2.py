"""
Unit tests for ActionCloud Phase 2 features:
  - MemoryJudge (tier transitions, promotion, demotion)
  - ExperienceExtractor (workflow and triple generation)
  - Hashing embedding provider (1536-dim, similarity-preserving)
  - Agent Role Registry (all 12 roles)
"""

from __future__ import annotations

import uuid
import pytest

from actioncloud.agents.roles import (
    ROLE_REGISTRY,
    AgentRole,
    CodingAgent,
    DataAnalysisAgent,
    DeploymentAgent,
    DocumentationAgent,
    ResearchAgent,
    TestingAgent,
)
from actioncloud.embeddings import MockEmbeddingProvider
from actioncloud.extractor import ExperienceExtractor
from actioncloud.judge import MemoryJudge
from actioncloud.schema import Experience, MemoryTier, SystemCondition


def _sample_experience(success: bool = True) -> Experience:
    return Experience(
        id=uuid.uuid4(),
        agent_id="test-agent",
        agent_role=AgentRole.CODING,
        task="Configure pgvector on PostgreSQL 16 container",
        problem="pgvector extension missing from base postgres image",
        action="Switched Docker image to pgvector/pgvector:pg16",
        solution="Updated docker-compose.yml to use pgvector/pgvector:pg16 image",
        result="Postgres started with vector extension active",
        tools_used=["docker", "bash"],
        technologies=["postgres", "docker", "pgvector"],
        success=success,
        run_id="run-phase2-test",
        system=SystemCondition.ACTIONCLOUD,
    )


class TestMemoryJudge:
    def test_initial_tier_assignment_success(self):
        exp = _sample_experience(success=True)
        tier, confidence, reason = MemoryJudge.assign_initial_tier(exp)
        assert tier == MemoryTier.AGENT
        assert confidence >= 0.5
        assert "Initial creation" in reason

    def test_initial_tier_assignment_failure(self):
        exp = _sample_experience(success=False)
        tier, confidence, reason = MemoryJudge.assign_initial_tier(exp)
        assert tier == MemoryTier.PRIVATE
        assert confidence == 0.3

    def test_promotion_agent_to_shared(self):
        new_tier, conf, reason = MemoryJudge.decide(MemoryTier.AGENT, 0.6, 1, 1)
        assert new_tier is MemoryTier.SHARED
        assert conf > 0.6
        assert "shared" in reason

    def test_no_change_without_evidence(self):
        assert MemoryJudge.decide(MemoryTier.AGENT, 0.6, 0, 0) is None
        assert MemoryJudge.decide(MemoryTier.AGENT, 0.6, 1, 0) is None

    def test_shared_to_validated_boundary(self):
        assert MemoryJudge.decide(MemoryTier.SHARED, 0.7, 2, 2) is None          # too few
        assert MemoryJudge.decide(MemoryTier.SHARED, 0.7, 5, 3) is None          # 60% < 80%
        tier, _, _ = MemoryJudge.decide(MemoryTier.SHARED, 0.7, 5, 4)            # 80%
        assert tier is MemoryTier.VALIDATED

    def test_validated_to_organizational(self):
        assert MemoryJudge.decide(MemoryTier.VALIDATED, 0.85, 9, 9) is None
        tier, _, _ = MemoryJudge.decide(MemoryTier.VALIDATED, 0.85, 10, 9)
        assert tier is MemoryTier.ORGANIZATIONAL

    def test_demotion_low_success_rate(self):
        tier, conf, reason = MemoryJudge.decide(MemoryTier.SHARED, 0.7, 4, 1)   # 25%
        assert tier is MemoryTier.PRIVATE
        assert conf < 0.5
        assert "Quarantined" in reason

    def test_private_is_terminal_for_automatic_governance(self):
        assert MemoryJudge.decide(MemoryTier.PRIVATE, 0.3, 5, 5) is None

    def test_posterior_confidence(self):
        assert MemoryJudge.posterior_confidence(0, 0) == pytest.approx(0.6)
        assert MemoryJudge.posterior_confidence(0, 1) == pytest.approx(0.4)
        assert MemoryJudge.posterior_confidence(1, 1) > 0.7


class TestMockEmbeddingProvider:
    def test_dimensions(self):
        provider = MockEmbeddingProvider(dim=1536)
        vec = provider.embed("PostgreSQL vector search query")
        assert len(vec) == 1536

    def test_determinism(self):
        provider = MockEmbeddingProvider(dim=1536)
        vec1 = provider.embed("pgvector test query")
        vec2 = provider.embed("pgvector test query")
        assert vec1 == vec2

    def test_different_texts_produce_different_vectors(self):
        provider = MockEmbeddingProvider(dim=1536)
        vec1 = provider.embed("pgvector test query")
        vec2 = provider.embed("completely unrelated text string")
        assert vec1 != vec2

    def test_similar_texts_are_closer_than_unrelated(self):
        """The old SHA-256 embedder failed this: similarity was noise."""
        from actioncloud.embeddings import cosine
        p = MockEmbeddingProvider(dim=1536)
        a = p.embed("configure postgres connection pool")
        b = p.embed("configure postgres connection pooling")
        c = p.embed("bake a chocolate cake")
        assert cosine(a, b) > 0.6
        assert cosine(a, c) < 0.1

    def test_unit_length(self):
        import math
        v = MockEmbeddingProvider(dim=1536).embed("pgvector hnsw index")
        assert math.sqrt(sum(x * x for x in v)) == pytest.approx(1.0, abs=1e-4)


class TestExperienceExtractor:
    def test_heuristic_extraction_uses_solution_steps(self):
        exp = _sample_experience().model_copy(update={
            "solution": "STEP: Switch image to pgvector/pgvector:pg16\nSTEP: Recreate the volume"
        })
        workflow, _ = ExperienceExtractor().extract(exp)
        assert workflow["steps"] == ["Switch image to pgvector/pgvector:pg16", "Recreate the volume"]

    def test_fallback_extraction(self):
        exp = _sample_experience()
        extractor = ExperienceExtractor()
        workflow, triples = extractor.extract(exp)

        assert workflow is not None
        assert "steps" in workflow
        assert len(triples) > 0
        assert "subject" in triples[0]


class TestRoleRegistry:
    def test_all_twelve_roles_registered(self):
        assert set(ROLE_REGISTRY) == set(AgentRole)
        assert ROLE_REGISTRY[AgentRole.CODING] == CodingAgent
        assert ROLE_REGISTRY[AgentRole.RESEARCH] == ResearchAgent
        assert ROLE_REGISTRY[AgentRole.TESTING] == TestingAgent
        assert ROLE_REGISTRY[AgentRole.DEPLOYMENT] == DeploymentAgent
        assert ROLE_REGISTRY[AgentRole.DOCUMENTATION] == DocumentationAgent
        assert ROLE_REGISTRY[AgentRole.DATA_ANALYSIS] == DataAnalysisAgent
