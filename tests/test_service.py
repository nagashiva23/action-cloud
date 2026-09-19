from __future__ import annotations

import uuid
import pytest

from actioncloud.builder import CompactContextBuilder, count_tokens, filter_redundant_memories
from actioncloud.policy import MemorySelectionPolicy
from actioncloud.schema import AgentRole, MemoryTier
from actioncloud.service import MemoryService


class TestMemorySelectionPolicy:
    def test_default_policy_instantiation(self):
        pol = MemorySelectionPolicy()
        assert pol.candidate_k == 10
        assert pol.max_context_memories == 1
        assert pol.similarity_threshold == 0.70
        assert pol.redundancy_threshold == 0.85
        assert pol.context_token_budget == 1000
        assert pol.min_tier == MemoryTier.PRIVATE


class TestRedundancyFilter:
    def test_filter_duplicate_task_keys(self):
        candidates = [
            {"id": uuid.uuid4(), "task": "Task A", "task_key": "task-pgvector", "solution": "Sol 1"},
            {"id": uuid.uuid4(), "task": "Task B", "task_key": "task-pgvector", "solution": "Sol 2"},
        ]
        filtered = filter_redundant_memories(candidates, redundancy_threshold=0.85)
        assert len(filtered) == 1
        assert filtered[0]["task"] == "Task A"

    def test_filter_highly_similar_workflows(self):
        candidates = [
            {"id": uuid.uuid4(), "task": "Configure pgvector docker image", "solution": "Use pgvector/pgvector:pg16 image"},
            {"id": uuid.uuid4(), "task": "Configure pgvector docker image", "solution": "Use pgvector/pgvector:pg16 image"},
        ]
        filtered = filter_redundant_memories(candidates, redundancy_threshold=0.85)
        assert len(filtered) == 1


class TestCompactContextBuilder:
    def test_token_counting(self):
        text = "Hello world context string"
        assert count_tokens(text) > 0

    def test_build_context_with_procedural_workflow(self):
        exp_id = uuid.uuid4()
        candidates = [
            {
                "id": exp_id,
                "task": "Configure PostgreSQL pgvector container",
                "problem": "postgres:16-alpine image lacks vector extension",
                "solution": "Use pgvector/pgvector:pg16 image",
                "result": "pgvector ready",
                "tier": "shared",
                "success": True,
                "relevance": 0.95,
                "workflow": {
                    "prerequisites": ["Docker"],
                    "steps": ["Set image: pgvector/pgvector:pg16", "Run CREATE EXTENSION vector;"],
                    "pitfalls": ["Do not use standard postgres image"],
                },
            }
        ]
        pol = MemorySelectionPolicy(max_context_memories=1, context_token_budget=1000)
        context_str, selected_ids, cand_cnt, final_cnt = CompactContextBuilder.build_context(candidates, pol)

        assert cand_cnt == 1
        assert final_cnt == 1
        assert selected_ids == [exp_id]
        assert "Procedure:" in context_str
        assert "Pitfalls:" in context_str
        assert "pgvector/pgvector:pg16" in context_str


class TestMemoryServiceUnit:
    def test_service_initialization(self):
        service = MemoryService(sync_write=True)
        assert service.sync_write is True
