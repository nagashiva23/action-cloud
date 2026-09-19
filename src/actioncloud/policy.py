from __future__ import annotations

from dataclasses import dataclass, field
import os
from .schema import MemoryTier


@dataclass
class MemorySelectionPolicy:
    """
    Configurable memory selection policy governing retrieval, filtering,
    redundancy elimination, and context token budgeting.
    """

    candidate_k: int = 10
    max_context_memories: int = 1
    similarity_threshold: float = 0.70
    redundancy_threshold: float = 0.85
    context_token_budget: int = 1000
    min_tier: MemoryTier = MemoryTier.PRIVATE

    @classmethod
    def from_env(cls) -> MemorySelectionPolicy:
        return cls(
            candidate_k=int(os.environ.get("MEMORY_CANDIDATE_K", "10")),
            max_context_memories=int(os.environ.get("MEMORY_MAX_CONTEXT_MEMORIES", "1")),
            similarity_threshold=float(os.environ.get("MEMORY_SIMILARITY_THRESHOLD", "0.70")),
            redundancy_threshold=float(os.environ.get("MEMORY_REDUNDANCY_THRESHOLD", "0.85")),
            context_token_budget=int(os.environ.get("MEMORY_CONTEXT_TOKEN_BUDGET", "1000")),
            min_tier=MemoryTier(os.environ.get("MEMORY_MIN_TIER", "private")),
        )
