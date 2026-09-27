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
    # Calibrated on the 136-task benchmark with the hashing embedder
    # (query = task + technologies): 0.30 keeps ~91 % of same-family matches
    # while ~1 % of different-family pairs clear it. Re-calibrate with
    # scripts/calibrate_threshold.py if you switch embedding provider.
    similarity_threshold: float = 0.30
    redundancy_threshold: float = 0.85
    context_token_budget: int = 1000
    # Ranking bonus per unit of Memory-Judge confidence (0..1).
    trust_weight: float = 0.10
    # Memories whose outcome record has pushed confidence below this are not
    # injected. New successful memories start at 0.6; one failed outcome
    # report -> 0.40 (excluded); one success -> 0.73; 1 success + 1 failure
    # -> 0.55 (kept).
    min_confidence: float = 0.45
    min_tier: MemoryTier = MemoryTier.PRIVATE

    @classmethod
    def from_env(cls) -> MemorySelectionPolicy:
        return cls(
            candidate_k=int(os.environ.get("MEMORY_CANDIDATE_K", "10")),
            max_context_memories=int(os.environ.get("MEMORY_MAX_CONTEXT_MEMORIES", "1")),
            similarity_threshold=float(os.environ.get("MEMORY_SIMILARITY_THRESHOLD", "0.30")),
            redundancy_threshold=float(os.environ.get("MEMORY_REDUNDANCY_THRESHOLD", "0.85")),
            context_token_budget=int(os.environ.get("MEMORY_CONTEXT_TOKEN_BUDGET", "1000")),
            trust_weight=float(os.environ.get("MEMORY_TRUST_WEIGHT", "0.10")),
            min_confidence=float(os.environ.get("MEMORY_MIN_CONFIDENCE", "0.45")),
            min_tier=MemoryTier(os.environ.get("MEMORY_MIN_TIER", "private")),
        )
