from __future__ import annotations

import hashlib
import math
import os
from typing import List, Protocol

from .config import settings


class EmbeddingProvider(Protocol):
    dim: int

    def embed(self, text: str) -> List[float]:
        ...


class MockEmbeddingProvider:
    """
    Deterministic fake embedding generator.

    Generates a normalized unit vector of dimension `dim` derived from the sha256
    hash of the text. Extremely fast, reproducible, and requires no API keys.
    """

    def __init__(self, dim: int = 1536) -> None:
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        if not text:
            return [0.0] * self.dim

        # Generate a pseudo-random seed sequence from hash
        h = hashlib.sha256(text.encode("utf-8")).digest()
        vec = []
        for i in range(self.dim):
            # Mix hash byte with index
            b = h[i % len(h)] ^ (i & 0xFF)
            val = (b / 255.0) - 0.5
            vec.append(val)

        # Normalize to unit length
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0:
            return [0.0] * self.dim

        return [round(x / norm, 6) for x in vec]


def get_embedding_provider() -> EmbeddingProvider:
    provider_name = os.environ.get("EMBEDDING_PROVIDER", "mock").lower()
    if provider_name == "mock":
        return MockEmbeddingProvider(dim=settings.embedding_dim)
    raise ValueError(f"Unknown embedding provider: {provider_name!r}")
