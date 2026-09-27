"""
ActionCloud — embedding providers.

The default provider is `HashingEmbeddingProvider`: a deterministic, offline,
lexical embedder based on signed feature hashing (the "hashing trick") over
word unigrams, word bigrams and character 4-grams. It is not a neural model —
it captures shared vocabulary, not paraphrase meaning — but unlike the previous
SHA-256 "mock" vectors, similar texts get similar vectors, so cosine similarity
is a real signal and the retrieval experiments measure something.

For semantic embeddings set EMBEDDING_PROVIDER=gemini (gemini-embedding-001,
truncated to EMBEDDING_DIM) or EMBEDDING_PROVIDER=openai
(text-embedding-3-small, natively 1536-dim). Both need an API key and network.

Changing provider changes the vector space: re-embed (or `make reset`) before
mixing rows written by different providers.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from functools import lru_cache
from typing import List, Protocol

from .config import settings


class EmbeddingProvider(Protocol):
    dim: int
    name: str

    def embed(self, text: str) -> List[float]:
        ...


# --------------------------------------------------------------------------
# Default: lexical feature hashing
# --------------------------------------------------------------------------

_STOPWORDS = frozenset(
    """a an and are as at be by for from has have in into is it its of on or
    that the this to was were will with using use via when which while your
    our their than then there these those""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _stem(tok: str) -> str:
    """Tiny suffix stripper so 'pooling'/'pool' and 'errors'/'error' collide."""
    for suf in ("ation", "ing", "ers", "ies", "ed", "es", "er", "s"):
        if len(tok) > len(suf) + 3 and tok.endswith(suf):
            return tok[: -len(suf)]
    return tok


def tokenize(text: str) -> List[str]:
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def _unit(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


class HashingEmbeddingProvider:
    """Deterministic lexical embedder. No network, no keys, reproducible."""

    name = "hashing"

    W_UNIGRAM = 1.0
    W_BIGRAM = 0.6
    W_CHARGRAM = 0.25

    def __init__(self, dim: int = 1536) -> None:
        self.dim = dim

    def _features(self, text: str) -> dict[str, float]:
        toks = tokenize(text)
        feats: dict[str, float] = {}

        def add(key: str, w: float) -> None:
            feats[key] = feats.get(key, 0.0) + w

        for t in toks:
            add("u:" + t, self.W_UNIGRAM)
            if len(t) >= 5:
                padded = f"#{t}#"
                for i in range(len(padded) - 3):
                    add("c:" + padded[i : i + 4], self.W_CHARGRAM)
        for a, b in zip(toks, toks[1:]):
            add(f"b:{a}_{b}", self.W_BIGRAM)
        return feats

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        if not text or not text.strip():
            return vec
        for key, weight in self._features(text).items():
            h = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self.dim
            sign = 1.0 if h[4] & 1 else -1.0
            # log-scaled term frequency damps repeated words
            vec[idx] += sign * (1.0 + math.log(weight)) if weight > 1 else sign * weight
        return [round(x, 6) for x in _unit(vec)]


# Backward-compatible name used by older tests/scripts.
MockEmbeddingProvider = HashingEmbeddingProvider


# --------------------------------------------------------------------------
# Optional real providers
# --------------------------------------------------------------------------

class GeminiEmbeddingProvider:
    """gemini-embedding-001 with output_dimensionality = EMBEDDING_DIM."""

    name = "gemini"

    def __init__(self, dim: int = 1536, model: str = "gemini-embedding-001") -> None:
        self.dim = dim
        self.model = model

    def embed(self, text: str) -> List[float]:
        import httpx  # noqa: PLC0415

        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY not set")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:embedContent"
        resp = httpx.post(
            url,
            headers={"x-goog-api-key": key},
            json={
                "model": f"models/{self.model}",
                "content": {"parts": [{"text": text or " "}]},
                "outputDimensionality": self.dim,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        # Truncated Gemini embeddings are not unit-length; normalise for cosine.
        return _unit([float(x) for x in resp.json()["embedding"]["values"]])


class OpenAIEmbeddingProvider:
    """text-embedding-3-small (1536-dim)."""

    name = "openai"

    def __init__(self, dim: int = 1536, model: str = "text-embedding-3-small") -> None:
        self.dim = dim
        self.model = model

    def embed(self, text: str) -> List[float]:
        import httpx  # noqa: PLC0415

        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        resp = httpx.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": self.model, "input": text or " ", "dimensions": self.dim},
            timeout=30.0,
        )
        resp.raise_for_status()
        return _unit([float(x) for x in resp.json()["data"][0]["embedding"]])


@lru_cache(maxsize=8)
def _provider(name: str, dim: int) -> EmbeddingProvider:
    if name in ("hashing", "mock", "local"):
        # "mock" kept as an alias so old .env files keep working.
        return HashingEmbeddingProvider(dim=dim)
    if name in ("gemini", "google"):
        return GeminiEmbeddingProvider(dim=dim)
    if name == "openai":
        return OpenAIEmbeddingProvider(dim=dim)
    raise ValueError(f"Unknown embedding provider: {name!r}")


def get_embedding_provider() -> EmbeddingProvider:
    name = os.environ.get("EMBEDDING_PROVIDER", "hashing").lower()
    return _provider(name, settings.embedding_dim)


def cosine(a: List[float], b: List[float]) -> float:
    """Cosine similarity (vectors from providers here are unit-length)."""
    return sum(x * y for x, y in zip(a, b))


def experience_embedding_text(task: str, technologies: list[str] | None = None,
                              problem: str | None = None) -> str:
    """
    What gets embedded for a stored experience.

    Retrieval queries are task descriptions, so experiences are embedded on the
    task (plus technologies and the problem hit). Embedding the long solution
    text as well dilutes the task signal and lowers match quality.
    """
    parts = [task]
    if technologies:
        parts.append(" ".join(technologies))
    if problem:
        parts.append(problem)
    return " ".join(parts)
