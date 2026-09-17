from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Protocol


@dataclass
class LLMResponse:
    """
    One model call, plus everything the experiment needs to measure it.

    Token counts and cost live here rather than being estimated later, because
    reconstructing them after the fact is guesswork and the entire cost model
    in Phase 3 rests on these numbers being real.
    """

    text: str
    tokens_input: int
    tokens_output: int
    latency_ms: int
    model: str
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.tokens_input + self.tokens_output


@dataclass
class Pricing:
    """
    Per-million-token pricing.

    Deliberately configurable rather than hardcoded: model prices change, and a
    stale constant buried in the code would silently corrupt every cost figure
    in your final report. Set these from the provider's pricing page on the day
    you run the experiment, and record the values you used.
    """

    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0

    def cost(self, tokens_in: int, tokens_out: int) -> float:
        return (
            tokens_in / 1_000_000 * self.input_per_mtok
            + tokens_out / 1_000_000 * self.output_per_mtok
        )


class LLM(Protocol):
    """Minimal interface every provider must satisfy."""

    model: str

    def complete(self, prompt: str, system: Optional[str] = None) -> LLMResponse: ...


# --------------------------------------------------------------------------
# Mock provider — Phase 1 default
# --------------------------------------------------------------------------

@dataclass
class MockLLM:
    """
    Deterministic fake model. No network, no cost, no API key.

    Token counts are derived from a hash of the prompt, so they are stable
    across runs but still vary between different prompts — enough to exercise
    the measurement path realistically without pretending to be a real model.

    `simulated_latency_ms` is not decorative. A real model call takes seconds,
    and if you develop against an instantly-returning mock you will not notice
    accidental synchronous blocking in the write path until it distorts the
    experiment. A small delay keeps you honest.
    """

    model: str = "mock-llm-v1"
    pricing: Pricing = field(default_factory=lambda: Pricing(0.25, 1.25))
    simulated_latency_ms: int = 50

    def complete(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        started = time.perf_counter()
        if self.simulated_latency_ms:
            time.sleep(self.simulated_latency_ms / 1000)

        digest = hashlib.sha256((system or "").encode() + prompt.encode()).hexdigest()
        seed = int(digest[:8], 16)

        # Roughly proportional to input length, with deterministic jitter.
        tokens_input = max(1, len(prompt) // 4 + len(system or "") // 4)
        tokens_output = 120 + (seed % 400)

        text = (
            f"[mock:{digest[:8]}] Completed the task described as: "
            f"{prompt[:120]}{'...' if len(prompt) > 120 else ''}"
        )

        latency_ms = int((time.perf_counter() - started) * 1000)
        return LLMResponse(
            text=text,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            latency_ms=latency_ms,
            model=self.model,
            cost_usd=self.pricing.cost(tokens_input, tokens_output),
        )


# --------------------------------------------------------------------------
# Real provider — wire up in Phase 3
# --------------------------------------------------------------------------

@dataclass
class AnthropicLLM:
    """
    Real provider for the Phase 3 experiment.

    Left unwired in Phase 1 on purpose. Add `anthropic` to requirements.txt and
    set ANTHROPIC_API_KEY when you get here.

    One warning for when you do: read token counts from the API response's
    usage field, never estimate them from string length. Your entire cost model
    and hypothesis H3 depend on these being the provider's actual numbers.
    """

    model: str = "claude-sonnet-4-5"
    pricing: Pricing = field(default_factory=lambda: Pricing(3.0, 15.0))
    max_tokens: int = 2048
    _client: object = field(default=None, repr=False)

    def _ensure_client(self):
        if self._client is None:
            try:
                import anthropic  # noqa: PLC0415
            except ImportError as e:
                raise RuntimeError(
                    "anthropic package not installed. Phase 1 uses MockLLM; "
                    "add `anthropic` to requirements.txt for Phase 3."
                ) from e
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY not set")
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def complete(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        client = self._ensure_client()
        started = time.perf_counter()
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        resp = client.messages.create(**kwargs)
        latency_ms = int((time.perf_counter() - started) * 1000)

        tokens_in = resp.usage.input_tokens
        tokens_out = resp.usage.output_tokens
        text = "".join(block.text for block in resp.content if block.type == "text")

        return LLMResponse(
            text=text,
            tokens_input=tokens_in,
            tokens_output=tokens_out,
            latency_ms=latency_ms,
            model=self.model,
            cost_usd=self.pricing.cost(tokens_in, tokens_out),
        )


@dataclass
class GeminiLLM:
    """
    Google Gemini API provider (gemini-1.5-flash / gemini-2.0-flash).
    Free Tier: 15 Requests Per Minute, 1,000,000 Tokens Per Minute.
    """

    model: str = "gemini-1.5-flash"
    pricing: Pricing = field(default_factory=lambda: Pricing(0.075, 0.30))
    max_tokens: int = 2048

    def complete(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        import httpx  # noqa: PLC0415

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY not set in environment")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={api_key}"

        contents = []
        if system:
            contents.append({"role": "user", "parts": [{"text": f"System Instruction: {system}"}]})
            contents.append({"role": "model", "parts": [{"text": "Understood."}]})

        contents.append({"role": "user", "parts": [{"text": prompt}]})

        payload = {
            "contents": contents,
            "generationConfig": {"maxOutputTokens": self.max_tokens},
        }

        started = time.perf_counter()
        resp = httpx.post(url, json=payload, timeout=30.0)
        resp.raise_for_status()
        latency_ms = int((time.perf_counter() - started) * 1000)

        data = resp.json()
        candidates = data.get("candidates", [])
        text = ""
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)

        usage = data.get("usageMetadata", {})
        tokens_in = usage.get("promptTokenCount", max(1, len(prompt) // 4))
        tokens_out = usage.get("candidatesTokenCount", max(1, len(text) // 4))

        return LLMResponse(
            text=text,
            tokens_input=tokens_in,
            tokens_output=tokens_out,
            latency_ms=latency_ms,
            model=self.model,
            cost_usd=self.pricing.cost(tokens_in, tokens_out),
        )


@dataclass
class GroqLLM:
    """
    Groq API provider (llama-3.3-70b-versatile).
    Free Tier: 30 Requests Per Minute.
    """

    model: str = "llama-3.3-70b-versatile"
    pricing: Pricing = field(default_factory=lambda: Pricing(0.59, 0.79))
    max_tokens: int = 2048

    def complete(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        import httpx  # noqa: PLC0415

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set in environment")

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }

        started = time.perf_counter()
        resp = httpx.post(url, headers=headers, json=payload, timeout=30.0)
        resp.raise_for_status()
        latency_ms = int((time.perf_counter() - started) * 1000)

        data = resp.json()
        choices = data.get("choices", [])
        text = choices[0]["message"]["content"] if choices else ""

        usage = data.get("usage", {})
        tokens_in = usage.get("prompt_tokens", max(1, len(prompt) // 4))
        tokens_out = usage.get("completion_tokens", max(1, len(text) // 4))

        return LLMResponse(
            text=text,
            tokens_input=tokens_in,
            tokens_output=tokens_out,
            latency_ms=latency_ms,
            model=self.model,
            cost_usd=self.pricing.cost(tokens_in, tokens_out),
        )


def get_llm(kind: str | None = None) -> LLM:
    """Factory. Defaults to mock so nothing accidentally costs money."""
    kind = (kind or os.environ.get("LLM_PROVIDER", "mock")).lower()
    if kind == "mock":
        return MockLLM()
    if kind == "anthropic":
        return AnthropicLLM()
    if kind in ("gemini", "google"):
        return GeminiLLM()
    if kind == "groq":
        return GroqLLM()
    raise ValueError(f"unknown LLM provider: {kind!r}")
