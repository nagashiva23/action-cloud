"""
Simulated agent environment (LLM_PROVIDER=sim).

WHAT THIS IS. A stand-in for a real LLM that lets the full ActionCloud
pipeline — storage, extraction, embedding, governed retrieval, context
building, reuse feedback, tier promotion/demotion — run end to end, offline,
reproducibly, with task outcomes that can be *verified*.

WHAT THIS IS NOT. Evidence about how much a real LLM benefits from memory.
The benefit of a correct memory is an input here (the parameters below), not
a finding. What the simulation does measure honestly is everything ActionCloud
itself controls:

  * whether retrieval puts the RIGHT memory in front of the agent
    (a memory from the same task family) or a wrong one, or none;
  * how many prompt tokens the injected context costs;
  * whether governance keeps subtly-wrong memories from spreading;
  * how those combine into success, tokens and cost under stated assumptions.

Re-run the same harness with LLM_PROVIDER=anthropic|gemini|groq to replace the
assumptions with a real model; the verifier and metrics are unchanged.

Behaviour per task (all randomness is seeded per task instance, so every arm
sees the same "luck" on the same task — common random numbers):

  1. The injected context is parsed back into memories (task + steps).
  2. A memory is APPLICABLE if its task belongs to the current task family.
     Memories marked FAILED are heeded as warnings and never followed.
  3. With an applicable memory the agent follows it (short answer, few
     output tokens). Following a correct memory succeeds unless it slips
     (p_follow_slip). Following an incomplete memory reproduces its gap and
     fails, unless the agent notices (1 - p_misled) and solves from scratch.
  4. Otherwise it solves from scratch (long answer, many output tokens):
     full procedure with p_scratch_success, else it misses 1 step
     (p_miss_one) or 2 steps. Irrelevant injected memories lower the success
     probability by p_distraction.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Optional

from ..builder import count_tokens
from ..llm import LLMResponse, Pricing
from . import load_procedures, load_tasks

_HEADER_RE = re.compile(r"^###\s+\d+\.\s+\[(?P<marker>[^\]]+)\]\s+\[[A-Z]+\]\s+(?P<task>.+?)\s*$")
_STEP_RE = re.compile(r"^\s+\d+\.\s+(?P<step>.+?)\s*$")


@dataclass
class ParsedMemory:
    task: str
    failed: bool
    steps: list[str] = field(default_factory=list)


def parse_context(prompt: str) -> tuple[list[ParsedMemory], str]:
    """Split a prompt into (injected memories, task text)."""
    memories: list[ParsedMemory] = []
    task_text = prompt
    if "## Your task" in prompt:
        head, task_text = prompt.split("## Your task", 1)
        current: Optional[ParsedMemory] = None
        for line in head.splitlines():
            m = _HEADER_RE.match(line)
            if m:
                current = ParsedMemory(m["task"], m["marker"].startswith("FAILED"))
                memories.append(current)
                continue
            s = _STEP_RE.match(line)
            if s and current is not None:
                current.steps.append(s["step"])
    return memories, task_text.strip()


@dataclass
class SimParams:
    p_scratch_success: float = 0.65
    p_miss_one: float = 0.60        # given a scratch failure: miss 1 step, else 2
    p_follow_slip: float = 0.03
    p_misled: float = 0.85
    p_distraction: float = 0.05
    scratch_out_base: int = 450     # output tokens: reasoning + exploration
    scratch_out_per_step: int = 60
    scratch_out_jitter: int = 300
    follow_out_base: int = 150      # output tokens: applying a known procedure
    follow_out_per_step: int = 25
    follow_out_jitter: int = 60
    latency_base_ms: int = 300
    latency_per_out_token_ms: float = 8.0


class SimulatedAgentLLM:
    """Implements the LLM protocol (`complete`) for the simulated environment."""

    def __init__(self, seed: int = 7, params: SimParams | None = None,
                 pricing: Pricing | None = None) -> None:
        self.model = "sim-agent-v1"
        self.seed = seed
        self.params = params or SimParams()
        # Priced like a mid-tier hosted model so $ figures are interpretable.
        self.pricing = pricing or Pricing(3.0, 15.0)
        self.procedures = load_procedures()
        self.task_to_key = {t["task"]: t["task_key"] for t in load_tasks()}
        self.instance: str = ""
        self.last_mode: str = ""

    def set_instance(self, instance_id: str) -> None:
        """Seed the next call. Same instance id => same luck in every arm."""
        self.instance = instance_id

    def _rng(self, salt: str) -> random.Random:
        digest = hashlib.sha256(f"{self.seed}|{self.instance}|{salt}".encode()).hexdigest()
        return random.Random(int(digest[:16], 16))

    def _key_for(self, task_text: str) -> Optional[str]:
        return self.task_to_key.get(task_text.strip())

    def complete(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        p = self.params
        memories, task_text = parse_context(prompt)
        key = self._key_for(task_text)
        if key is None:
            raise ValueError(f"sim provider only runs benchmark tasks; unknown task {task_text[:60]!r}")
        truth = [s["step"] for s in self.procedures[key]]

        applicable = [m for m in memories if not m.failed and self._key_for(m.task) == key and m.steps]
        irrelevant = [m for m in memories if self._key_for(m.task) != key]

        outcome = self._rng("outcome").random()
        drop = self._rng("drop")
        tokens = self._rng("tokens")
        steps: list[str]

        mem = applicable[0] if applicable else None
        if mem is not None and all(s in mem.steps for s in truth):
            self.last_mode = "followed_correct"
            steps = list(truth)
            if outcome < p.p_follow_slip:
                steps.pop(drop.randrange(len(steps)))
        elif mem is not None and outcome < p.p_misled:
            self.last_mode = "misled"
            steps = [s for s in mem.steps if s in truth]
        else:
            self.last_mode = "scratch"
            p_ok = p.p_scratch_success - (p.p_distraction if irrelevant else 0.0)
            second = self._rng("scratch").random()
            steps = list(truth)
            if second >= p_ok:
                n_missing = 1 if drop.random() < p.p_miss_one else 2
                for _ in range(n_missing):
                    steps.pop(drop.randrange(len(steps)))

        n = len(truth)
        if self.last_mode == "scratch":
            out_tokens = p.scratch_out_base + p.scratch_out_per_step * n + tokens.randrange(p.scratch_out_jitter)
        else:
            out_tokens = p.follow_out_base + p.follow_out_per_step * n + tokens.randrange(p.follow_out_jitter)

        text = "Procedure applied:\n" + "\n".join(f"STEP: {s}" for s in steps)
        in_tokens = count_tokens((system or "") + prompt)
        return LLMResponse(
            text=text,
            tokens_input=in_tokens,
            tokens_output=out_tokens,
            latency_ms=int(p.latency_base_ms + p.latency_per_out_token_ms * out_tokens),
            model=self.model,
            cost_usd=self.pricing.cost(in_tokens, out_tokens),
        )
