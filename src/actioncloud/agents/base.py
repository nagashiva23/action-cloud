"""
Base agent.

The single most important design decision in this file: **System A (baseline)
and System B (ActionCloud) are the same class with one flag flipped.**

    Agent(..., use_memory=False)   -> System A, stateless
    Agent(..., use_memory=True)    -> System B, memory-enabled

They share the prompt construction, the measurement code, the storage call, and
the error handling. The only divergence is whether `search()` runs and whether
its results are injected into the prompt.

This matters for the experiment's validity. If the two arms were separate
implementations, any measured difference could be an artifact of one being
written more carefully than the other, and a sceptical evaluator would be right
to say so. With one code path, the difference is attributable to the memory
layer because there is nothing else it could be attributable to.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from ..client import ActionCloudClient
from ..llm import LLM, get_llm
from ..schema import AgentRole, ExperienceResult, MemoryTier, SystemCondition

log = logging.getLogger(__name__)


@dataclass
class TaskSpec:
    """
    One unit of work.

    `task_key` is what makes redundancy measurable: several TaskSpecs in the
    100-task workload deliberately share a key, marking them as the same
    logical problem. Two agents solving the same task_key from scratch is,
    by definition, the redundant work this project exists to eliminate.
    """

    task: str
    task_key: Optional[str] = None
    technologies: list[str] = field(default_factory=list)
    expected_tools: list[str] = field(default_factory=list)


@dataclass
class TaskOutcome:
    """Everything one task run produced, before it becomes an Experience."""

    task: str
    task_key: Optional[str]
    success: bool
    result: str
    tokens_input: int
    tokens_output: int
    execution_time_ms: int
    cost_usd: float
    tool_calls: int
    retrieved_ids: list[uuid.UUID]
    retrieved_count: int
    experience_id: Optional[uuid.UUID] = None

    @property
    def total_tokens(self) -> int:
        return self.tokens_input + self.tokens_output


class Agent:
    """A task-execution unit that reads from and writes to ActionCloud."""

    #: Overridden per role in roles.py
    role: AgentRole = AgentRole.RESEARCH
    system_prompt: str = "You are a helpful autonomous agent. Complete the task concisely."

    def __init__(
        self,
        agent_id: str,
        client: ActionCloudClient,
        run_id: str,
        use_memory: bool,
        llm: Optional[LLM] = None,
        memory_limit: int = 3,
        min_tier: MemoryTier = MemoryTier.PRIVATE,
    ) -> None:
        self.agent_id = agent_id
        self.client = client
        self.run_id = run_id
        self.use_memory = use_memory
        self.llm = llm or get_llm()
        self.memory_limit = memory_limit
        self.min_tier = min_tier

    @property
    def condition(self) -> SystemCondition:
        return SystemCondition.ACTIONCLOUD if self.use_memory else SystemCondition.BASELINE

    # -- prompt construction ----------------------------------------------

    def _format_memory(self, experiences: list[ExperienceResult]) -> str:
        """
        Render retrieved experiences into prompt context.

        Only successful experiences are offered as guidance. A failed
        experience still has value — knowing what *didn't* work is real
        information — but presenting the two identically invites the model to
        copy a failure, so failures are labelled explicitly.
        """
        if not experiences:
            return ""

        lines = ["## Relevant prior experience from other agents", ""]
        for i, exp in enumerate(experiences, 1):
            marker = "WORKED" if exp.success else "FAILED — avoid this approach"
            lines.append(f"### {i}. [{marker}] {exp.task}")
            if exp.problem:
                lines.append(f"- Problem encountered: {exp.problem}")
            if exp.solution:
                lines.append(f"- Solution: {exp.solution}")
            lines.append(f"- Outcome: {exp.result}")
            if exp.workflow:  # populated from Phase 2 onward
                lines.append(f"- Reusable workflow: {exp.workflow}")
            lines.append("")

        lines.append(
            "Use the above if it applies. If it does not, solve the task directly "
            "and ignore it."
        )
        return "\n".join(lines)

    def _build_prompt(self, spec: TaskSpec, memory: list[ExperienceResult]) -> str:
        context = self._format_memory(memory)
        if context:
            return f"{context}\n\n## Your task\n{spec.task}"
        return f"## Your task\n{spec.task}"

    # -- outcome interpretation -------------------------------------------

    def evaluate(self, spec: TaskSpec, response_text: str) -> tuple[bool, str]:
        """
        Decide whether the task succeeded.

        Trivially permissive here because Phase 1 is verifying plumbing, not
        capability. Before the Phase 3 run this must become a real check —
        per-task assertions, or an LLM judge with a fixed rubric. Whatever you
        choose, it has to be identical across both arms, or your success-rate
        comparison measures your grader instead of your system.
        """
        return True, response_text[:500]

    # -- the main loop -----------------------------------------------------

    def run_task(self, spec: TaskSpec) -> TaskOutcome:
        """
        Execute one task end to end.

        Note what is and isn't timed: `execution_time_ms` covers retrieval plus
        model call, because that is the latency an agent actually experiences.
        It excludes the store() call, since that returns 202 immediately and the
        real work happens in a worker — charging asynchronous processing to the
        agent's clock would understate exactly the benefit the queue provides.
        """
        started = time.perf_counter()

        # 1. Retrieve — the only step that differs between the two arms.
        memory: list[ExperienceResult] = []
        if self.use_memory:
            memory = self.client.search(
                query=spec.task,
                limit=self.memory_limit,
                min_tier=self.min_tier,
                agent_id=self.agent_id,  # this agent's identity, not the client's
            )
            log.debug("retrieved %d prior experience(s)", len(memory))

        # 2. Solve.
        prompt = self._build_prompt(spec, memory)
        response = self.llm.complete(prompt, system=self.system_prompt)

        # 3. Judge.
        success, result_text = self.evaluate(spec, response.text)
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        outcome = TaskOutcome(
            task=spec.task,
            task_key=spec.task_key,
            success=success,
            result=result_text,
            tokens_input=response.tokens_input,
            tokens_output=response.tokens_output,
            execution_time_ms=elapsed_ms,
            cost_usd=response.cost_usd,
            tool_calls=len(spec.expected_tools),
            retrieved_ids=[e.id for e in memory],
            retrieved_count=len(memory),
        )

        # 4. Contribute back.
        #
        # The baseline arm stores too. It never *reads*, but its experiences are
        # what give you baseline token and latency figures to compare against —
        # without them there is no experiment, only a demo.
        outcome.experience_id = self.client.store(
            # Identity is passed explicitly: one client is shared by several
            # agents, and letting the client's own defaults win would attribute
            # every experience to whichever agent happened to create it.
            agent_id=self.agent_id,
            agent_role=self.role,
            task=spec.task,
            action=f"Invoked {self.llm.model} with {len(memory)} prior experience(s) in context",
            result=result_text,
            success=success,
            run_id=self.run_id,
            system=self.condition,
            problem=None,
            solution=result_text if success else None,
            tools_used=spec.expected_tools,
            technologies=spec.technologies,
            tokens_input=response.tokens_input,
            tokens_output=response.tokens_output,
            tool_calls=len(spec.expected_tools),
            execution_time_ms=elapsed_ms,
            cost_usd=response.cost_usd,
            task_key=spec.task_key,
            retrieved_experience_ids=outcome.retrieved_ids,
        )

        log.info(
            "%s [%s] task=%r success=%s tokens=%d retrieved=%d",
            self.agent_id,
            self.condition.value,
            spec.task[:48],
            success,
            outcome.total_tokens,
            outcome.retrieved_count,
        )
        return outcome
