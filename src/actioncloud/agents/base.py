from __future__ import annotations

import hashlib
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from ..benchmark.verifier import TaskVerifier
from ..llm import LLM, get_llm
from ..policy import MemorySelectionPolicy
from ..schema import AgentRole, SystemCondition

log = logging.getLogger(__name__)

ANSWER_FORMAT = (
    "Answer with the concrete procedure you would follow, one step per line, "
    "each line starting with 'STEP:'."
)


@dataclass
class TaskSpec:
    """
    One unit of work.

    Tasks sharing a `task_key` are paraphrases of the same logical problem.
    Solving one from scratch when a correct solution to its key already
    exists is the redundant work ActionCloud exists to eliminate.
    """

    task: str
    task_key: Optional[str] = None
    technologies: list[str] = field(default_factory=list)
    expected_tools: list[str] = field(default_factory=list)
    instance_id: Optional[str] = None  # seeds the simulated environment


@dataclass
class TaskOutcome:
    task: str
    task_key: Optional[str]
    success: bool                 # strict, verified outcome
    self_check_success: bool      # lenient self-assessment the agent stored
    result: str
    tokens_input: int
    tokens_output: int
    execution_time_ms: int
    cost_usd: float
    tool_calls: int
    retrieved_ids: list[uuid.UUID]
    retrieved_count: int
    injected_task_keys: list[Optional[str]] = field(default_factory=list)
    context_tokens: int = 0
    candidate_count: int = 0
    mode: str = ""
    experience_id: Optional[uuid.UUID] = None

    @property
    def total_tokens(self) -> int:
        return self.tokens_input + self.tokens_output

    @property
    def relevant_injection(self) -> bool:
        return self.task_key is not None and self.task_key in self.injected_task_keys


class Agent:
    """A task-execution unit that reads from and writes to ActionCloud."""

    role: AgentRole = AgentRole.RESEARCH
    system_prompt: str = "You are a helpful autonomous agent. Complete the task concisely."

    def __init__(
        self,
        agent_id: str,
        client: Any,
        run_id: str,
        use_memory: bool,
        llm: Optional[LLM] = None,
        policy: Optional[MemorySelectionPolicy] = None,
        verifier: Optional[TaskVerifier] = None,
        report_reuse: bool = True,
        feedback_rate: float = 1.0,
        scope_run_id: Optional[str] = None,
        memory_limit: Optional[int] = None,  # legacy: maps to policy.max_context_memories
        **_: Any,
    ) -> None:
        self.agent_id = agent_id
        self.client = client
        self.run_id = run_id
        self.use_memory = use_memory
        self.llm = llm or get_llm()
        self.policy = policy or MemorySelectionPolicy.from_env()
        if memory_limit is not None:
            self.policy.max_context_memories = memory_limit
        self.verifier = verifier or TaskVerifier()
        self.report_reuse = report_reuse
        # Probability that the true (downstream) outcome of a task is ever
        # observed — tests run, CI reports, a deploy visibly breaks. 1.0 when
        # the verifier result is always available (real-LLM runs).
        self.feedback_rate = feedback_rate
        self.scope_run_id = scope_run_id

    @property
    def condition(self) -> SystemCondition:
        return SystemCondition.ACTIONCLOUD if self.use_memory else SystemCondition.BASELINE

    def _feedback_observed(self, spec: TaskSpec) -> bool:
        if self.feedback_rate >= 1.0:
            return True
        seed = hashlib.sha256(f"feedback|{spec.instance_id or spec.task}".encode()).hexdigest()
        return random.Random(int(seed[:16], 16)).random() < self.feedback_rate

    def _build_prompt(self, spec: TaskSpec, context: str) -> str:
        task = f"## Your task\n{spec.task}"
        return f"{context}\n\n{task}" if context else task

    def run_task(self, spec: TaskSpec) -> TaskOutcome:
        """
        Retrieve (memory arm only) -> solve -> verify -> store -> report reuse.

        execution_time_ms = retrieval wall time + model latency. Storage is
        excluded: it is asynchronous in production (HTTP 202 + worker).
        """
        ctx: dict[str, Any] = {
            "context": "", "injected_experience_ids": [], "injected_task_keys": [],
            "context_tokens": 0, "candidate_count": 0,
        }
        t0 = time.perf_counter()
        if self.use_memory:
            ctx = self.client.context(
                query=spec.task,
                agent_id=self.agent_id,
                agent_role=self.role,
                technologies=spec.technologies,
                policy=self.policy,
                scope_run_id=self.scope_run_id,
            )
        retrieval_ms = int((time.perf_counter() - t0) * 1000)

        if spec.instance_id and hasattr(self.llm, "set_instance"):
            self.llm.set_instance(spec.instance_id)
        response = self.llm.complete(
            self._build_prompt(spec, ctx["context"]),
            system=f"{self.system_prompt}\n{ANSWER_FORMAT}",
        )

        # Tasks outside the benchmark have no ground truth; they are treated
        # as successful (the old behaviour) rather than failing the run.
        verdict = (
            self.verifier.verify(spec.task_key, response.text)
            if spec.task_key and spec.task_key in self.verifier.procedures
            else None
        )
        success = verdict.strict if verdict else True
        self_check = verdict.lenient if verdict else True
        injected = [uuid.UUID(str(i)) for i in ctx["injected_experience_ids"]]

        outcome = TaskOutcome(
            task=spec.task,
            task_key=spec.task_key,
            success=success,
            self_check_success=self_check,
            result=response.text[:2000],
            tokens_input=response.tokens_input,
            tokens_output=response.tokens_output,
            execution_time_ms=retrieval_ms + response.latency_ms,
            cost_usd=response.cost_usd,
            tool_calls=len(spec.expected_tools),
            retrieved_ids=injected,
            retrieved_count=len(injected),
            injected_task_keys=list(ctx.get("injected_task_keys", [])),
            context_tokens=ctx.get("context_tokens", 0),
            candidate_count=ctx.get("candidate_count", 0),
            mode=getattr(self.llm, "last_mode", ""),
        )

        # Contribute back. The agent stores its OWN (lenient) judgement of
        # success; it cannot see the strict verdict any more than a real agent
        # can. The baseline arm stores too, for its token/latency figures.
        outcome.experience_id = self.client.store(
            agent_id=self.agent_id,
            agent_role=self.role,
            task=spec.task,
            action=response.text,
            result="Completed" if self_check else "Did not complete",
            success=self_check,
            run_id=self.run_id,
            system=self.condition,
            solution=response.text if self_check else None,
            tools_used=spec.expected_tools,
            technologies=spec.technologies,
            tokens_input=response.tokens_input,
            tokens_output=response.tokens_output,
            tool_calls=len(spec.expected_tools),
            execution_time_ms=outcome.execution_time_ms,
            cost_usd=response.cost_usd,
            task_key=spec.task_key,
            retrieved_experience_ids=injected,
        )

        # Governance feedback, sent only when the true outcome is observed:
        #  * for each injected memory: did reusing it actually work?
        #  * for our own new memory: if the task really failed, say so — a
        #    negative self-report, which the Judge counts (positive ones it
        #    ignores).
        if self.report_reuse and self._feedback_observed(spec):
            for exp_id in injected:
                self.client.report_reuse(exp_id, success=success, agent_id=self.agent_id)
            if not success and self_check and outcome.experience_id is not None:
                self.client.report_reuse(outcome.experience_id, success=False,
                                         agent_id=self.agent_id)

        log.debug(
            "%s [%s] key=%s success=%s injected=%s tokens=%d",
            self.agent_id, self.condition.value, spec.task_key, success,
            outcome.injected_task_keys, outcome.total_tokens,
        )
        return outcome
