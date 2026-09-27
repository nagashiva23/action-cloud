"""
Agent client library.

Agents talk to ActionCloud through this, never through raw HTTP and never
through the database. Two reasons that matters:

  1. Governance is only enforceable if there is exactly one way in.
  2. When Phase 2 changes retrieval (hybrid ranking, tier gating, the utility
     score), agents should not need editing. They call `.search()` and the
     meaning of the results improves underneath them.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

import httpx

from .schema import AgentRole, ExperienceResult, MemoryTier, SystemCondition

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:8000"


class ActionCloudError(RuntimeError):
    pass


class ActionCloudClient:
    """
    Thin synchronous client.

    Synchronous on purpose: agents in this project execute one task at a time,
    and async would add concurrency semantics you would then have to reason
    about while interpreting latency measurements.
    """

    def __init__(
        self,
        agent_id: str,
        agent_role: AgentRole,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 10.0,
    ) -> None:
        # These are *defaults*. Every call accepts an override, because a single
        # client is often shared by several agents with different identities —
        # and if the client's identity silently won, every experience would be
        # attributed to the wrong agent and provenance would be worthless.
        self.agent_id = agent_id
        self.agent_role = agent_role
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout)

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "ActionCloudClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- reads -------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        return self._http.get("/health").json()

    def search(
        self,
        query: str,
        limit: int = 5,
        min_tier: MemoryTier = MemoryTier.PRIVATE,
        include_own_private: bool = True,
        agent_id: Optional[str] = None,
        agent_role: Optional[AgentRole] = None,
    ) -> list[ExperienceResult]:
        """
        Find prior experiences relevant to a task.

        Failures are swallowed and logged rather than raised. This is a
        deliberate availability choice: memory is an *optimisation*, so if
        ActionCloud is down an agent should degrade to baseline behaviour and
        still finish its task, not crash. The alternative would make the memory
        layer a single point of failure for the entire fleet.
        """
        params: dict[str, Any] = {"q": query, "limit": limit, "min_tier": min_tier.value}
        if include_own_private:
            params["agent_id"] = agent_id or self.agent_id
            # Role makes same-role AGENT-tier memories visible (see VISIBILITY_SQL).
            params["agent_role"] = (agent_role or self.agent_role).value

        try:
            resp = self._http.get("/search", params=params)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            log.warning("search failed (%s) — continuing without memory", e)
            return []

        return [ExperienceResult(**r) for r in resp.json()["results"]]

    def get(self, experience_id: uuid.UUID) -> dict[str, Any]:
        resp = self._http.get(f"/experiences/{experience_id}")
        resp.raise_for_status()
        return resp.json()

    def context(
        self,
        query: str,
        agent_id: Optional[str] = None,
        agent_role: Optional[AgentRole] = None,
        technologies: Optional[list[str]] = None,
        policy: Any = None,
        scope_run_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Compact, governed memory context for a task (POST /context).
        Like search(), degrades to "no memory" rather than failing the task.
        """
        body: dict[str, Any] = {
            "query": query,
            "agent_id": agent_id or self.agent_id,
            "role": (agent_role or self.agent_role).value,
            "technologies": technologies or [],
            "scope_run_id": scope_run_id,
        }
        if policy is not None:
            body.update(
                candidate_k=policy.candidate_k,
                max_context_memories=policy.max_context_memories,
                similarity_threshold=policy.similarity_threshold,
                redundancy_threshold=policy.redundancy_threshold,
                context_token_budget=policy.context_token_budget,
            )
        try:
            resp = self._http.post("/context", json=body)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as e:
            log.warning("context failed (%s) — continuing without memory", e)
            return {"context": "", "injected_experience_ids": [], "injected_task_keys": [],
                    "context_tokens": 0, "candidate_count": 0, "final_count": 0}

    def report_reuse(
        self, experience_id: uuid.UUID, success: bool, agent_id: Optional[str] = None
    ) -> dict[str, Any]:
        """Report whether a reused experience worked, as `agent_id` (the reporter)."""
        try:
            resp = self._http.post(
                f"/experiences/{experience_id}/reuse",
                json={"success": success, "agent_id": agent_id or self.agent_id},
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as e:
            log.warning("report_reuse failed (%s)", e)
            return {}

    # -- writes ------------------------------------------------------------

    def store(
        self,
        *,
        task: str,
        action: str,
        result: str,
        success: bool,
        run_id: str,
        system: SystemCondition,
        problem: Optional[str] = None,
        solution: Optional[str] = None,
        tools_used: Optional[list[str]] = None,
        technologies: Optional[list[str]] = None,
        tokens_input: int = 0,
        tokens_output: int = 0,
        tool_calls: int = 0,
        execution_time_ms: int = 0,
        cost_usd: float = 0.0,
        task_key: Optional[str] = None,
        retrieved_experience_ids: Optional[list[uuid.UUID]] = None,
        agent_id: Optional[str] = None,
        agent_role: Optional[AgentRole] = None,
    ) -> uuid.UUID:
        """
        Submit a completed experience. Returns immediately (202 from the API).

        Unlike `search`, a failure here *does* raise. Losing an experience means
        losing a data point from the experiment, and silently dropping those
        would leave you with results you cannot trust.
        """
        payload = {
            "agent_id": agent_id or self.agent_id,
            "agent_role": (agent_role or self.agent_role).value,
            "task": task,
            "problem": problem,
            "action": action,
            "solution": solution,
            "result": result,
            "tools_used": tools_used or [],
            "technologies": technologies or [],
            "success": success,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "tool_calls": tool_calls,
            "execution_time_ms": execution_time_ms,
            "cost_usd": cost_usd,
            "run_id": run_id,
            "system": system.value,
            "task_key": task_key,
            "retrieved_experience_ids": [str(i) for i in (retrieved_experience_ids or [])],
        }

        try:
            resp = self._http.post("/experiences", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise ActionCloudError(f"failed to store experience: {e}") from e

        return uuid.UUID(resp.json()["id"])


class InProcessClient:
    """
    Same interface as ActionCloudClient, calling MemoryService directly.

    Used by the experiment scripts so a run needs only Postgres (no API
    process, worker or LocalStack). Writes are processed synchronously through
    the same ingest pipeline the worker uses.
    """

    def __init__(self, service: Any = None, agent_id: str = "inprocess",
                 agent_role: AgentRole = AgentRole.CODING) -> None:
        from .service import MemoryService  # noqa: PLC0415

        self.service = service or MemoryService(sync_write=True)
        self.agent_id = agent_id
        self.agent_role = agent_role

    def close(self) -> None:
        pass

    def __enter__(self) -> "InProcessClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def context(self, query: str, agent_id: Optional[str] = None,
                agent_role: Optional[AgentRole] = None,
                technologies: Optional[list[str]] = None, policy: Any = None,
                scope_run_id: Optional[str] = None) -> dict[str, Any]:
        return self.service.prepare_context(
            query=query, agent_id=agent_id or self.agent_id,
            role=agent_role or self.agent_role, policy=policy,
            technologies=technologies or [], scope_run_id=scope_run_id,
        )

    def search(self, query: str, limit: int = 5, min_tier: MemoryTier = MemoryTier.PRIVATE,
               agent_id: Optional[str] = None, **kw: Any) -> list[ExperienceResult]:
        return self.service.search_memory(query=query, limit=limit, min_tier=min_tier,
                                          agent_id=agent_id or self.agent_id, **kw)

    def report_reuse(self, experience_id: uuid.UUID, success: bool,
                     agent_id: Optional[str] = None) -> dict[str, Any]:
        return self.service.record_reuse(experience_id, success=success,
                                         agent_id=agent_id or self.agent_id)

    def store(self, *, agent_id: Optional[str] = None,
              agent_role: Optional[AgentRole] = None, **fields: Any) -> uuid.UUID:
        from .schema import ExperienceCreate  # noqa: PLC0415

        payload = ExperienceCreate(
            agent_id=agent_id or self.agent_id,
            agent_role=agent_role or self.agent_role,
            **{k: v for k, v in fields.items() if v is not None},
        )
        return self.service.store_experience(payload)["id"]
