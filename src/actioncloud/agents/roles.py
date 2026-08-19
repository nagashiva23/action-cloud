"""
Concrete agent roles.

Phase 1 implements two (coding, research) — enough to prove the loop works with
two distinct identities writing into shared memory. The remaining four arrive in
Phase 2, and each is a four-line subclass.

Agents are deliberately unsophisticated. From the proposal (Section 9.2): their
essential property is that they interact *honestly* with shared memory — search
before, store after, report real numbers. A cleverer agent would make the
experiment harder to interpret, not better, because you would no longer know
whether an improvement came from the memory layer or from the agent's own
reasoning.
"""

from __future__ import annotations

from ..schema import AgentRole
from .base import Agent


class CodingAgent(Agent):
    role = AgentRole.CODING
    system_prompt = (
        "You are a coding agent. Write correct, minimal code and briefly explain "
        "any non-obvious decision. If prior experience is provided, reuse it "
        "rather than re-deriving the solution."
    )


class ResearchAgent(Agent):
    role = AgentRole.RESEARCH
    system_prompt = (
        "You are a research agent. Gather and summarise information accurately "
        "and concisely, citing what you relied on. If prior experience is "
        "provided, build on it rather than repeating the search."
    )


# --- Phase 2 ---------------------------------------------------------------
# class TestingAgent(Agent):       role = AgentRole.TESTING
# class DeploymentAgent(Agent):    role = AgentRole.DEPLOYMENT
# class DocumentationAgent(Agent): role = AgentRole.DOCUMENTATION
# class DataAnalysisAgent(Agent):  role = AgentRole.DATA_ANALYSIS


ROLE_REGISTRY: dict[AgentRole, type[Agent]] = {
    AgentRole.CODING: CodingAgent,
    AgentRole.RESEARCH: ResearchAgent,
}


def build_agent(role: AgentRole, **kwargs) -> Agent:
    """
    Construct an agent for a role.

    Note `agent_id` encodes the arm (e.g. `coding-baseline`). Keeping the two
    arms' identities distinct means a baseline agent can never accidentally
    retrieve its own memories through an agent_id match — a subtle way to leak
    memory into the supposedly stateless condition and quietly invalidate the
    whole comparison.
    """
    cls = ROLE_REGISTRY.get(role)
    if cls is None:
        raise ValueError(f"no agent implemented for role {role.value!r} yet")
    agent = cls(**kwargs)
    agent.role = role
    return agent
