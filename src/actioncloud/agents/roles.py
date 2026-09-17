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


class TestingAgent(Agent):
    __test__ = False
    role = AgentRole.TESTING
    system_prompt = (
        "You are a testing agent. Write comprehensive test cases and verify failure modes. "
        "If prior experience is provided, build on established test patterns."
    )


class DeploymentAgent(Agent):
    role = AgentRole.DEPLOYMENT
    system_prompt = (
        "You are a deployment agent. Automate containerization and infrastructure setups. "
        "If prior experience is provided, reuse proven configuration parameters."
    )


class DocumentationAgent(Agent):
    role = AgentRole.DOCUMENTATION
    system_prompt = (
        "You are a documentation agent. Write clear, structured documentation and runbooks. "
        "If prior experience is provided, synthesize existing knowledge."
    )


class DataAnalysisAgent(Agent):
    role = AgentRole.DATA_ANALYSIS
    system_prompt = (
        "You are a data analysis agent. Query metrics and evaluate experimental statistical results. "
        "If prior experience is provided, build on previous analysis queries."
    )


ROLE_REGISTRY: dict[AgentRole, type[Agent]] = {
    AgentRole.CODING: CodingAgent,
    AgentRole.RESEARCH: ResearchAgent,
    AgentRole.TESTING: TestingAgent,
    AgentRole.DEPLOYMENT: DeploymentAgent,
    AgentRole.DOCUMENTATION: DocumentationAgent,
    AgentRole.DATA_ANALYSIS: DataAnalysisAgent,
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
