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


class SecurityAgent(Agent):
    role = AgentRole.SECURITY
    system_prompt = (
        "You are a security agent. Audit code, validate sanitization, and enforce authentication protocols. "
        "If prior experience is provided, reuse verified security rules."
    )


class DevOpsAgent(Agent):
    role = AgentRole.DEVOPS
    system_prompt = (
        "You are a devops agent. Configure CI/CD pipelines, container registries, and build scripts. "
        "If prior experience is provided, reuse proven build scripts."
    )


class DatabaseAgent(Agent):
    role = AgentRole.DATABASE
    system_prompt = (
        "You are a database agent. Optimize SQL queries, indexes, and schema migrations. "
        "If prior experience is provided, reuse proven database schemas."
    )


class MLAgent(Agent):
    role = AgentRole.ML
    system_prompt = (
        "You are an ML agent. Fine-tune, evaluate, and embed machine learning models. "
        "If prior experience is provided, reuse proven model parameters."
    )


class CloudAgent(Agent):
    role = AgentRole.CLOUD
    system_prompt = (
        "You are a cloud infrastructure agent. Provision cloud resources across AWS, GCP, and Azure. "
        "If prior experience is provided, reuse proven Terraform/CloudFormation templates."
    )


class MonitoringAgent(Agent):
    role = AgentRole.MONITORING
    system_prompt = (
        "You are a monitoring agent. Configure Prometheus metrics, Grafana dashboards, and alerts. "
        "If prior experience is provided, reuse proven alert rules."
    )


ROLE_REGISTRY: dict[AgentRole, type[Agent]] = {
    AgentRole.CODING: CodingAgent,
    AgentRole.RESEARCH: ResearchAgent,
    AgentRole.TESTING: TestingAgent,
    AgentRole.DEPLOYMENT: DeploymentAgent,
    AgentRole.DOCUMENTATION: DocumentationAgent,
    AgentRole.DATA_ANALYSIS: DataAnalysisAgent,
    AgentRole.SECURITY: SecurityAgent,
    AgentRole.DEVOPS: DevOpsAgent,
    AgentRole.DATABASE: DatabaseAgent,
    AgentRole.ML: MLAgent,
    AgentRole.CLOUD: CloudAgent,
    AgentRole.MONITORING: MonitoringAgent,
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
