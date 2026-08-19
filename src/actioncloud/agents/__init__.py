"""Agent implementations for ActionCloud."""

from .base import Agent, TaskOutcome, TaskSpec
from .roles import ROLE_REGISTRY, CodingAgent, ResearchAgent, build_agent

__all__ = [
    "Agent",
    "TaskSpec",
    "TaskOutcome",
    "CodingAgent",
    "ResearchAgent",
    "ROLE_REGISTRY",
    "build_agent",
]
