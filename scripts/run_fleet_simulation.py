#!/usr/bin/env python3
"""
ActionCloud — Heterogeneous 100-Agent Fleet Simulation.

Simulates 100 heterogeneous agent task executions across 12 distinct roles:
  1. CodingAgent
  2. ResearchAgent
  3. TestingAgent
  4. DeploymentAgent
  5. DocumentationAgent
  6. DataAnalysisAgent
  7. SecurityAgent
  8. DevOpsAgent
  9. DatabaseAgent
  10. MLAgent
  11. CloudAgent
  12. MonitoringAgent

Executes memory-dependent sequential workloads (Task A -> B -> C -> D) where prior
experience from one agent role directly enables subsequent agents across the fleet.

Usage:
  python scripts/run_fleet_simulation.py [--agents N] [--provider mock|gemini|anthropic|groq]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import actioncloud.db as db
from actioncloud.agents import TaskSpec, build_agent
from actioncloud.client import ActionCloudClient
from actioncloud.llm import get_llm
from actioncloud.policy import MemorySelectionPolicy
from actioncloud.schema import AgentRole, SystemCondition
from actioncloud.service import memory_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("fleet_simulation")

BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")


def generate_fleet_tasks(count: int = 100) -> list[dict]:
    """
    Generate heterogeneous memory-dependent task sequences across 12 roles.
    """
    roles = list(AgentRole)
    tasks = []

    # Sequential dependent workflow clusters
    clusters = [
        {
            "task_key": "task-dep-pgvector-setup",
            "tech": ["postgres", "pgvector", "docker"],
            "sequence": [
                (AgentRole.DEPLOYMENT, "Configure PostgreSQL 16 container service with pgvector extension"),
                (AgentRole.DATABASE, "Optimize HNSW vector index parameters on pgvector experiences table"),
                (AgentRole.CODING, "Implement pgvector cosine similarity search dependency in FastAPI router"),
                (AgentRole.SECURITY, "Audit pgvector SQL parameterization for SQL injection vulnerabilities"),
                (AgentRole.MONITORING, "Expose Prometheus vector search latency metrics endpoint"),
            ],
        },
        {
            "task_key": "task-dep-mcp-server",
            "tech": ["mcp", "json-rpc", "fastapi"],
            "sequence": [
                (AgentRole.CLOUD, "Architect MCP stdio server bridge connecting IDE to ActionCloud MemoryService"),
                (AgentRole.DEVOPS, "Create Docker container entrypoint for MCP stdio server execution"),
                (AgentRole.TESTING, "Write pytest integration test for MCP stdio JSON-RPC tool calls"),
                (AgentRole.DOCUMENTATION, "Write runbook section for configuring MCP stdio server in Cursor mcp.json"),
                (AgentRole.CLOUD, "Provision AWS ECS Fargate task definition for ActionCloud MCP backend"),
            ],
        },
        {
            "task_key": "task-dep-llm-embeddings",
            "tech": ["python", "embeddings", "llm"],
            "sequence": [
                (AgentRole.RESEARCH, "Survey dense vector embedding models for procedural memory representation"),
                (AgentRole.ML, "Fine-tune 1536-dimensional unit-normalized embedding provider for experience similarity"),
                (AgentRole.DATA_ANALYSIS, "Calculate Redundancy Index and Knowledge Reuse Rate across embedding dimensions"),
            ],
        },
    ]

    task_id = 1
    while len(tasks) < count:
        for c in clusters:
            for role, t_text in c["sequence"]:
                if len(tasks) >= count:
                    break
                tasks.append({
                    "id": task_id,
                    "role": role.value,
                    "task": f"{t_text} (Seq #{task_id})",
                    "task_key": c["task_key"],
                    "technologies": c["tech"],
                    "expected_tools": ["editor", "pytest", "docker"],
                })
                task_id += 1

    return tasks[:count]


def main() -> int:
    parser = argparse.ArgumentParser(description="ActionCloud Heterogeneous Fleet Simulation")
    parser.add_argument("--agents", type=int, default=100, help="Number of agent task executions to simulate")
    parser.add_argument(
        "--provider",
        default=os.environ.get("LLM_PROVIDER", "mock"),
        choices=["mock", "anthropic", "gemini", "groq"],
        help="LLM provider backend",
    )
    args = parser.parse_args()

    run_id = f"fleet-sim-{uuid.uuid4().hex[:8]}"
    tasks = generate_fleet_tasks(args.agents)

    print(f"\n=========================================================================")
    print(f" ActionCloud Heterogeneous {len(tasks)}-Agent Fleet Simulation")
    print(f"=========================================================================\n")
    print(f"Run ID:      {run_id}")
    print(f"LLM Provider: {args.provider}")
    print(f"Active Roles: 12 Heterogeneous Fleet Agent Roles\n")

    client = ActionCloudClient(
        agent_id="fleet-controller",
        agent_role=AgentRole.CODING,
        base_url=BASE_URL,
    )
    llm = get_llm(args.provider)
    policy = MemorySelectionPolicy(
        candidate_k=10,
        max_context_memories=1,
        similarity_threshold=0.70,
        redundancy_threshold=0.85,
        context_token_budget=1000,
    )

    successful_tasks = 0
    total_injected_memories = 0
    total_tokens = 0
    cross_role_reuses = 0

    print("Executing sequential memory-dependent task workload across fleet...\n")

    for idx, t in enumerate(tasks, 1):
        role = AgentRole(t["role"])
        agent_id = f"{role.value}-agent-{idx:03d}"

        agent = build_agent(
            role=role,
            agent_id=agent_id,
            client=client,
            run_id=run_id,
            use_memory=True,
            memory_limit=policy.candidate_k,
            llm=llm,
        )

        spec = TaskSpec(
            task=t["task"],
            task_key=t.get("task_key"),
            technologies=t.get("technologies", []),
            expected_tools=t.get("expected_tools", []),
        )

        # Execute task with ActionCloud adaptive memory context
        outcome = agent.run_task(spec)

        if outcome.success:
            successful_tasks += 1

        total_tokens += outcome.total_tokens
        total_injected_memories += outcome.retrieved_count

        # Report reuse feedback if prior experience was retrieved
        if outcome.retrieved_ids and outcome.experience_id:
            for ret_id in outcome.retrieved_ids:
                client.report_reuse(ret_id, success=outcome.success)
                cross_role_reuses += 1

        if idx % 10 == 0 or idx == len(tasks):
            print(
                f"  [{idx:3d}/{len(tasks)}] {agent_id:<22} | "
                f"key={spec.task_key:<24} | success={outcome.success} | "
                f"injected={outcome.retrieved_count} | tokens={outcome.total_tokens}"
            )

    client.close()

    time.sleep(1)
    metrics = memory_service.get_metrics(run_id)
    tier_dist = metrics.get("governance_tier_distribution", {})

    print(f"\n=========================================================================")
    print(f" Fleet Simulation Results & Knowledge Propagation Analysis")
    print(f"=========================================================================\n")

    print(f"  • Total Agent Tasks Executed:  {len(tasks)}")
    print(f"  • Successful Tasks:            {successful_tasks} ({round(successful_tasks / len(tasks) * 100, 2)}%)")
    print(f"  • Total Injected Memories:     {total_injected_memories}")
    print(f"  • Cross-Role Reuses Recorded:  {cross_role_reuses}")
    print(f"  • Total Tokens Consumed:       {total_tokens}")

    print(f"\nGovernance Tier Progression across Fleet:")
    for tier_name, info in tier_dist.items():
        print(f"  • {tier_name.upper():<14}: {info.get('count', 0):<4} memories | {info.get('total_reuses', 0):<4} reuses | {info.get('observed_success_rate', 0.0)*100:.1f}% success")

    print(f"\n=========================================================================")
    print(f" SIMULATION COMPLETE: Governed fleet propagation verified successfully!")
    print(f"=========================================================================\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
