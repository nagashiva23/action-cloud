#!/usr/bin/env python3
"""
ActionCloud Phase 3 — Automated Benchmark Execution Harness.

Executes a benchmark workload dataset across two experimental arms:
  Arm A: Baseline (System A, use_memory=False, stateless)
  Arm B: ActionCloud (System B, use_memory=True, memory-enabled)

Usage:
  python scripts/run_benchmark.py [--provider mock|anthropic|gemini|groq] [--limit N]
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
from actioncloud.schema import AgentRole, SystemCondition

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("run_benchmark")

BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
DATASET_PATH = Path(__file__).resolve().parent.parent / "src" / "actioncloud" / "benchmark" / "tasks.json"


def load_tasks(limit: int | None = None) -> list[dict]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Task dataset missing at {DATASET_PATH}")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    if limit:
        return tasks[:limit]
    return tasks


def run_arm(
    arm_name: str,
    system_cond: SystemCondition,
    use_memory: bool,
    tasks: list[dict],
    run_id: str,
    llm_kind: str,
) -> None:
    log.info("==================================================")
    log.info("Starting Arm: %s (%s, use_memory=%s)", arm_name, system_cond.value, use_memory)
    log.info("==================================================")

    llm = get_llm(llm_kind)
    client = ActionCloudClient(
        agent_id=f"benchmark-client-{system_cond.value}",
        agent_role=AgentRole.CODING,
        base_url=BASE_URL,
    )

    completed = 0
    for idx, t in enumerate(tasks, 1):
        role = AgentRole(t["role"])
        agent_id = f"{role.value}-{system_cond.value}"

        agent = build_agent(
            role=role,
            agent_id=agent_id,
            client=client,
            run_id=run_id,
            use_memory=use_memory,
            llm=llm,
        )

        spec = TaskSpec(
            task=t["task"],
            task_key=t.get("task_key"),
            technologies=t.get("technologies", []),
            expected_tools=t.get("expected_tools", []),
        )

        outcome = agent.run_task(spec)
        completed += 1

        # In ActionCloud arm, report reuse outcome if prior experience was retrieved
        if use_memory and outcome.retrieved_ids and outcome.experience_id:
            for ret_id in outcome.retrieved_ids:
                client.report_reuse(ret_id, success=outcome.success)

        log.info(
            "[%d/%d] %s | task_key=%s | success=%s | tokens=%d | retrieved=%d",
            idx,
            len(tasks),
            agent_id,
            spec.task_key,
            outcome.success,
            outcome.total_tokens,
            outcome.retrieved_count,
        )

    log.info("Arm %s finished executing %d tasks.", arm_name, completed)
    client.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ActionCloud Phase 3 Benchmark")
    parser.add_argument(
        "--provider",
        default=os.environ.get("LLM_PROVIDER", "mock"),
        choices=["mock", "anthropic", "gemini", "groq"],
        help="LLM model provider",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks to run")
    args = parser.parse_args()

    run_id = f"benchmark-{uuid.uuid4().hex[:8]}"
    tasks = load_tasks(args.limit)

    print(f"\nActionCloud Phase 3 Benchmark Runner  (run_id={run_id}, provider={args.provider})\n")
    print(f"Loaded {len(tasks)} task(s) from dataset.")

    # 1. Run Arm A: Baseline (Stateless)
    run_arm(
        arm_name="System A (Baseline)",
        system_cond=SystemCondition.BASELINE,
        use_memory=False,
        tasks=tasks,
        run_id=run_id,
        llm_kind=args.provider,
    )

    # Allow worker to settle
    time.sleep(2)

    # 2. Run Arm B: ActionCloud (Memory-Enabled)
    run_arm(
        arm_name="System B (ActionCloud Memory)",
        system_cond=SystemCondition.ACTIONCLOUD,
        use_memory=True,
        tasks=tasks,
        run_id=run_id,
        llm_kind=args.provider,
    )

    print(f"\nBenchmark completed successfully! Run ID: {run_id}")
    print(f"To evaluate metrics, run:\n    python scripts/evaluate_results.py --run-id {run_id}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
