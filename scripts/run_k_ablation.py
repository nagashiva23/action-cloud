#!/usr/bin/env python3
"""
ActionCloud — Top-K Memory Context Ablation Harness.

Executes the benchmark workload dataset across multiple memory retrieval limits:
  K = 0 (Baseline, stateless)
  K = 1, 2, 3, 5, 10 (ActionCloud Memory Enabled with limit K)

Usage:
  python scripts/run_k_ablation.py [--provider mock|anthropic|gemini|groq] [--limit N] [--k-values 0 1 2 3 5 10]
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
log = logging.getLogger("run_k_ablation")

BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
DATASET_PATH = Path(__file__).resolve().parent.parent / "src" / "actioncloud" / "benchmark" / "tasks.json"
RESULTS_OUTPUT_PATH = Path(__file__).resolve().parent.parent / "src" / "actioncloud" / "benchmark" / "k_ablation_results.json"


def load_tasks(limit: int | None = None) -> list[dict]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Task dataset missing at {DATASET_PATH}")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    if limit:
        return tasks[:limit]
    return tasks


def run_k_arm(
    k: int,
    tasks: list[dict],
    run_id: str,
    llm_kind: str,
) -> dict:
    arm_name = "Baseline (K=0)" if k == 0 else f"ActionCloud (K={k})"
    use_memory = (k > 0)
    system_cond = SystemCondition.BASELINE if k == 0 else SystemCondition.ACTIONCLOUD

    log.info("==================================================")
    log.info("Starting Arm: %s (K=%d, use_memory=%s)", arm_name, k, use_memory)
    log.info("==================================================")

    llm = get_llm(llm_kind)
    client = ActionCloudClient(
        agent_id=f"ablation-client-k{k}",
        agent_role=AgentRole.CODING,
        base_url=BASE_URL,
    )

    arm_outcomes = []
    total_tokens = 0
    total_cost = 0.0
    total_latency_ms = 0
    successful = 0
    total_retrieved = 0

    for idx, t in enumerate(tasks, 1):
        role = AgentRole(t["role"])
        agent_id = f"{role.value}-k{k}"

        agent = build_agent(
            role=role,
            agent_id=agent_id,
            client=client,
            run_id=f"{run_id}-k{k}",
            use_memory=use_memory,
            memory_limit=k,
            llm=llm,
        )

        spec = TaskSpec(
            task=t["task"],
            task_key=t.get("task_key"),
            technologies=t.get("technologies", []),
            expected_tools=t.get("expected_tools", []),
        )

        outcome = agent.run_task(spec)
        if outcome.success:
            successful += 1

        total_tokens += outcome.total_tokens
        total_cost += outcome.cost_usd
        total_latency_ms += outcome.execution_time_ms
        total_retrieved += outcome.retrieved_count

        # Report reuse feedback if memories were retrieved
        if use_memory and outcome.retrieved_ids and outcome.experience_id:
            for ret_id in outcome.retrieved_ids:
                client.report_reuse(ret_id, success=outcome.success)

        arm_outcomes.append({
            "task_id": t.get("id"),
            "task_key": t.get("task_key"),
            "role": role.value,
            "success": outcome.success,
            "total_tokens": outcome.total_tokens,
            "tokens_input": outcome.tokens_input,
            "tokens_output": outcome.tokens_output,
            "execution_time_ms": outcome.execution_time_ms,
            "cost_usd": outcome.cost_usd,
            "retrieved_count": outcome.retrieved_count,
        })

        log.info(
            "[K=%d] [%d/%d] %s | task_key=%s | success=%s | tokens=%d | retrieved=%d",
            k,
            idx,
            len(tasks),
            agent_id,
            spec.task_key,
            outcome.success,
            outcome.total_tokens,
            outcome.retrieved_count,
        )

    client.close()

    summary = {
        "k": k,
        "arm_name": arm_name,
        "total_tasks": len(tasks),
        "successful_tasks": successful,
        "success_rate_pct": round((successful / len(tasks)) * 100.0, 2) if tasks else 0.0,
        "total_tokens": total_tokens,
        "avg_tokens_per_task": round(total_tokens / len(tasks), 2) if tasks else 0.0,
        "avg_latency_ms": round(total_latency_ms / len(tasks), 2) if tasks else 0.0,
        "total_cost_usd": round(total_cost, 6),
        "avg_retrieved_per_task": round(total_retrieved / len(tasks), 2) if tasks else 0.0,
        "outcomes": arm_outcomes,
    }

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="ActionCloud Top-K Memory Context Ablation Harness")
    parser.add_argument(
        "--provider",
        default=os.environ.get("LLM_PROVIDER", "mock"),
        choices=["mock", "anthropic", "gemini", "groq"],
        help="LLM provider backend",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks to run per K configuration")
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 5, 10],
        help="Space-separated list of K values to test (default: 0 1 2 3 5 10)",
    )
    args = parser.parse_args()

    run_id = f"ablation-{uuid.uuid4().hex[:8]}"
    tasks = load_tasks(args.limit)

    print(f"\n=========================================================================")
    print(f" ActionCloud Top-K Memory Context Ablation Study  (run_id={run_id})")
    print(f"=========================================================================\n")
    print(f"Provider:  {args.provider}")
    print(f"K Values:  {args.k_values}")
    print(f"Task Count: {len(tasks)} task(s) per K configuration\n")

    k_results = {}
    for k in args.k_values:
        summary = run_k_arm(k=k, tasks=tasks, run_id=run_id, llm_kind=args.provider)
        k_results[f"K={k}"] = summary
        time.sleep(1)  # allow background worker to settle

    # Save ablation study output
    output_payload = {
        "run_id": run_id,
        "provider": args.provider,
        "task_count": len(tasks),
        "k_values": args.k_values,
        "results": k_results,
    }

    RESULTS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\n{'-' * 60}")
    print(f"Ablation study execution complete!")
    print(f"Results saved to: {RESULTS_OUTPUT_PATH}")
    print(f"To render empirical comparison & marginal utility report, run:\n    python scripts/evaluate_k_ablation.py\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
