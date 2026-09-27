"""
Experiment harness.

Runs the benchmark workload through one or more *arms* (configurations) and
computes metrics with the shared definitions in actioncloud.metrics.

Design choices that keep arms comparable:
  * Every arm gets its own run_id AND retrieval is scoped to that run_id, so
    no arm can read another arm's memories (the old ablation let K=2 reuse
    K=1's memories through the shared table).
  * Every arm sees the same task order and, in the simulated environment, the
    same per-task randomness (instance_id seeds it).
  * Success is graded by the same TaskVerifier in every arm.
"""

from __future__ import annotations

import random
import statistics
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

from .. import db
from ..agents import TaskSpec, build_agent
from ..client import InProcessClient
from ..llm import LLM, get_llm
from ..metrics import compare, compute_metrics
from ..policy import MemorySelectionPolicy
from ..schema import AgentRole
from ..service import MemoryService
from . import load_tasks
from .verifier import TaskVerifier


@dataclass
class ArmConfig:
    name: str
    use_memory: bool = True
    governance: bool = True
    k_inject: int = 1
    similarity_threshold: float = 0.30
    candidate_k: int = 10
    token_budget: int = 1000
    feedback_rate: float = 0.7   # share of task outcomes that are ever observed

    def policy(self) -> MemorySelectionPolicy:
        return MemorySelectionPolicy(
            candidate_k=self.candidate_k,
            max_context_memories=self.k_inject,
            similarity_threshold=self.similarity_threshold,
            context_token_budget=self.token_budget,
        )


BASELINE = ArmConfig("baseline", use_memory=False)
FLAT = ArmConfig("flat_memory", governance=False)
ACTIONCLOUD = ArmConfig("actioncloud")


@dataclass
class Workload:
    epochs: int = 3
    agents_per_role: int = 3
    seed: int = 7
    limit: Optional[int] = None

    def instances(self) -> list[dict]:
        tasks = list(load_tasks())[: self.limit] if self.limit else list(load_tasks())
        out = []
        for epoch in range(self.epochs):
            order = list(tasks)
            random.Random(f"{self.seed}:{epoch}").shuffle(order)
            for t in order:
                out.append({**t, "instance_id": f"{t['id']}:{epoch}"})
        return out


def _tier_distribution(run_id: str) -> dict[str, int]:
    with db.get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT tier::text AS tier, count(*) AS n FROM experiences WHERE run_id = %s GROUP BY tier",
            (run_id,),
        )
        dist = {r["tier"]: r["n"] for r in cur.fetchall()}
        cur.execute(
            """
            SELECT count(*) AS n FROM tier_transitions t
            JOIN experiences e ON e.id = t.experience_id
            WHERE e.run_id = %s AND t.from_tier IS NOT NULL AND t.to_tier = 'private'
            """,
            (run_id,),
        )
        dist["quarantined"] = cur.fetchone()["n"]
    return dist


def run_arm(
    arm: ArmConfig,
    workload: Workload,
    llm_factory: Callable[[int], LLM],
    experiment_id: str,
    progress: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    run_id = f"{experiment_id}-{arm.name}-s{workload.seed}"
    service = MemoryService(sync_write=True, governance=arm.governance)
    client = InProcessClient(service=service)
    llm = llm_factory(workload.seed)
    verifier = TaskVerifier()
    policy = arm.policy()

    outcomes: list[dict[str, Any]] = []
    memory_correct: dict[str, bool] = {}   # experience id -> strictly correct?
    poisoned_injections = relevant_injections = 0
    false_positive_stores = 0

    instances = workload.instances()
    for i, t in enumerate(instances):
        role = AgentRole(t["role"])
        agent = build_agent(
            role=role,
            agent_id=f"{role.value}-{arm.name}-{i % workload.agents_per_role}",
            client=client,
            run_id=run_id,
            use_memory=arm.use_memory,
            llm=llm,
            policy=policy,
            verifier=verifier,
            report_reuse=arm.governance,
            feedback_rate=arm.feedback_rate,
            scope_run_id=run_id,
        )
        o = agent.run_task(TaskSpec(
            task=t["task"], task_key=t["task_key"], technologies=t["technologies"],
            expected_tools=t.get("expected_tools", []), instance_id=t["instance_id"],
        ))
        for exp_id, key in zip(o.retrieved_ids, o.injected_task_keys):
            if key == t["task_key"]:
                relevant_injections += 1
                poisoned_injections += not memory_correct.get(str(exp_id), True)
        if o.experience_id is not None:
            memory_correct[str(o.experience_id)] = o.success
        false_positive_stores += o.self_check_success and not o.success

        outcomes.append({
            "task_key": o.task_key,
            "success": o.success,
            "tokens_input": o.tokens_input,
            "tokens_output": o.tokens_output,
            "execution_time_ms": o.execution_time_ms,
            "cost_usd": o.cost_usd,
            "injected_task_keys": o.injected_task_keys,
            "context_tokens": o.context_tokens,
            "mode": o.mode,
        })
        if progress and (i + 1) % 100 == 0:
            progress(f"{arm.name} seed={workload.seed}: {i + 1}/{len(instances)}")

    m = compute_metrics(outcomes)
    m.update({
        "poisoned_injection_pct": round(100 * poisoned_injections / relevant_injections, 2)
        if relevant_injections else 0.0,
        "false_positive_memories": false_positive_stores,
        "avg_context_tokens": round(sum(o["context_tokens"] for o in outcomes) / len(outcomes), 1),
        "tier_distribution": _tier_distribution(run_id),
    })
    return {"arm": asdict(arm), "run_id": run_id, "seed": workload.seed, "metrics": m}


NUMERIC_KEYS = (
    "success_rate_pct", "injection_rate_pct", "retrieval_precision_pct",
    "effective_reuse_rate_pct", "misled_rate_pct", "redundancy_index",
    "poisoned_injection_pct", "avg_tokens_per_task", "avg_context_tokens",
    "avg_latency_ms", "total_tokens", "total_cost_usd", "cost_per_success_usd",
    "false_positive_memories",
)


def summarise(runs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Mean and sample std-dev of each metric across seeds."""
    out: dict[str, dict[str, float]] = {}
    for key in NUMERIC_KEYS:
        vals = [r["metrics"][key] for r in runs if r["metrics"].get(key) is not None]
        if vals:
            out[key] = {
                "mean": round(statistics.mean(vals), 4),
                "sd": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
            }
    return out


def run_experiment(
    arms: list[ArmConfig],
    seeds: list[int],
    workload: Workload,
    provider: str = "sim",
    progress: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    experiment_id = f"exp-{uuid.uuid4().hex[:8]}"

    def llm_factory(seed: int) -> LLM:
        if provider == "sim":
            from .simulator import SimulatedAgentLLM  # noqa: PLC0415
            return SimulatedAgentLLM(seed=seed)
        return get_llm(provider)

    per_arm: dict[str, list[dict[str, Any]]] = {a.name: [] for a in arms}
    for seed in seeds:
        wl = Workload(workload.epochs, workload.agents_per_role, seed, workload.limit)
        for arm in arms:
            per_arm[arm.name].append(run_arm(arm, wl, llm_factory, experiment_id, progress))

    summary = {name: summarise(runs) for name, runs in per_arm.items()}
    comparisons = {}
    if "baseline" in per_arm:
        for name, runs in per_arm.items():
            if name == "baseline":
                continue
            deltas = [compare(b["metrics"], r["metrics"]) for b, r in zip(per_arm["baseline"], runs)]
            comparisons[name] = {
                k: {
                    "mean": round(statistics.mean(d[k] for d in deltas), 2),
                    "sd": round(statistics.stdev(d[k] for d in deltas), 2) if len(deltas) > 1 else 0.0,
                }
                for k in deltas[0]
            }
    return {
        "experiment_id": experiment_id,
        "provider": provider,
        "seeds": seeds,
        "workload": {"epochs": workload.epochs, "agents_per_role": workload.agents_per_role,
                     "tasks_per_epoch": len(load_tasks()) if not workload.limit else workload.limit},
        "arms": [asdict(a) for a in arms],
        "summary": summary,
        "vs_baseline": comparisons,
        "runs": per_arm,
    }
