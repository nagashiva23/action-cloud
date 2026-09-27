"""
ActionCloud — metric definitions.

One pure function, `compute_metrics`, defines every metric over an ordered
list of task outcomes. The experiment harness calls it on verified outcomes;
`MetricCalculator` (GET /metrics, MCP get_memory_metrics) calls it on rows
from Postgres. Same definitions everywhere.

Definitions (per arm / system):

  success_rate          tasks succeeded / tasks.
  injection_rate        tasks that received >= 1 injected memory / tasks.
                        (Previously reported as "Knowledge Reuse Rate", which
                        overstated it: an injected memory may be irrelevant.)
  retrieval_precision   injected memories from the SAME task family /
                        injected memories.
  effective_reuse_rate  tasks that received a same-family memory AND
                        succeeded / tasks.
  misled_rate           tasks that received a same-family memory and still
                        failed / tasks that received one.
  redundancy_index      of the tasks whose family had ALREADY been solved
                        successfully earlier in the run, the fraction solved
                        again without a same-family memory. 1.0 = every
                        repeat re-derived from scratch (the stateless
                        baseline, by construction); 0.0 = every repeat reused.
                        (The old formula (tasks - unique keys) / tasks
                        described the workload, not the system, and was
                        identical for both arms.)
  tokens / cost / latency  totals and per-task means.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

from . import db

log = logging.getLogger(__name__)


def _pct(num: float, den: float) -> float:
    return round(100.0 * num / den, 2) if den else 0.0


def compute_metrics(outcomes: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """
    outcomes: ordered (by execution time) dicts with keys
      task_key, success, tokens_input, tokens_output, execution_time_ms,
      cost_usd, injected_task_keys (list).
    """
    rows = list(outcomes)
    n = len(rows)
    solved_keys: set[str] = set()
    repeats = redundant = 0
    tasks_injected = injected_total = injected_relevant = 0
    tasks_relevant = relevant_success = 0
    successes = 0
    tok_in = tok_out = 0
    latency = 0
    cost = 0.0

    for r in rows:
        key = r.get("task_key")
        inj = [k for k in (r.get("injected_task_keys") or [])]
        relevant = key is not None and key in inj
        ok = bool(r.get("success"))

        if inj:
            tasks_injected += 1
        injected_total += len(inj)
        injected_relevant += sum(1 for k in inj if k is not None and k == key)
        if relevant:
            tasks_relevant += 1
            relevant_success += ok
        if key is not None and key in solved_keys:
            repeats += 1
            redundant += not relevant
        if ok and key is not None:
            solved_keys.add(key)

        successes += ok
        tok_in += int(r.get("tokens_input") or 0)
        tok_out += int(r.get("tokens_output") or 0)
        latency += int(r.get("execution_time_ms") or 0)
        cost += float(r.get("cost_usd") or 0.0)

    return {
        "total_tasks": n,
        "successful_tasks": successes,
        "success_rate_pct": _pct(successes, n),
        "injection_rate_pct": _pct(tasks_injected, n),
        "retrieval_precision_pct": _pct(injected_relevant, injected_total),
        "effective_reuse_rate_pct": _pct(relevant_success, n),
        "misled_rate_pct": _pct(tasks_relevant - relevant_success, tasks_relevant),
        "redundancy_index": round(redundant / repeats, 4) if repeats else 0.0,
        "repeat_tasks": repeats,
        "total_tokens_input": tok_in,
        "total_tokens_output": tok_out,
        "total_tokens": tok_in + tok_out,
        "avg_tokens_per_task": round((tok_in + tok_out) / n, 1) if n else 0.0,
        "avg_latency_ms": round(latency / n, 1) if n else 0.0,
        "total_cost_usd": round(cost, 6),
        "cost_per_success_usd": round(cost / successes, 6) if successes else None,
    }


def compare(baseline: Dict[str, Any], treatment: Dict[str, Any]) -> Dict[str, float]:
    """Relative change of treatment vs baseline (positive = saving)."""
    def saving(key: str) -> float:
        b, t = baseline.get(key) or 0, treatment.get(key) or 0
        return round(100.0 * (b - t) / b, 2) if b else 0.0

    return {
        "token_savings_pct": saving("total_tokens"),
        "cost_savings_pct": saving("total_cost_usd"),
        "latency_reduction_pct": saving("avg_latency_ms"),
        "success_rate_delta_pp": round(
            (treatment.get("success_rate_pct") or 0) - (baseline.get("success_rate_pct") or 0), 2
        ),
    }


_ROWS_SQL = """
SELECT e.system::text AS system, e.task_key, e.success,
       e.tokens_input, e.tokens_output, e.execution_time_ms, e.cost_usd,
       COALESCE((
           SELECT array_agg(r.task_key ORDER BY r.task_key)
           FROM experiences r
           WHERE r.id = ANY(e.retrieved_experience_ids)
       ), '{}') AS injected_task_keys
FROM experiences e
WHERE (%(run_id)s::text IS NULL OR e.run_id = %(run_id)s::text)
ORDER BY e.created_at, e.id
"""

_TIERS_SQL = """
SELECT tier::text AS tier, count(*) AS count,
       sum(reuse_count) AS total_reuses,
       sum(reuse_success_count) AS total_successful_reuses
FROM experiences
WHERE (%(run_id)s::text IS NULL OR run_id = %(run_id)s::text)
GROUP BY tier
"""


class MetricCalculator:
    """Metrics over stored experiences (success = the agent's stored verdict)."""

    @staticmethod
    def calculate_run_metrics(run_id: Optional[str] = None) -> Dict[str, Any]:
        run_id = run_id or None
        with db.get_conn() as conn, conn.cursor() as cur:
            cur.execute(_ROWS_SQL, {"run_id": run_id})
            rows = cur.fetchall()
            cur.execute(_TIERS_SQL, {"run_id": run_id})
            tier_rows = cur.fetchall()

        by_system: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            by_system.setdefault(r["system"], []).append(r)
        systems = {name: compute_metrics(rs) for name, rs in by_system.items()}

        comparative = {}
        if "baseline" in systems and "actioncloud" in systems:
            comparative = compare(systems["baseline"], systems["actioncloud"])

        tiers = {
            r["tier"]: {
                "count": r["count"],
                "total_reuses": r["total_reuses"] or 0,
                "total_successful_reuses": r["total_successful_reuses"] or 0,
                "observed_success_rate": round(
                    (r["total_successful_reuses"] or 0) / r["total_reuses"], 4
                ) if r["total_reuses"] else None,
            }
            for r in tier_rows
        }
        return {
            "run_id": run_id,
            "systems": systems,
            "comparative_metrics": comparative,
            "governance_tier_distribution": tiers,
        }
