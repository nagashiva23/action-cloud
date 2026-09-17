"""
ActionCloud — Metric Calculation Engine.

Computes the core evaluation metrics comparing System A (Baseline, stateless)
versus System B (ActionCloud, memory-enabled):

1. Knowledge Reuse Rate (KRR %)
2. Redundancy Index (RI)
3. Cumulative Token Savings (% reduction)
4. Task Execution Latency (mean execution_time_ms)
5. Total Cost ($ USD)
6. Observed Reuse Success Rate per memory tier
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from . import db

log = logging.getLogger(__name__)


class MetricCalculator:
    """
    Computes experiment metrics across runs or system conditions.
    """

    @staticmethod
    def calculate_run_metrics(run_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Aggregate all metrics from Postgres experiences table.
        """
        with db.get_conn() as conn, conn.cursor() as cur:
            # Aggregate overall per-system metrics
            query = """
            SELECT system,
                   count(*) AS total_tasks,
                   count(DISTINCT task_key) FILTER (WHERE task_key IS NOT NULL) AS unique_task_keys,
                   sum(tokens_input) AS total_tokens_in,
                   sum(tokens_output) AS total_tokens_out,
                   sum(tokens_input + tokens_output) AS total_tokens,
                   avg(execution_time_ms) AS avg_latency_ms,
                   sum(cost_usd) AS total_cost_usd,
                   count(*) FILTER (WHERE array_length(retrieved_experience_ids, 1) > 0) AS tasks_with_retrieval,
                   count(*) FILTER (WHERE success = TRUE) AS successful_tasks
            FROM experiences
            WHERE (%s::text IS NULL OR run_id = %s)
            GROUP BY system;
            """
            cur.execute(query, (run_id, run_id))
            system_rows = cur.fetchall()

            # Query governance tier distribution
            cur.execute(
                """
                SELECT tier,
                       count(*) AS count,
                       sum(reuse_count) AS total_reuses,
                       sum(reuse_success_count) AS total_successful_reuses
                FROM experiences
                WHERE (%s::text IS NULL OR run_id = %s)
                GROUP BY tier;
                """,
                (run_id, run_id),
            )
            tier_rows = cur.fetchall()

        systems_data = {}
        for r in system_rows:
            sys_name = r["system"]
            total = r["total_tasks"]
            tasks_retrieved = r["tasks_with_retrieval"]
            successes = r["successful_tasks"]
            unique_keys = r["unique_task_keys"] or 0

            # Knowledge Reuse Rate (KRR)
            krr = (tasks_retrieved / total * 100.0) if total > 0 else 0.0

            # Success Rate
            success_rate = (successes / total * 100.0) if total > 0 else 0.0

            # Redundancy Index (RI): (Total Tasks - Unique Tasks) / Total Tasks
            redundancy_index = ((total - unique_keys) / total) if total > 0 else 0.0

            systems_data[sys_name] = {
                "total_tasks": total,
                "successful_tasks": successes,
                "task_success_rate_pct": round(success_rate, 2),
                "knowledge_reuse_rate_pct": round(krr, 2),
                "redundancy_index": round(redundancy_index, 4),
                "total_tokens_input": r["total_tokens_in"] or 0,
                "total_tokens_output": r["total_tokens_out"] or 0,
                "total_tokens": r["total_tokens"] or 0,
                "avg_latency_ms": round(float(r["avg_latency_ms"] or 0.0), 2),
                "total_cost_usd": round(float(r["total_cost_usd"] or 0.0), 6),
            }

        # Calculate comparative savings if both arms exist
        baseline = systems_data.get("baseline", {})
        actioncloud = systems_data.get("actioncloud", {})

        token_savings_pct = 0.0
        cost_savings_pct = 0.0
        latency_reduction_pct = 0.0

        if baseline.get("total_tokens", 0) > 0 and actioncloud.get("total_tokens", 0) >= 0:
            b_tokens = baseline["total_tokens"]
            a_tokens = actioncloud["total_tokens"]
            token_savings_pct = round(((b_tokens - a_tokens) / b_tokens) * 100.0, 2)

        if baseline.get("total_cost_usd", 0) > 0 and actioncloud.get("total_cost_usd", 0) >= 0:
            b_cost = baseline["total_cost_usd"]
            a_cost = actioncloud["total_cost_usd"]
            cost_savings_pct = round(((b_cost - a_cost) / b_cost) * 100.0, 2)

        if baseline.get("avg_latency_ms", 0) > 0 and actioncloud.get("avg_latency_ms", 0) >= 0:
            b_lat = baseline["avg_latency_ms"]
            a_lat = actioncloud["avg_latency_ms"]
            latency_reduction_pct = round(((b_lat - a_lat) / b_lat) * 100.0, 2)

        tier_distribution = {
            r["tier"]: {
                "count": r["count"],
                "total_reuses": r["total_reuses"] or 0,
                "total_successful_reuses": r["total_successful_reuses"] or 0,
                "observed_success_rate": round(
                    (r["total_successful_reuses"] / r["total_reuses"]) if (r["total_reuses"] or 0) > 0 else 0.0,
                    4,
                ),
            }
            for r in tier_rows
        }

        return {
            "run_id": run_id,
            "systems": systems_data,
            "comparative_metrics": {
                "token_savings_pct": token_savings_pct,
                "cost_savings_pct": cost_savings_pct,
                "latency_reduction_pct": latency_reduction_pct,
            },
            "governance_tier_distribution": tier_distribution,
        }
