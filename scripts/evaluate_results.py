#!/usr/bin/env python3
"""
ActionCloud Phase 3 — Empirical Results Generator.

Fetches evaluation metrics from ActionCloud GET /metrics endpoint
and outputs formatted empirical comparison tables.

Usage:
  python scripts/evaluate_results.py [--run-id RUN_ID]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx

BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate ActionCloud Empirical Metrics")
    parser.add_argument("--run-id", default=None, help="Filter by specific benchmark run_id")
    args = parser.parse_args()

    try:
        resp = httpx.get(f"{BASE_URL}/metrics", params={"run_id": args.run_id}, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"Error fetching metrics from API ({BASE_URL}/metrics): {e}")
        return 1

    systems = data.get("systems", {})
    comp = data.get("comparative_metrics", {})
    tiers = data.get("governance_tier_distribution", {})

    print("\n=========================================================================")
    print(f" ActionCloud Empirical Evaluation Report  (run_id={data.get('run_id') or 'ALL'})")
    print("=========================================================================\n")

    print("1. System Performance Comparison")
    print("-------------------------------------------------------------------------")
    print(f"{'Metric':<30} | {'Baseline (System A)':<20} | {'ActionCloud (System B)':<20}")
    print("-------------------------------------------------------------------------")

    base = systems.get("baseline", {})
    ac = systems.get("actioncloud", {})

    print(f"{'Total Tasks Executed':<30} | {base.get('total_tasks', 0):<20} | {ac.get('total_tasks', 0):<20}")
    print(f"{'Successful Tasks':<30} | {base.get('successful_tasks', 0):<20} | {ac.get('successful_tasks', 0):<20}")
    print(f"{'Task Success Rate (%)':<30} | {str(base.get('task_success_rate_pct', 0.0)) + '%' :<20} | {str(ac.get('task_success_rate_pct', 0.0)) + '%' :<20}")
    print(f"{'Knowledge Reuse Rate (KRR %)':<30} | {str(base.get('knowledge_reuse_rate_pct', 0.0)) + '%' :<20} | {str(ac.get('knowledge_reuse_rate_pct', 0.0)) + '%' :<20}")
    print(f"{'Redundancy Index (RI)':<30} | {base.get('redundancy_index', 0.0):<20} | {ac.get('redundancy_index', 0.0):<20}")
    print(f"{'Total Tokens Consumed':<30} | {base.get('total_tokens', 0):<20} | {ac.get('total_tokens', 0):<20}")
    print(f"{'Mean Execution Latency (ms)':<30} | {base.get('avg_latency_ms', 0.0):<20} | {ac.get('avg_latency_ms', 0.0):<20}")
    print(f"{'Total Cost ($ USD)':<30} | {str(base.get('total_cost_usd', 0.0)) :<20} | {str(ac.get('total_cost_usd', 0.0)) :<20}")
    print("-------------------------------------------------------------------------\n")

    print("2. Efficiency & Cost Savings Achieved by ActionCloud")
    print("-------------------------------------------------------------------------")
    print(f"  - Token Savings:          {comp.get('token_savings_pct', 0.0)}%")
    print(f"  - Financial Cost Savings: {comp.get('cost_savings_pct', 0.0)}%")
    print(f"  - Latency Reduction:     {comp.get('latency_reduction_pct', 0.0)}%")
    print("-------------------------------------------------------------------------\n")

    print("3. MemoryJudge Governance Tier Distribution")
    print("-------------------------------------------------------------------------")
    print(f"{'Tier':<18} | {'Count':<10} | {'Reuses':<10} | {'Observed Success Rate':<22}")
    print("-------------------------------------------------------------------------")
    for tier_name, t_info in tiers.items():
        print(f"{tier_name:<18} | {t_info.get('count', 0):<10} | {t_info.get('total_reuses', 0):<10} | {str(round(t_info.get('observed_success_rate', 0.0)*100, 2)) + '%':<22}")
    print("-------------------------------------------------------------------------\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
