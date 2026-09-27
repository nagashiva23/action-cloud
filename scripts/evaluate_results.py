#!/usr/bin/env python3
"""
Print the metrics the API computes from stored experiences (GET /metrics).

Note: stored `success` is each agent's own self-assessment. The experiment
scripts (run_experiment.py) report the verified outcome instead.

  python scripts/evaluate_results.py [--run-id RUN_ID]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import httpx

BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
KEYS = ["total_tasks", "success_rate_pct", "injection_rate_pct", "retrieval_precision_pct",
        "effective_reuse_rate_pct", "redundancy_index", "avg_tokens_per_task",
        "avg_latency_ms", "total_cost_usd"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--json", action="store_true", help="print raw JSON")
    args = ap.parse_args()
    try:
        resp = httpx.get(f"{BASE_URL}/metrics", params={"run_id": args.run_id} if args.run_id else {}, timeout=30)
        resp.raise_for_status()
    except Exception as e:  # noqa: BLE001
        print(f"Error fetching {BASE_URL}/metrics: {e}")
        return 1
    data = resp.json()
    if args.json:
        print(json.dumps(data, indent=2))
        return 0
    systems = data.get("systems", {})
    print(f"{'metric':28}" + "".join(f"{s:>14}" for s in systems))
    for k in KEYS:
        print(f"{k:28}" + "".join(f"{str(systems[s].get(k)):>14}" for s in systems))
    if data.get("comparative_metrics"):
        print("\nactioncloud vs baseline:", data["comparative_metrics"])
    print("\ntiers:", {t: v["count"] for t, v in data.get("governance_tier_distribution", {}).items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
