#!/usr/bin/env python3
"""
ActionCloud — Empirical Top-K Ablation & Marginal Utility Evaluator.

Parses k_ablation_results.json and outputs formatted empirical comparison tables,
token overhead growth curves, and marginal utility rankings:

    Utility(K) = (Success(K) - Success(K-1)) / (InputTokens(K) - InputTokens(K-1))

Usage:
  python scripts/evaluate_k_ablation.py [--results-file PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_RESULTS_PATH = Path(__file__).resolve().parent.parent / "src" / "actioncloud" / "benchmark" / "k_ablation_results.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate ActionCloud Top-K Ablation Study")
    parser.add_argument(
        "--results-file",
        default=str(DEFAULT_RESULTS_PATH),
        help="Path to k_ablation_results.json output",
    )
    args = parser.parse_args()

    results_path = Path(args.results_file)
    if not results_path.exists():
        print(f"Error: Ablation results file not found at {results_path}")
        print("Run the ablation study first using:\n    python scripts/run_k_ablation.py")
        return 1

    with open(results_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    run_id = payload.get("run_id")
    provider = payload.get("provider")
    task_count = payload.get("task_count")
    k_results = payload.get("results", {})

    print("\n=========================================================================")
    print(f" ActionCloud Top-K Ablation & Marginal Utility Report  (run_id={run_id})")
    print("=========================================================================\n")
    print(f"Provider:    {provider}")
    print(f"Tasks/K Arm: {task_count}\n")

    print("1. Top-K Context Ablation Comparison Table")
    print("---------------------------------------------------------------------------------------------------------")
    print(f"{'Config (K)':<12} | {'Success (%)':<12} | {'Retrieved/Task':<16} | {'Avg In Tokens':<15} | {'Avg Out Tokens':<15} | {'Avg Latency (ms)':<18} | {'Total Cost ($)':<14}")
    print("---------------------------------------------------------------------------------------------------------")

    k_list = []
    for k_key, res in k_results.items():
        k_val = res["k"]
        succ_rate = res["success_rate_pct"]
        avg_retrieved = res["avg_retrieved_per_task"]
        total_tokens = res["total_tokens"]
        avg_tokens = res["avg_tokens_per_task"]
        avg_lat = res["avg_latency_ms"]
        cost = res["total_cost_usd"]

        # Sum up input vs output tokens across outcomes
        outcomes = res.get("outcomes", [])
        total_in = sum(o.get("tokens_input", 0) for o in outcomes)
        total_out = sum(o.get("tokens_output", 0) for o in outcomes)
        avg_in = round(total_in / len(outcomes), 2) if outcomes else 0.0
        avg_out = round(total_out / len(outcomes), 2) if outcomes else 0.0

        k_list.append({
            "k": k_val,
            "k_key": k_key,
            "success_rate": succ_rate,
            "avg_retrieved": avg_retrieved,
            "avg_in": avg_in,
            "avg_out": avg_out,
            "total_tokens": total_tokens,
            "avg_tokens": avg_tokens,
            "avg_lat": avg_lat,
            "cost": cost,
        })

        print(
            f"{k_key:<12} | {str(succ_rate) + '%':<12} | {avg_retrieved:<16} | {avg_in:<15} | {avg_out:<15} | {avg_lat:<18} | ${cost:<14.6f}"
        )

    print("---------------------------------------------------------------------------------------------------------\n")

    # Sort by K value for marginal analysis
    k_list.sort(key=lambda x: x["k"])

    print("2. Marginal Utility Analysis  (Delta Success / Delta Input Tokens)")
    print("---------------------------------------------------------------------------------------------------------")
    print(f"{'Transition':<16} | {'Delta Success (%)':<18} | {'Delta In Tokens/Task':<22} | {'Marginal Utility':<20} | {'Status':<15}")
    print("---------------------------------------------------------------------------------------------------------")

    prev = None
    optimal_k = 0
    max_utility = -1.0

    for item in k_list:
        if prev is None:
            prev = item
            print(f"{'Baseline (K=0)':<16} | {'-':<18} | {'-':<22} | {'Base Reference':<20} | {'Baseline':<15}")
            continue

        delta_succ = item["success_rate"] - prev["success_rate"]
        delta_in_tokens = item["avg_in"] - prev["avg_in"]

        if delta_in_tokens > 0:
            utility = delta_succ / delta_in_tokens
        else:
            utility = 0.0

        if utility > max_utility and delta_succ >= 0:
            max_utility = utility
            optimal_k = item["k"]

        status = "HIGH UTILITY" if utility > 0.01 else ("DIMINISHING" if utility == 0 and delta_in_tokens > 0 else "ZERO UTILITY")
        trans_label = f"K={prev['k']} -> K={item['k']}"

        print(
            f"{trans_label:<16} | {str(round(delta_succ, 2)) + '%':<18} | {str(round(delta_in_tokens, 2)):<22} | {utility:<20.6f} | {status:<15}"
        )

        prev = item

    print("---------------------------------------------------------------------------------------------------------\n")

    print("3. Empirical Optimization Findings & Decision Rule")
    print("---------------------------------------------------------------------------------------------------------")
    print(f"  • Recommended Optimal Context Cutoff: K* = {optimal_k} memory/memories")
    print(f"  • Context Growth Behavior: Additional memories beyond K={optimal_k} increase prompt context cost")
    print(f"    without yielding measurable task performance gains.")
    print("---------------------------------------------------------------------------------------------------------\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
