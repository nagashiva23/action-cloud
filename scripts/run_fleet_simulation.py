#!/usr/bin/env python3
"""
ActionCloud — heterogeneous fleet simulation.

Every task instance is executed by a DISTINCT agent (136 tasks x N epochs ->
one agent each) across all 12 roles, so every reuse is cross-agent and
knowledge can only spread through governance: same-role reuse promotes a
memory to SHARED, after which other roles can see it.

Reports how knowledge propagated: tier distribution, cross-role reuses, and
the headline metrics against a stateless baseline on the same workload.

  PYTHONPATH=src python scripts/run_fleet_simulation.py --epochs 3 --seed 1
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from actioncloud import db  # noqa: E402
from actioncloud.benchmark.harness import ACTIONCLOUD, BASELINE, Workload, run_arm  # noqa: E402
from actioncloud.benchmark.simulator import SimulatedAgentLLM  # noqa: E402
from actioncloud.llm import get_llm  # noqa: E402

UNIQUE = 10**9  # agents_per_role so large that every task gets its own agent


def propagation(run_id: str) -> dict:
    with db.get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT count(*) FILTER (WHERE r.counted) AS counted,
                   count(*) FILTER (WHERE r.counted AND split_part(r.reporter_agent_id, '-', 1)
                                    <> e.agent_role::text) AS cross_role,
                   count(DISTINCT e.agent_role) FILTER (WHERE e.tier >= 'shared') AS roles_sharing
            FROM reuse_events r JOIN experiences e ON e.id = r.experience_id
            WHERE e.run_id = %s
            """,
            (run_id,),
        )
        return dict(cur.fetchone())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--provider", default=os.environ.get("LLM_PROVIDER", "sim"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    from actioncloud.db import require_database  # noqa: PLC0415
    require_database()
    exp_id = f"fleet-{uuid.uuid4().hex[:8]}"
    wl = Workload(epochs=args.epochs, agents_per_role=UNIQUE, seed=args.seed)
    factory = (lambda s: SimulatedAgentLLM(seed=s)) if args.provider == "sim" else (lambda s: get_llm(args.provider))

    print(f"Fleet simulation {exp_id}: {len(wl.instances())} tasks, one agent each, 12 roles, provider={args.provider}\n")
    base = run_arm(BASELINE, wl, factory, exp_id)["metrics"]
    ac_run = run_arm(ACTIONCLOUD, wl, factory, exp_id)
    ac = ac_run["metrics"]
    prop = propagation(ac_run["run_id"])

    print(f"{'':28}{'baseline':>12}{'actioncloud':>14}")
    for key, label in [("success_rate_pct", "success %"), ("avg_tokens_per_task", "tokens / task"),
                       ("total_cost_usd", "total cost $"), ("redundancy_index", "redundancy index"),
                       ("retrieval_precision_pct", "retrieval precision %"),
                       ("poisoned_injection_pct", "poisoned injections %")]:
        print(f"  {label:26}{base[key]:>12}{ac[key]:>14}")
    print("\nGovernance tiers:", ac["tier_distribution"])
    print(f"Counted reuse reports: {prop['counted']}  (cross-role: {prop['cross_role']})")
    print(f"Roles with memories promoted to SHARED or above: {prop['roles_sharing']} / 12")
    return 0


if __name__ == "__main__":
    sys.exit(main())
