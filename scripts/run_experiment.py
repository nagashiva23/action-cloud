#!/usr/bin/env python3
"""
ActionCloud — run the evaluation.

Presets:
  main       baseline vs flat shared memory vs ActionCloud (governed)
  k          ActionCloud with k_inject = 1, 2, 3, 5
  threshold  ActionCloud with similarity threshold 0.0 (inject top hit) vs 0.30
  feedback   ActionCloud when 30 % / 70 % / 100 % of task outcomes are observed
  all        every preset above

Needs only Postgres (in-process client; no API, worker or LocalStack).

  PYTHONPATH=src python scripts/run_experiment.py --preset all --seeds 1,2,3,4,5
  PYTHONPATH=src LLM_PROVIDER=anthropic python scripts/run_experiment.py --preset main --seeds 1 --epochs 1

Writes results/<preset>.json and a markdown table to results/<preset>.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from actioncloud.benchmark.harness import (  # noqa: E402
    ACTIONCLOUD, BASELINE, FLAT, ArmConfig, Workload, run_experiment,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

PRESETS: dict[str, list[ArmConfig]] = {
    "main": [BASELINE, FLAT, ACTIONCLOUD],
    "k": [BASELINE] + [replace(ACTIONCLOUD, name=f"k{k}", k_inject=k) for k in (1, 2, 3, 5)],
    "threshold": [BASELINE,
                  replace(ACTIONCLOUD, name="threshold_0.00", similarity_threshold=0.0),
                  replace(ACTIONCLOUD, name="threshold_0.30", similarity_threshold=0.30)],
    "feedback": [BASELINE, FLAT] + [
        replace(ACTIONCLOUD, name=f"feedback_{int(f * 100)}", feedback_rate=f) for f in (0.3, 0.7, 1.0)
    ],
}

COLUMNS = [
    ("success_rate_pct", "Success %", "{:.1f}"),
    ("retrieval_precision_pct", "Precision %", "{:.1f}"),
    ("injection_rate_pct", "Injected %", "{:.1f}"),
    ("misled_rate_pct", "Misled %", "{:.1f}"),
    ("poisoned_injection_pct", "Poisoned inj. %", "{:.1f}"),
    ("redundancy_index", "Redundancy idx", "{:.3f}"),
    ("avg_tokens_per_task", "Tokens/task", "{:.0f}"),
    ("avg_context_tokens", "Ctx tokens", "{:.0f}"),
    ("cost_per_success_usd", "$/success", "{:.4f}"),
]


def to_markdown(result: dict) -> str:
    seeds = result["seeds"]
    lines = [
        f"Provider: `{result['provider']}` · seeds: {seeds} · "
        f"{result['workload']['tasks_per_epoch']} tasks × {result['workload']['epochs']} epochs "
        f"per arm · values are mean ± sd across seeds",
        "",
        "| Arm | " + " | ".join(c[1] for c in COLUMNS) + " |",
        "|---|" + "---:|" * len(COLUMNS),
    ]
    for arm, s in result["summary"].items():
        cells = []
        for key, _, fmt in COLUMNS:
            if key in s:
                cells.append(fmt.format(s[key]["mean"]) + (f" ± {fmt.format(s[key]['sd'])}" if len(seeds) > 1 else ""))
            else:
                cells.append("—")
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    if result["vs_baseline"]:
        lines += ["", "| vs baseline | Token savings % | Cost savings % | Latency reduction % | Success Δ (pp) |",
                  "|---|---:|---:|---:|---:|"]
        for arm, d in result["vs_baseline"].items():
            f = lambda k: f"{d[k]['mean']:.1f}" + (f" ± {d[k]['sd']:.1f}" if len(seeds) > 1 else "")
            lines.append(f"| {arm} | {f('token_savings_pct')} | {f('cost_savings_pct')} | "
                         f"{f('latency_reduction_pct')} | {f('success_rate_delta_pp')} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="main", choices=[*PRESETS, "all"])
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--agents-per-role", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N tasks")
    ap.add_argument("--provider", default=os.environ.get("LLM_PROVIDER", "sim"))
    ap.add_argument("--out", default=str(RESULTS_DIR))
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)
    from actioncloud.db import require_database  # noqa: PLC0415
    require_database()
    seeds = [int(s) for s in args.seeds.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    presets = list(PRESETS) if args.preset == "all" else [args.preset]

    for name in presets:
        t0 = time.time()
        print(f"\n=== {name}: arms={[a.name for a in PRESETS[name]]} seeds={seeds} provider={args.provider}")
        result = run_experiment(
            PRESETS[name], seeds,
            Workload(epochs=args.epochs, agents_per_role=args.agents_per_role, limit=args.limit),
            provider=args.provider, progress=lambda m: print("  ", m, flush=True),
        )
        (out / f"{name}.json").write_text(json.dumps(result, indent=1, default=str))
        md = to_markdown(result)
        (out / f"{name}.md").write_text(f"## {name}\n\n{md}")
        print(md)
        print(f"({time.time() - t0:.0f}s) -> {out / (name + '.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
