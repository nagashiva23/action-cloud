#!/usr/bin/env python3
"""Print the K-ablation table from results/k.json (produced by run_k_ablation.py)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiment import RESULTS_DIR, to_markdown  # noqa: E402

path = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS_DIR / "k.json"
if not path.exists():
    sys.exit(f"{path} not found — run scripts/run_k_ablation.py first")
print(to_markdown(json.loads(path.read_text())))
