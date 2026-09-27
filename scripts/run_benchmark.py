#!/usr/bin/env python3
"""Baseline vs flat memory vs ActionCloud. Alias for: run_experiment.py --preset main"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiment import main  # noqa: E402

if __name__ == "__main__":
    if "--preset" not in sys.argv:
        sys.argv[1:1] = ["--preset", "main"]
    sys.exit(main())
