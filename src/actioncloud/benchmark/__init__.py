"""Benchmark data (tasks.json, procedures.json) and the evaluation harness."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
TASKS_PATH = DATA_DIR / "tasks.json"
PROCEDURES_PATH = DATA_DIR / "procedures.json"


@lru_cache(maxsize=1)
def load_tasks() -> tuple[dict, ...]:
    return tuple(json.loads(TASKS_PATH.read_text(encoding="utf-8")))


@lru_cache(maxsize=1)
def load_procedures() -> dict[str, list[dict]]:
    return json.loads(PROCEDURES_PATH.read_text(encoding="utf-8"))
