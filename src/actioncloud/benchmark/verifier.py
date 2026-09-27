"""
Per-task success verification.

Every task family (task_key) has a ground-truth procedure in procedures.json:
an ordered list of steps, each with keyword alternatives that a correct answer
must contain. The same verifier grades every arm and every LLM provider, so a
success-rate difference measures the system, not the grader.

Two verdicts are produced:

  strict   — every step satisfied. This is the task's TRUE outcome; it is what
             experiment metrics report and what a downstream consumer of a
             reused memory observes.
  lenient  — at most one step missing. This models an agent's own weak
             self-check ("looks done to me"). It is what the agent stores as
             `success` on its experience, so a slightly-wrong solution can be
             stored as a success — exactly the kind of bad memory the
             governance layer has to catch through reuse feedback.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import load_procedures


@dataclass
class Verdict:
    task_key: str
    satisfied: int
    total: int
    missing: list[str] = field(default_factory=list)

    @property
    def strict(self) -> bool:
        return self.total > 0 and self.satisfied == self.total

    @property
    def lenient(self) -> bool:
        return self.total > 0 and self.satisfied >= self.total - 1


class TaskVerifier:
    def __init__(self, procedures: dict[str, list[dict]] | None = None) -> None:
        self.procedures = procedures if procedures is not None else load_procedures()

    def steps(self, task_key: str) -> list[str]:
        return [s["step"] for s in self.procedures[task_key]]

    def verify(self, task_key: str, response_text: str) -> Verdict:
        if task_key not in self.procedures:
            raise KeyError(f"no ground-truth procedure for task_key {task_key!r}")
        text = (response_text or "").lower()
        missing = [
            s["step"] for s in self.procedures[task_key]
            if not any(kw in text for kw in s["any"])
        ]
        total = len(self.procedures[task_key])
        return Verdict(task_key, total - len(missing), total, missing)
