"""
ActionCloud — Memory Judge.

The Judge is a *pure decision function*: given an experience's governance
state it says which tier the experience belongs in and why. It never touches
the database. `db.apply_reuse` calls it inside the same transaction (and row
lock) that increments the reuse counters, so two concurrent reuse reports can
never both read stale counts and double-apply a transition.

Ladder (see README):
    PRIVATE -> AGENT -> SHARED -> VALIDATED -> ORGANIZATIONAL

  * Successful experience enters at AGENT (visible to agents of the same role).
    ("Successful" is the agent's own self-assessment at write time.)
  * Failed experience enters at PRIVATE (visible only to its author).
  * AGENT  -> SHARED          after >= 1 counted successful reuse.
  * SHARED -> VALIDATED       after >= 3 counted reuses at >= 80 % success.
  * VALIDATED -> ORGANIZATIONAL after >= 10 counted reuses at >= 90 % success.
  * Any tier above PRIVATE -> PRIVATE (quarantine) when >= 3 counted reuses
    fall below 40 % success. Quarantine is terminal for automatic governance:
    only the author can still see the memory, and the author's own reuse
    reports are never counted, so it cannot climb back without a human.

"Counted" outcome reports exclude POSITIVE self-reports (reporter == author,
success). Without that rule an agent could promote its own memory fleet-wide
by vouching for it once. NEGATIVE self-reports do count: an author learning
downstream (tests, CI, a failed deploy) that its own solution was wrong is
exactly the evidence governance needs, and it stops the memory spreading
before anyone else is misled by it.

Confidence is a running estimate of "will this memory work if reused?",
updated on EVERY counted reuse (not only on tier changes):

    confidence = (successes + PRIOR_WEIGHT * prior) / (reuses + PRIOR_WEIGHT)

i.e. the posterior mean of a Beta prior centred on the initial confidence.
Retrieval skips memories below MemorySelectionPolicy.min_confidence, so a
memory that fails its first reuses stops being injected immediately, long
before it has enough evidence to be formally quarantined.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from .schema import Experience, MemoryTier

log = logging.getLogger(__name__)

DECIDED_BY = "memory_judge_v3"

PRIOR_WEIGHT = 2.0
DEFAULT_PRIOR = 0.6


@dataclass(frozen=True)
class JudgeThresholds:
    shared_min_successes: int = 1
    validated_min_reuses: int = 3
    validated_min_rate: float = 0.80
    org_min_reuses: int = 10
    org_min_rate: float = 0.90
    demote_min_reuses: int = 3
    demote_max_rate: float = 0.40


DEFAULT_THRESHOLDS = JudgeThresholds()


class MemoryJudge:
    thresholds: JudgeThresholds = DEFAULT_THRESHOLDS

    @staticmethod
    def assign_initial_tier(exp: Experience) -> Tuple[MemoryTier, float, str]:
        if not exp.success:
            return (
                MemoryTier.PRIVATE,
                0.3,
                "Initial creation: failed experiences remain private",
            )
        confidence = 0.6 if exp.solution else 0.5
        tier = MemoryTier.AGENT
        return tier, confidence, f"Initial creation: assigned {tier.value} tier"

    @staticmethod
    def posterior_confidence(successes: int, reuses: int, prior: float = DEFAULT_PRIOR) -> float:
        return round((successes + PRIOR_WEIGHT * prior) / (reuses + PRIOR_WEIGHT), 4)

    @classmethod
    def decide(
        cls,
        tier: MemoryTier,
        confidence: float,
        reuse_count: int,
        reuse_success_count: int,
    ) -> Optional[Tuple[MemoryTier, float, str]]:
        """
        Pure tier decision. Returns (new_tier, new_confidence, reason) when the
        tier should change, otherwise None. (Confidence is updated separately
        on every counted reuse via posterior_confidence.)
        """
        th = cls.thresholds
        if reuse_count <= 0:
            return None
        rate = reuse_success_count / reuse_count
        conf = cls.posterior_confidence(reuse_success_count, reuse_count)

        if reuse_count >= th.demote_min_reuses and rate < th.demote_max_rate:
            if tier.rank > MemoryTier.PRIVATE.rank:
                return (
                    MemoryTier.PRIVATE,
                    conf,
                    f"Quarantined to private: reuse success rate {rate:.1%} "
                    f"({reuse_success_count}/{reuse_count}) below {th.demote_max_rate:.0%}",
                )
            return None

        if tier == MemoryTier.AGENT and reuse_success_count >= th.shared_min_successes:
            return (
                MemoryTier.SHARED,
                conf,
                "Promoted to shared: successful reuse by another agent",
            )
        if (
            tier == MemoryTier.SHARED
            and reuse_count >= th.validated_min_reuses
            and rate >= th.validated_min_rate
        ):
            return (
                MemoryTier.VALIDATED,
                conf,
                f"Promoted to validated: {reuse_count} reuses at {rate:.1%} success",
            )
        if (
            tier == MemoryTier.VALIDATED
            and reuse_count >= th.org_min_reuses
            and rate >= th.org_min_rate
        ):
            return (
                MemoryTier.ORGANIZATIONAL,
                conf,
                f"Promoted to organizational: {reuse_count} reuses at {rate:.1%} success",
            )
        return None

    @classmethod
    def evaluate_reuse(cls, exp_dict: dict) -> Optional[Tuple[MemoryTier, float, str]]:
        """Backward-compatible wrapper: decide from a row dict (no side effects)."""
        return cls.decide(
            MemoryTier(exp_dict["tier"]),
            float(exp_dict["confidence"]),
            int(exp_dict["reuse_count"]),
            int(exp_dict["reuse_success_count"]),
        )
