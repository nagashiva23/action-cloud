from __future__ import annotations

import logging
import uuid
from typing import Optional, Tuple

from . import db
from .schema import Experience, MemoryTier

log = logging.getLogger(__name__)


class MemoryJudge:
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
    def evaluate_reuse(
        exp_dict: dict,
    ) -> Optional[Tuple[MemoryTier, float, str]]:
        """
        Evaluate an existing experience after a reuse event.

        Returns (new_tier, new_confidence, reason) if a tier change occurs,
        or None if tier remains unchanged.
        """
        exp_id = uuid.UUID(str(exp_dict["id"]))
        current_tier = MemoryTier(exp_dict["tier"])
        current_confidence = float(exp_dict["confidence"])
        reuse_count = int(exp_dict["reuse_count"])
        reuse_success_count = int(exp_dict["reuse_success_count"])

        if reuse_count <= 0:
            return None

        success_rate = reuse_success_count / reuse_count
        new_tier = current_tier
        new_confidence = current_confidence
        reason = ""

        # Demotion check (applies if reused at least 3 times with low success)
        if reuse_count >= 3 and success_rate < 0.4:
            if current_tier.rank > MemoryTier.PRIVATE.rank:
                new_tier = MemoryTier.PRIVATE
                new_confidence = max(0.1, current_confidence - 0.3)
                reason = (
                    f"Demoted to private: reuse success rate dropped to "
                    f"{success_rate:.1%} ({reuse_success_count}/{reuse_count})"
                )

        # Promotion checks
        elif current_tier == MemoryTier.AGENT:
            # Promote to SHARED after 1 successful reuse
            if reuse_success_count >= 1:
                new_tier = MemoryTier.SHARED
                new_confidence = 0.7
                reason = "Promoted to shared: proven by initial successful reuse"

        elif current_tier == MemoryTier.SHARED:
            # Promote to VALIDATED after >= 3 reuses with >= 80% success
            if reuse_count >= 3 and success_rate >= 0.8:
                new_tier = MemoryTier.VALIDATED
                new_confidence = 0.85
                reason = (
                    f"Promoted to validated: {reuse_count} reuses with "
                    f"{success_rate:.1%} success rate"
                )

        elif current_tier == MemoryTier.VALIDATED:
            # Promote to ORGANIZATIONAL after >= 10 reuses with >= 90% success
            if reuse_count >= 10 and success_rate >= 0.9:
                new_tier = MemoryTier.ORGANIZATIONAL
                new_confidence = 0.95
                reason = (
                    f"Promoted to organizational canonical knowledge: "
                    f"{reuse_count} reuses with {success_rate:.1%} success rate"
                )

        if new_tier != current_tier:
            log.info(
                "MemoryJudge: transition %s -> %s for experience %s (%s)",
                current_tier.value,
                new_tier.value,
                exp_id,
                reason,
            )
            # Update database record & audit log
            db.update_experience_tier(exp_id, new_tier, new_confidence)
            db.record_tier_transition(
                experience_id=exp_id,
                to_tier=new_tier,
                reason=reason,
                from_tier=current_tier,
                decided_by="memory_judge_v2",
            )
            return new_tier, new_confidence, reason

        return None
