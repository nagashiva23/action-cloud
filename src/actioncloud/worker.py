"""
Stub worker — Phase 1 queue consumer.

All it does is read events and persist raw experiences. No LLM extraction, no
embeddings, no graph. That is the point: this proves the asynchronous path
works before anything that can fail *subtly* is added to it.

In Phase 2 this file splits into three workers (Memory / Embedding / Graph)
consuming the same queue. The dispatch structure below is already shaped for
that, so adding them is a matter of registering handlers rather than rewriting
the loop.

Run with:  python -m actioncloud.worker
"""

from __future__ import annotations

import logging
import signal
import sys
import time
from typing import Any, Callable

from . import db, queue
from .schema import Experience, MemoryTier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s [worker] %(message)s",
)
log = logging.getLogger(__name__)

_running = True


def _stop(signum, frame):  # noqa: ARG001
    """
    Graceful shutdown.

    Without this, Ctrl-C mid-message means the message is neither processed nor
    acknowledged — it just reappears after the visibility timeout. Harmless
    here, but the habit matters once workers do expensive LLM calls you would
    rather not pay for twice.
    """
    global _running
    log.info("shutdown signal received — finishing current batch")
    _running = False


signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)


# --------------------------------------------------------------------------
# Handlers
# --------------------------------------------------------------------------

def assign_initial_tier(exp: Experience) -> tuple[MemoryTier, str]:
    """
    Placeholder for the Memory Judge — REPLACE IN PHASE 2.

    The real Judge weighs provenance, confidence, and observed success rate
    across repeated reuse. This stand-in does something far cruder: it promotes
    any successful experience straight to SHARED, and leaves failures private.

    Why a stand-in is needed at all: the schema defaults every experience to
    PRIVATE (untrusted until proven), and search correctly refuses to show one
    agent's private memory to another. With no Judge, nothing would ever leave
    PRIVATE and cross-agent retrieval — the entire point of the system — could
    never happen. So Phase 1 needs *something* occupying this slot.

    Be clear-eyed about what this costs: promoting on first success is exactly
    the ungoverned behaviour the proposal argues against, because one wrong
    solution propagates fleet-wide unchallenged. That is acceptable only while
    Phase 1 is testing plumbing, and it is the first thing Phase 2 must fix.
    """
    if exp.success:
        return MemoryTier.SHARED, "phase-1 placeholder: promoted on first success"
    return MemoryTier.PRIVATE, "phase-1 placeholder: failures stay private"


def handle_experience_created(payload: dict[str, Any]) -> None:
    """
    Persist a newly submitted experience and assign its initial tier.

    Phase 2 will extend this to: extract knowledge + workflow via LLM, enqueue
    embedding and graph jobs, then hand off to the real Memory Judge.
    """
    exp = Experience(**payload)

    tier, reason = assign_initial_tier(exp)
    exp.tier = tier

    db.insert_experience(exp)  # idempotent — safe under at-least-once redelivery

    # Audit the decision separately from the row it affects. The proposal
    # promises auditability of what was shared and by whom; that is only
    # credible if the decision trail survives independently of the record.
    db.record_tier_transition(
        experience_id=exp.id,
        to_tier=tier,
        reason=reason,
        from_tier=None,
        decided_by="phase1-placeholder-judge",
    )

    log.info(
        "stored %s | %s/%s | success=%s | tier=%s | %d tokens",
        exp.id,
        exp.agent_role.value,
        exp.system.value,
        exp.success,
        tier.value,
        exp.total_tokens,
    )


HANDLERS: dict[str, Callable[[dict[str, Any]], None]] = {
    "experience.created": handle_experience_created,
}


# --------------------------------------------------------------------------
# Loop
# --------------------------------------------------------------------------

def process_one(message: dict[str, Any]) -> bool:
    """
    Process a single message. Returns True if it should be deleted.

    A message with no registered handler is deleted rather than retried: an
    unknown event type will never become known by being redelivered, and
    leaving it in the queue turns one bad message into an infinite loop.
    Genuine processing failures return False so SQS redelivers them.
    """
    try:
        event_type, payload = queue.parse(message)
    except Exception:  # noqa: BLE001
        log.exception("unparseable message — discarding")
        return True

    handler = HANDLERS.get(event_type)
    if handler is None:
        log.warning("no handler for event type %r — discarding", event_type)
        return True

    try:
        handler(payload)
        return True
    except Exception:  # noqa: BLE001
        log.exception("handler failed for %s — leaving on queue for retry", event_type)
        return False


def run(poll_wait_seconds: int = 20, idle_sleep: float = 0.5) -> None:
    # Announce the mode *before* touching AWS. If .env failed to load, this line
    # says "aws (real)" and you know immediately why the next call is asking for
    # credentials — instead of reading a 40-line boto3 traceback.
    from .config import settings  # noqa: PLC0415

    log.info(
        "mode=%s endpoint=%s",
        "local (localstack)" if settings.is_local else "aws (real)",
        settings.aws_endpoint_url or "default AWS",
    )
    log.info("worker starting (queue=%s)", queue.get_queue_url())
    processed = 0

    while _running:
        try:
            messages = queue.receive(max_messages=10, wait_seconds=poll_wait_seconds)
        except Exception:  # noqa: BLE001
            # Usually a transient network or credential problem. Back off rather
            # than exiting, so a brief blip does not end the run.
            log.exception("receive failed — backing off")
            time.sleep(5)
            continue

        if not messages:
            time.sleep(idle_sleep)
            continue

        for msg in messages:
            if process_one(msg):
                queue.delete(msg["ReceiptHandle"])
                processed += 1

    log.info("worker stopped after processing %d message(s)", processed)


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        sys.exit(0)
