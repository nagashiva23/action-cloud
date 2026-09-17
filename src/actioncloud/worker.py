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
    global _running
    log.info("shutdown signal received — finishing current batch")
    _running = False


signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)


# --------------------------------------------------------------------------
# Handlers
# --------------------------------------------------------------------------

from .embeddings import get_embedding_provider
from .extractor import ExperienceExtractor
from .judge import MemoryJudge


def handle_experience_created(payload: dict[str, Any]) -> None:
    """
    Phase 2 processing:
      1. Assign initial tier and confidence via MemoryJudge.
      2. Insert experience into DB.
      3. Extract procedural workflow and knowledge triples via LLM out-of-band.
      4. Generate text embedding vector.
      5. Persist extractions & vector to DB.
      6. Audit tier decision in tier_transitions.
    """
    exp = Experience(**payload)

    tier, confidence, reason = MemoryJudge.assign_initial_tier(exp)
    exp.tier = tier
    exp.confidence = confidence

    # 1. Primary insert (idempotent)
    db.insert_experience(exp)

    # 2. Extract workflow & knowledge triples via LLM
    extractor = ExperienceExtractor()
    workflow, triples = extractor.extract(exp)

    # 3. Compute vector embedding
    embedder = get_embedding_provider()
    embedding_text = f"{exp.task} {exp.problem or ''} {exp.solution or ''} {exp.result}"
    embedding = embedder.embed(embedding_text)

    # 4. Save extractions & embedding
    db.update_experience_extractions(
        experience_id=exp.id,
        workflow=workflow,
        knowledge_triples=triples,
        embedding=embedding,
        embedded=True,
    )

    # 5. Record tier transition audit log
    db.record_tier_transition(
        experience_id=exp.id,
        to_tier=tier,
        reason=reason,
        from_tier=None,
        decided_by="memory_judge_v2",
    )

    log.info(
        "stored & enriched %s | %s/%s | success=%s | tier=%s | %d tokens",
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
