from __future__ import annotations

import logging
import signal
import sys
import time
from typing import Any, Callable

from . import db, queue
from .schema import Experience

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

from pydantic import ValidationError

from .pipeline import process_experience

# After this many failed deliveries a message is dropped (and logged) instead
# of being retried forever. On AWS, also attach a real dead-letter queue with a
# redrive policy so dropped messages can be inspected.
MAX_RECEIVES = int(__import__("os").environ.get("WORKER_MAX_RECEIVES", "5"))


class PermanentError(Exception):
    """Processing can never succeed for this payload; do not retry."""


def handle_experience_created(payload: dict[str, Any]) -> None:
    try:
        exp = Experience(**payload)
    except ValidationError as e:
        raise PermanentError(f"invalid experience payload: {e}") from e
    import os  # noqa: PLC0415
    process_experience(exp, author_prior=os.environ.get("MEMORY_AUTHOR_PRIOR", "false").lower()
                       in {"1", "true", "yes"})


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
    except PermanentError:
        log.exception("permanent failure for %s — discarding", event_type)
        return True
    except Exception:  # noqa: BLE001
        receives = int(message.get("Attributes", {}).get("ApproximateReceiveCount", "1"))
        if receives >= MAX_RECEIVES:
            log.exception(
                "handler failed for %s on delivery %d/%d — giving up and discarding",
                event_type, receives, MAX_RECEIVES,
            )
            return True
        log.exception(
            "handler failed for %s (delivery %d/%d) — leaving on queue for retry",
            event_type, receives, MAX_RECEIVES,
        )
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
