from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import boto3
from botocore.exceptions import ClientError

from .config import settings

log = logging.getLogger(__name__)

_client = None
_queue_url: Optional[str] = None


def get_client():
    """
    boto3 SQS client. Honours AWS_ENDPOINT_URL so LocalStack works unchanged.

    When an endpoint override is set we are talking to LocalStack, so dummy
    credentials are passed *explicitly*. Without this, boto3 walks its usual
    credential chain and may pick up your real ~/.aws/credentials — which would
    work fine against LocalStack and then, the day a config line goes missing,
    silently create real billable resources instead of failing. Pinning fake
    credentials in local mode makes that failure impossible rather than
    unlikely.
    """
    global _client
    if _client is None:
        kwargs: dict[str, Any] = {"region_name": settings.aws_region}
        if settings.aws_endpoint_url:
            kwargs["endpoint_url"] = settings.aws_endpoint_url
            kwargs["aws_access_key_id"] = os.environ.get("AWS_ACCESS_KEY_ID", "test")
            kwargs["aws_secret_access_key"] = os.environ.get("AWS_SECRET_ACCESS_KEY", "test")
        _client = boto3.client("sqs", **kwargs)
    return _client


def get_queue_url(create_if_missing: bool = True) -> str:
    """
    Resolve (and locally, create) the queue URL.

    Auto-creation is convenient against LocalStack. In real AWS the queue
    should be created by your infrastructure definition, not by application
    code — but leaving this in costs nothing and avoids a confusing first-run
    failure while you are still learning the tooling.
    """
    global _queue_url
    if _queue_url:
        return _queue_url

    client = get_client()
    try:
        _queue_url = client.get_queue_url(QueueName=settings.queue_name)["QueueUrl"]
    except ClientError as e:
        nonexistent = e.response["Error"]["Code"] in (
            "AWS.SimpleQueueService.NonExistentQueue",
            "QueueDoesNotExist",
        )
        if nonexistent and create_if_missing:
            log.info("Queue %s missing — creating it", settings.queue_name)
            _queue_url = client.create_queue(QueueName=settings.queue_name)["QueueUrl"]
        else:
            raise
    return _queue_url


def publish(event_type: str, payload: dict[str, Any]) -> str:
    """
    Publish one event. Returns the SQS message id.

    The envelope (type + payload) exists so a single queue can carry more event
    kinds in Phase 2 — 'experience.created', 'experience.reused',
    'experience.superseded' — without a breaking change to consumers.
    """
    body = json.dumps({"type": event_type, "payload": payload}, default=str)
    resp = get_client().send_message(QueueUrl=get_queue_url(), MessageBody=body)
    log.debug("published %s (%s)", event_type, resp["MessageId"])
    return resp["MessageId"]


def receive(max_messages: int = 10, wait_seconds: int = 20) -> list[dict[str, Any]]:
    """
    Pull messages.

    wait_seconds enables long polling: the call blocks server-side until a
    message arrives or the timeout expires, instead of the worker hammering SQS
    in a busy loop. On real AWS that difference is billable — short polling a
    quiet queue is a genuinely common way to run up a surprise bill.
    """
    resp = get_client().receive_message(
        QueueUrl=get_queue_url(),
        MaxNumberOfMessages=max_messages,
        WaitTimeSeconds=wait_seconds,
    )
    return resp.get("Messages", [])


def delete(receipt_handle: str) -> None:
    """
    Acknowledge a message.

    Until this is called the message stays invisible but undeleted, and SQS
    redelivers it after the visibility timeout. That is the at-least-once
    guarantee: a worker that crashes mid-processing loses nothing, because the
    message comes back. It is also why inserts must be idempotent.
    """
    get_client().delete_message(QueueUrl=get_queue_url(), ReceiptHandle=receipt_handle)


def parse(message: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Unwrap an SQS message into (event_type, payload)."""
    body = json.loads(message["Body"])
    return body["type"], body["payload"]


def purge() -> None:
    """Empty the queue. Test helper — do not call during an experiment run."""
    try:
        get_client().purge_queue(QueueUrl=get_queue_url())
    except ClientError as e:
        log.warning("purge failed: %s", e)
