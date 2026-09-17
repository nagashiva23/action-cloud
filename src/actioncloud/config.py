from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_bool(key: str, default: bool = False) -> bool:
    return os.environ.get(key, str(default)).strip().lower() in {"1", "true", "yes"}


@dataclass
class Settings:
    # --- Postgres ---------------------------------------------------------
    db_host: str = field(default_factory=lambda: _env("DB_HOST", "localhost"))
    db_port: int = field(default_factory=lambda: int(_env("DB_PORT", "5432")))
    db_name: str = field(default_factory=lambda: _env("DB_NAME", "actioncloud"))
    db_user: str = field(default_factory=lambda: _env("DB_USER", "actioncloud"))
    db_password: str = field(default_factory=lambda: _env("DB_PASSWORD", "actioncloud"))

    # --- Queue (SQS or LocalStack) ---------------------------------------
    # Locally this points at LocalStack. In AWS, leave AWS_ENDPOINT_URL unset
    # and boto3 resolves the real endpoint automatically.
    aws_region: str = field(default_factory=lambda: _env("AWS_REGION", "us-east-1"))
    aws_endpoint_url: str | None = field(
        default_factory=lambda: os.environ.get("AWS_ENDPOINT_URL") or None
    )
    queue_name: str = field(default_factory=lambda: _env("QUEUE_NAME", "actioncloud-experiences"))

    # --- Embeddings -------------------------------------------------------
    # Dimension is fixed at the DB level (vector(N)), so changing model means a
    # migration. 1536 matches OpenAI text-embedding-3-small.
    embedding_dim: int = field(default_factory=lambda: int(_env("EMBEDDING_DIM", "1536")))

    # --- Behaviour flags --------------------------------------------------
    # When true, the API writes straight to Postgres instead of publishing to
    # the queue. Useful for unit tests; never enable in the experiment, since
    # it would change the latency profile you are trying to measure.
    sync_write: bool = field(default_factory=lambda: _env_bool("SYNC_WRITE", False))

    @property
    def dsn(self) -> str:
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def is_local(self) -> bool:
        return self.aws_endpoint_url is not None


settings = Settings()
