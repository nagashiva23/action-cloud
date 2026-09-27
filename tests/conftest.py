from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def database():
    """Skip DB-backed tests cleanly when Postgres (make up) is not running."""
    from actioncloud import db

    if not db.health_check():
        pytest.skip("Postgres not reachable — run `make up` (and `make migrate`)")
    return db
