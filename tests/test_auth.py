"""
Authentication and anti-gaming tests against the real REST API.

Each test is an attack that worked before migration 003 / auth.py.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.usefixtures("database")

ADMIN = "test-admin-key"


@pytest.fixture
def api(monkeypatch):
    from actioncloud import api as api_mod
    from actioncloud.config import settings
    from actioncloud.service import memory_service

    monkeypatch.setattr(settings, "auth_required", True)
    monkeypatch.setattr(settings, "admin_key", ADMIN)
    monkeypatch.setattr(memory_service, "sync_write", True)
    return TestClient(api_mod.app)


def _agent(api, role="coding"):
    agent_id = f"{role}-{uuid.uuid4().hex[:8]}"
    r = api.post("/agents", json={"agent_id": agent_id, "agent_role": role},
                 headers={"X-API-Key": ADMIN})
    assert r.status_code == 201, r.text
    return agent_id, {"X-API-Key": r.json()["api_key"]}


def _exp(run_id, success=True, **kw):
    body = dict(task="Configure pgvector HNSW index for cosine search",
                action="STEP: Create an HNSW index\nSTEP: Use vector_cosine_ops",
                solution="STEP: Create an HNSW index\nSTEP: Use vector_cosine_ops" if success else None,
                result="done", success=success, run_id=run_id, system="actioncloud",
                technologies=["postgres", "pgvector"], task_key="k")
    body.update(kw)
    return body


CTX = {"query": "Configure pgvector HNSW index for cosine search",
       "technologies": ["postgres", "pgvector"]}


def test_health_is_public_everything_else_needs_a_key(api):
    assert api.get("/health").status_code == 200
    assert api.get("/search", params={"q": "x"}).status_code == 401
    assert api.get("/search", params={"q": "x"}, headers={"X-API-Key": "ac_nope"}).status_code == 401
    assert api.get("/metrics").status_code == 401


def test_identity_comes_from_the_key(api):
    run = f"auth-{uuid.uuid4().hex[:6]}"
    me, h = _agent(api)
    eid = api.post("/experiences", json=_exp(run), headers=h).json()["id"]
    row = api.get(f"/experiences/{eid}", headers=h).json()
    assert row["agent_id"] == me and row["agent_role"] == "coding"


def test_cannot_write_as_someone_else(api):
    _, h = _agent(api)
    r = api.post("/experiences", json=_exp("r", agent_id="victim", agent_role="coding"), headers=h)
    assert r.status_code == 403
    r = api.post("/experiences", json=_exp("r", agent_role="security"), headers=h)
    assert r.status_code == 403


def test_cannot_read_another_agents_private_memories(api):
    """Old exploit: GET /search?agent_id=<victim> exposed the victim's PRIVATE rows."""
    run = f"auth-{uuid.uuid4().hex[:6]}"
    victim, vh = _agent(api)
    secret = api.post("/experiences", json=_exp(run, success=False), headers=vh).json()["id"]
    _, ah = _agent(api)
    assert api.get("/search", params={"q": "pgvector HNSW", "agent_id": victim,
                                      "scope_run_id": run}, headers=ah).status_code == 403
    own = api.get("/search", params={"q": "pgvector HNSW", "scope_run_id": run}, headers=ah).json()
    assert secret not in {r["id"] for r in own["results"]}
    assert api.get(f"/experiences/{secret}", headers=ah).status_code == 404
    assert api.get(f"/experiences/{secret}", headers=vh).status_code == 200


def test_cannot_report_as_someone_else(api):
    run = f"auth-{uuid.uuid4().hex[:6]}"
    _, ah = _agent(api)
    eid = api.post("/experiences", json=_exp(run), headers=ah).json()["id"]
    _, bh = _agent(api)
    r = api.post(f"/experiences/{eid}/reuse", json={"success": True, "agent_id": "someone-else"}, headers=bh)
    assert r.status_code == 403


def test_self_promotion_via_second_identity_is_blocked(api):
    """
    Old exploit: author reports success under a fake second agent_id -> SHARED.
    Now the reporter is the key's agent (author -> not counted), and a real
    second agent that was never shown the memory gets no credit either.
    """
    run = f"auth-{uuid.uuid4().hex[:6]}"
    _, ah = _agent(api)
    eid = api.post("/experiences", json=_exp(run), headers=ah).json()["id"]
    r = api.post(f"/experiences/{eid}/reuse", json={"success": True}, headers=ah).json()
    assert r["counted"] is False and r["tier"] == "agent"
    _, bh = _agent(api)
    r = api.post(f"/experiences/{eid}/reuse", json={"success": True}, headers=bh).json()
    assert r["counted"] is False and "no unreported injection" in r["reason"]


def test_genuine_reuse_promotes(api):
    run = f"auth-{uuid.uuid4().hex[:6]}"
    _, ah = _agent(api)
    eid = api.post("/experiences", json=_exp(run), headers=ah).json()["id"]
    _, bh = _agent(api)
    ctx = api.post("/context", json=CTX | {"scope_run_id": run}, headers=bh).json()
    assert ctx["injected_experience_ids"] == [eid]
    r = api.post(f"/experiences/{eid}/reuse", json={"success": True}, headers=bh).json()
    assert r["counted"] is True and r["transition"] == "shared"
    r = api.post(f"/experiences/{eid}/reuse", json={"success": True}, headers=bh).json()
    assert r["counted"] is False                                   # one injection, one report


def test_admin_endpoints(api):
    _, h = _agent(api)
    assert api.post("/agents", json={"agent_id": "x", "agent_role": "ml"}, headers=h).status_code == 403
    agent_id, h2 = _agent(api, "ml")
    assert api.get("/metrics", headers=h2).status_code == 200
    assert api.delete(f"/agents/{agent_id}", headers={"X-API-Key": ADMIN}).status_code == 200
    assert api.get("/metrics", headers=h2).status_code == 401      # revoked key stops working


def test_bearer_header_also_accepted(api):
    _, h = _agent(api)
    assert api.get("/metrics", headers={"Authorization": f"Bearer {h['X-API-Key']}"}).status_code == 200


def test_auth_off_keeps_legacy_behaviour(api, monkeypatch):
    from actioncloud.config import settings

    monkeypatch.setattr(settings, "auth_required", False)
    r = api.post("/experiences", json=_exp("legacy", agent_id="a", agent_role="coding"))
    assert r.status_code == 202


def test_keys_are_stored_hashed(api):
    from actioncloud import db

    agent_id, h = _agent(api)
    with db.get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT key_hash FROM agents WHERE agent_id = %s", (agent_id,))
        stored = cur.fetchone()["key_hash"]
    assert h["X-API-Key"] not in stored and len(stored) == 64


def test_mcp_server_bound_to_key_identity():
    from actioncloud import auth
    from actioncloud.mcp_server import ActionCloudMCPServer

    agent_id = f"mcp-{uuid.uuid4().hex[:8]}"
    key = auth.register_agent(agent_id, "deployment")
    s = ActionCloudMCPServer(api_key=key)
    text, err = s.call_tool("remember_experience", {"task": "t", "action_taken": "a", "result": "r",
                                                    "success": True, "agent_id": "impostor"})
    assert err and "bound to agent" in text
    text, err = s.call_tool("search_memory", {"query": "anything"})
    assert not err
    with pytest.raises(SystemExit):
        ActionCloudMCPServer(api_key="ac_revoked_or_unknown")
