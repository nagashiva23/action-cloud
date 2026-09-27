"""
ActionCloud — agent identity.

Every agent is registered once (agent_id + role) and receives an API key.
Only the SHA-256 hash of the key is stored; keys are 256-bit random tokens, so
a plain hash is sufficient (no need for a slow password hash).

Where identity is enforced:
  * REST API — always, unless AUTH_REQUIRED=false. agent_id and role are taken
    from the key; a request body naming a different agent is rejected (403).
  * MCP server — when ACTIONCLOUD_API_KEY is set, the server acts as that one
    agent for every tool call. (The stdio server holds database credentials
    itself, so for it this binds honest clients to one identity rather than
    being a security boundary; the REST API is the boundary.)
  * In-process client (experiment harness) — trusted, not authenticated.

Registering agents is an admin action (ACTIONCLOUD_ADMIN_KEY or this CLI with
direct database access), so one operator cannot mint identities at will.

CLI:
    python -m actioncloud.auth create <agent_id> <role>   # prints the key once
    python -m actioncloud.auth revoke <agent_id>
    python -m actioncloud.auth list
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sys
from dataclasses import dataclass
from typing import Optional

from . import db
from .schema import AgentRole

KEY_PREFIX = "ac_"


@dataclass(frozen=True)
class Principal:
    agent_id: str
    role: AgentRole
    is_admin: bool = False


ADMIN = Principal(agent_id="__admin__", role=AgentRole.RESEARCH, is_admin=True)


def generate_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(32)


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def register_agent(agent_id: str, role: AgentRole | str, *, rotate: bool = False) -> str:
    """
    Create an agent and return its API key (the only time the key is visible).
    rotate=True replaces the key of an existing agent and un-revokes it.
    """
    role = AgentRole(role)
    key = generate_key()
    with db.get_conn() as conn, conn.cursor() as cur:
        if rotate:
            cur.execute(
                """
                INSERT INTO agents (agent_id, agent_role, key_hash) VALUES (%s, %s, %s)
                ON CONFLICT (agent_id) DO UPDATE
                SET agent_role = EXCLUDED.agent_role, key_hash = EXCLUDED.key_hash,
                    revoked_at = NULL
                """,
                (agent_id, role.value, hash_key(key)),
            )
        else:
            cur.execute(
                "INSERT INTO agents (agent_id, agent_role, key_hash) VALUES (%s, %s, %s)",
                (agent_id, role.value, hash_key(key)),
            )
        conn.commit()
    return key


def revoke_agent(agent_id: str) -> bool:
    with db.get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE agents SET revoked_at = now() WHERE agent_id = %s AND revoked_at IS NULL",
            (agent_id,),
        )
        changed = cur.rowcount > 0
        conn.commit()
    return changed


def resolve_key(key: Optional[str], admin_key: Optional[str] = None) -> Optional[Principal]:
    """Map an API key to its principal, or None if unknown/revoked."""
    if not key:
        return None
    if admin_key and hmac.compare_digest(key, admin_key):
        return ADMIN
    with db.get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT agent_id, agent_role::text AS role FROM agents "
            "WHERE key_hash = %s AND revoked_at IS NULL",
            (hash_key(key),),
        )
        row = cur.fetchone()
    return Principal(row["agent_id"], AgentRole(row["role"])) if row else None


def list_agents() -> list[dict]:
    with db.get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT agent_id, agent_role::text AS role, created_at, revoked_at "
            "FROM agents ORDER BY created_at"
        )
        return cur.fetchall()


def main(argv: list[str]) -> int:
    usage = "usage: python -m actioncloud.auth create <agent_id> <role> | revoke <agent_id> | list"
    if not argv:
        print(usage)
        return 2
    db.require_database()
    cmd = argv[0]
    if cmd == "create" and len(argv) == 3:
        try:
            key = register_agent(argv[1], argv[2])
        except ValueError:
            print(f"unknown role {argv[2]!r}; one of: {', '.join(r.value for r in AgentRole)}")
            return 2
        except Exception as e:  # noqa: BLE001
            print(f"could not create agent: {e}")
            return 1
        print(f"agent {argv[1]} ({argv[2]}) created. API key (shown once, store it now):\n{key}")
        return 0
    if cmd == "revoke" and len(argv) == 2:
        print("revoked" if revoke_agent(argv[1]) else "no active agent with that id")
        return 0
    if cmd == "list":
        for a in list_agents():
            status = f"revoked {a['revoked_at']:%Y-%m-%d}" if a["revoked_at"] else "active"
            print(f"{a['agent_id']:32} {a['role']:14} {status}")
        return 0
    print(usage)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
