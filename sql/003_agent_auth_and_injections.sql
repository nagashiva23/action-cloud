-- ActionCloud — migration 003: authenticated agent identity + injection ledger
--
-- 1. agents: the registry of who may talk to ActionCloud. Each agent has a
--    fixed role and an API key; only the SHA-256 hash of the key is stored.
--    The REST API derives agent_id and role from the key, so a caller can no
--    longer claim another agent's identity (to read its PRIVATE memories, or
--    to report reuse as a "different" agent and dodge the self-report rule).
--
-- 2. injections: one row every time get_memory_context/POST /context puts a
--    memory in front of an agent. A reuse report is counted only against an
--    unreported injection of that memory to that same agent — one injection,
--    at most one counted report. Without this, a single agent could report
--    "success" ten times on one memory and promote it to ORGANIZATIONAL alone.
--
-- Fresh Docker volumes run this automatically; existing ones: `make migrate`.

CREATE TABLE IF NOT EXISTS agents (
    agent_id    TEXT        PRIMARY KEY,
    agent_role  agent_role  NOT NULL,
    key_hash    TEXT        NOT NULL UNIQUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    revoked_at  TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS injections (
    id            BIGSERIAL   PRIMARY KEY,
    experience_id UUID        NOT NULL REFERENCES experiences(id) ON DELETE CASCADE,
    agent_id      TEXT        NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    reported_at   TIMESTAMPTZ
);

-- apply_reuse looks up "oldest unreported injection of X to agent A".
CREATE INDEX IF NOT EXISTS idx_injections_open
    ON injections (experience_id, agent_id, created_at)
    WHERE reported_at IS NULL;
