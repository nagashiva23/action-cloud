-- ActionCloud — migration 002
--
-- 1. Extends agent_role to all 12 fleet roles. 001 originally declared only 6,
--    so every experience from security/devops/database/ml/cloud/monitoring
--    agents was rejected by Postgres (and retried forever by the worker).
--    ADD VALUE IF NOT EXISTS makes this safe on both old and fresh databases.
--
-- 2. Adds reuse_events: an append-only record of WHO reported each reuse
--    outcome. The Memory Judge ignores positive self-reports (an author
--    vouching for its own memory); this table makes that rule auditable.
--
-- 3. Adds an HNSW index for cosine search over embeddings.
--
-- Fresh Docker volumes run this automatically after 001 (filename order).
-- Existing volumes: `make migrate`.

ALTER TYPE agent_role ADD VALUE IF NOT EXISTS 'security';
ALTER TYPE agent_role ADD VALUE IF NOT EXISTS 'devops';
ALTER TYPE agent_role ADD VALUE IF NOT EXISTS 'database';
ALTER TYPE agent_role ADD VALUE IF NOT EXISTS 'ml';
ALTER TYPE agent_role ADD VALUE IF NOT EXISTS 'cloud';
ALTER TYPE agent_role ADD VALUE IF NOT EXISTS 'monitoring';

CREATE TABLE IF NOT EXISTS reuse_events (
    id                BIGSERIAL PRIMARY KEY,
    experience_id     UUID        NOT NULL REFERENCES experiences(id) ON DELETE CASCADE,
    reporter_agent_id TEXT        NOT NULL,
    success           BOOLEAN     NOT NULL,
    counted           BOOLEAN     NOT NULL,   -- false = positive self-report, ignored by the Judge
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_reuse_events_exp
    ON reuse_events (experience_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_exp_embedding_hnsw
    ON experiences USING hnsw (embedding vector_cosine_ops);

-- Retrieval can be scoped to one experimental run so ablation arms never
-- read each other's memories.
CREATE INDEX IF NOT EXISTS idx_exp_run ON experiences (run_id);
