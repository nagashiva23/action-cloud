-- ActionCloud — initial schema (Phase 1)
--
-- Design notes worth keeping in mind:
--
-- 1. Phase 2 columns (embedding, workflow, knowledge_triples) exist NOW as
--    nullable columns. Adding them later would mean migrating live experiment
--    data, which is exactly the kind of avoidable risk that eats a semester.
--
-- 2. Governance columns default to the least-trusted state. An experience is
--    private and unproven until the Memory Judge says otherwise.
--
-- 3. run_id / system / task_key are indexed because every Phase 3 metric is a
--    GROUP BY over them.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enums keep bad values out at the database boundary, not just in Python.
DO $$ BEGIN
    CREATE TYPE memory_tier AS ENUM (
        'private', 'agent', 'shared', 'validated', 'organizational'
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE system_condition AS ENUM ('baseline', 'actioncloud');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE agent_role AS ENUM (
        'research', 'coding', 'testing',
        'deployment', 'documentation', 'data_analysis'
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;


CREATE TABLE IF NOT EXISTS experiences (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    schema_version      TEXT        NOT NULL DEFAULT '1.0.0',

    -- provenance
    agent_id            TEXT        NOT NULL,
    agent_role          agent_role  NOT NULL,

    -- the experience
    task                TEXT        NOT NULL,
    problem             TEXT,
    action              TEXT        NOT NULL,
    solution            TEXT,
    result              TEXT        NOT NULL,
    tools_used          TEXT[]      NOT NULL DEFAULT '{}',
    technologies        TEXT[]      NOT NULL DEFAULT '{}',
    success             BOOLEAN     NOT NULL,

    -- measurement
    tokens_input        INTEGER     NOT NULL DEFAULT 0 CHECK (tokens_input >= 0),
    tokens_output       INTEGER     NOT NULL DEFAULT 0 CHECK (tokens_output >= 0),
    tool_calls          INTEGER     NOT NULL DEFAULT 0 CHECK (tool_calls >= 0),
    execution_time_ms   INTEGER     NOT NULL DEFAULT 0 CHECK (execution_time_ms >= 0),
    cost_usd            NUMERIC(12, 6) NOT NULL DEFAULT 0 CHECK (cost_usd >= 0),

    -- experiment bookkeeping
    run_id              TEXT        NOT NULL,
    system              system_condition NOT NULL,
    task_key            TEXT,
    retrieved_experience_ids UUID[] NOT NULL DEFAULT '{}',

    -- governance (Memory Judge only)
    tier                memory_tier NOT NULL DEFAULT 'private',
    confidence          REAL        NOT NULL DEFAULT 0.5
                                    CHECK (confidence >= 0 AND confidence <= 1),
    reuse_count         INTEGER     NOT NULL DEFAULT 0 CHECK (reuse_count >= 0),
    reuse_success_count INTEGER     NOT NULL DEFAULT 0 CHECK (reuse_success_count >= 0),

    -- versioning: supersede, never overwrite
    version             INTEGER     NOT NULL DEFAULT 1 CHECK (version >= 1),
    superseded_by       UUID        REFERENCES experiences(id) ON DELETE SET NULL,

    -- Phase 2 (nullable until the workers fill them)
    embedding           vector(1536),
    workflow            JSONB,
    knowledge_triples   JSONB       NOT NULL DEFAULT '[]'::jsonb,
    embedded            BOOLEAN     NOT NULL DEFAULT FALSE,

    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- A reuse cannot succeed more often than it happened.
    CONSTRAINT reuse_success_lte_count CHECK (reuse_success_count <= reuse_count)
);

-- Experiment queries: "all rows for this run, in this arm".
CREATE INDEX IF NOT EXISTS idx_exp_run_system   ON experiences (run_id, system);
-- Redundancy Index: "how many times was this same logical task solved?"
CREATE INDEX IF NOT EXISTS idx_exp_task_key     ON experiences (task_key)
    WHERE task_key IS NOT NULL;
-- Retrieval filters by trust tier on every query, so index it.
CREATE INDEX IF NOT EXISTS idx_exp_tier         ON experiences (tier);
CREATE INDEX IF NOT EXISTS idx_exp_agent        ON experiences (agent_id);
-- Tag lookup ("which experiences touched docker?") needs GIN for array containment.
CREATE INDEX IF NOT EXISTS idx_exp_technologies ON experiences USING GIN (technologies);

-- Phase 1 search is naive full-text over the meaningful prose fields. It is
-- deliberately unsophisticated: a dumb search that returns a wrong row is
-- obviously wrong, whereas a wrong row from a five-term ranking function costs
-- you a week of debugging. Phase 2 replaces this with hybrid retrieval.
CREATE INDEX IF NOT EXISTS idx_exp_fts ON experiences
    USING GIN (to_tsvector('english',
        coalesce(task, '') || ' ' ||
        coalesce(problem, '') || ' ' ||
        coalesce(solution, '') || ' ' ||
        coalesce(result, '')
    ));

-- Only rows that still need embedding — keeps the Phase 2 worker's poll cheap.
CREATE INDEX IF NOT EXISTS idx_exp_not_embedded ON experiences (embedded)
    WHERE embedded = FALSE;


-- Keep updated_at honest without relying on the application to remember.
CREATE OR REPLACE FUNCTION touch_updated_at() RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_experiences_touch ON experiences;
CREATE TRIGGER trg_experiences_touch
    BEFORE UPDATE ON experiences
    FOR EACH ROW EXECUTE FUNCTION touch_updated_at();


-- Append-only audit of every tier change. The proposal promises auditability
-- of "what was shared and by whom"; that is only credible if promotions are
-- recorded separately from the row they mutate.
CREATE TABLE IF NOT EXISTS tier_transitions (
    id            BIGSERIAL PRIMARY KEY,
    experience_id UUID        NOT NULL REFERENCES experiences(id) ON DELETE CASCADE,
    from_tier     memory_tier,
    to_tier       memory_tier NOT NULL,
    reason        TEXT        NOT NULL,
    decided_by    TEXT        NOT NULL DEFAULT 'memory_judge',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_tier_transitions_exp
    ON tier_transitions (experience_id, created_at DESC);
