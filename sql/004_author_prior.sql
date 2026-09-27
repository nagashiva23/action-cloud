-- ActionCloud — migration 004: per-memory prior for author reputation
--
-- Each memory stores the prior mean its confidence is computed from:
--     confidence = (s + 2 * prior) / (n + 2)
-- With MEMORY_AUTHOR_PRIOR=true, prior = the author's reputation at write
-- time (Beta posterior over all counted reports on the author's memories),
-- so an author whose memories keep failing starts every new memory below the
-- injection floor. With the flag off (default) prior = 0.6 as before.
ALTER TABLE experiences ADD COLUMN IF NOT EXISTS prior REAL NOT NULL DEFAULT 0.6
    CHECK (prior >= 0 AND prior <= 1);
CREATE INDEX IF NOT EXISTS idx_exp_author ON experiences (agent_id) INCLUDE (reuse_count, reuse_success_count);
