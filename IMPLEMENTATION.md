# ActionCloud: Implementation and Development History

This document records how ActionCloud was built: the phases, what each one added, what went wrong, and why the system looks the way it does now. For usage, see [README.md](README.md). For the research write-up, see [paper/](paper/).

---

## Timeline

| Phase | Dates (2026) | Commits | Outcome |
| :--- | :--- | :--- | :--- |
| 0. Proposal and literature | Aug | `7842639` | Problem framing, literature notes, Review 1 presentation |
| 1. Plumbing | Aug – Sep | `7842639`, `a808868` | Schema, REST API, SQS worker, client, agents, end-to-end check |
| 2. Memory intelligence | Sep 17 | `a808868` | Extraction, embeddings, Memory Judge, hybrid search, audit trail |
| 3. Integration and first evaluation | Sep 17 – 21 | `337fd8b` … `5396263` | MCP server, LLM providers, 100-task benchmark, K ablation, service refactor |
| 4. Audit and correctness | Sep 28 | `e8dfe31` (PR #1) | Nine confirmed bugs fixed; visibility rules enforced in SQL |
| 5. Evaluation redesign | Sep 28 | `e8dfe31` (PR #1) | Ground-truth verifier, simulated environment, isolated arms, honest metrics |
| 6. Hardening and security | Sep 28 | `a4b0ead`, `74b49e4`, `81f2e89` | Fail-fast startup, agent API keys, injection-bound reuse accounting |
| 7. Documentation | Sep 28 | `9ac13ff` | Rewritten README |
| 8. Research paper and scale | Sep 28 | this change | FGCS manuscript, adversarial and scale experiments, author reputation, index-backed retrieval |

---

## Phase 0: Proposal and literature

**Goal:** establish that fleet-level experience reuse is a real problem and position the project against existing work.

- `ActionCloud_Literature_Notes.md` surveys governed memory, procedural memory, memory poisoning and benchmarks. It flags the closest prior work (Margalit et al., *Governed Shared Memory for Multi-Agent LLM Systems*, 2026).
- `ActionCloud_Review1_Presentation.tex` is the first review deck.
- **Key decision:** treat a memory as untrusted until other agents have reused it successfully. This idea survived every later phase.

## Phase 1: Plumbing

**Goal:** a complete write-and-read path before any intelligence.

| Component | File | Notes |
| :--- | :--- | :--- |
| Schema | `sql/001_init.sql` | `experiences` table with governance columns from day one, plus `tier_transitions` audit table |
| Data model | `schema.py` | Pydantic models; the task / problem / action / solution / result decomposition |
| REST API | `api.py` | FastAPI; returns `202 Accepted` and publishes to SQS |
| Queue | `queue.py` | boto3 SQS; LocalStack locally, with pinned fake credentials so local runs can never touch real AWS |
| Worker | `worker.py` | Long polling; idempotent insert (`ON CONFLICT DO NOTHING`) for at-least-once delivery |
| Client and agents | `client.py`, `agents/` | Typed client; role-specialised agents |
| Checks | `scripts/verify_e2e.py`, `tests/test_schema.py` | Agent → API → queue → worker → Postgres → search |

**Decision worth keeping:** port 5433 instead of 5432, which avoids clashing with a Postgres installed locally.

## Phase 2: Memory intelligence

**Goal:** turn stored transcripts into reusable, governed knowledge.

- **Extractor** (`extractor.py`): an LLM extracts a workflow (prerequisites, steps, pitfalls) and knowledge triples, with a heuristic fallback.
- **Embeddings** (`embeddings.py`): 1536-dimensional vectors, stored in pgvector.
- **Memory Judge** (`judge.py`): a five-tier ladder, `PRIVATE → AGENT → SHARED → VALIDATED → ORGANIZATIONAL`, with promotion and demotion thresholds.
- **Hybrid search:** pgvector cosine similarity combined with Postgres full-text rank.

## Phase 3: Integration and first evaluation

**Goal:** make ActionCloud usable from real tools, and measure it.

- **MCP server** (`337fd8b`): stdio server for Cursor, Claude Desktop and other MCP clients.
- **Multi-provider LLMs** (`84fdd22`): mock, Anthropic, Gemini and Groq; a 100-task benchmark; a metrics engine.
- **K ablation** (`f17ee36`): how many memories to inject per task.
- **Service refactor** (`d9c3205`): a single `MemoryService` behind REST and MCP; `MemorySelectionPolicy` and `CompactContextBuilder` introduced; 12 agent roles.
- **Architecture diagram** (`5396263`).

**What Phase 3 got wrong** (found in Phase 4): the evaluation couldn't measure anything real.
- The mock embeddings were random.
- Every task was marked successful.
- The benchmark crashed at task 31.
- The README's fleet numbers could not be reproduced from the code.

## Phase 4: Audit and correctness

A full code review. Each bug was reproduced before it was fixed.

| # | Bug | Effect | Fix |
| :--- | :--- | :--- | :--- |
| 1 | `agent_role` enum in SQL had 6 roles; Python had 12 | 54 of 100 fleet-simulation experiences silently rejected; the worker retried them forever | 12 roles in `001`, plus migration `002` |
| 2 | "Mock" embeddings came from a SHA-256 hash | Similarity was noise; an unrelated sentence outscored a paraphrase | Lexical feature-hashing embedder, calibrated (94 % top-1) |
| 3 | `relevance or 1.0` | A relevance of 0.0 passed the 0.70 threshold | Missing relevance is treated as 0 |
| 4 | The builder injected the top hit when nothing passed the threshold | Irrelevant memories were injected | Inject nothing |
| 5 | `evaluate()` always returned success | 100 % success everywhere; self-promotion guaranteed | Ground-truth verifier (Phase 5) |
| 6 | Visibility checked `tier > private` | AGENT-tier memories leaked fleet-wide; anonymous callers saw private failures | Visibility rule in SQL: author, same role, then fleet |
| 7 | Reuse counted from any reporter, any number of times | One agent could promote a memory to ORGANIZATIONAL alone | Reporter required; reports counted per injection (Phase 6) |
| 8 | Sync-write path skipped tiering, extraction and embedding | Rows lacked vectors and workflows | One ingest pipeline (`pipeline.py`) for both paths |
| 9 | MCP argument names differed from the README; not JSON-RPC compliant | Clients following the docs failed | Aliases, `ping`, `isError`, notifications, sanitised errors |

**Also in this phase:**
- **Benchmark roles:** roles that don't exist were remapped.
- **`DATABASE_URL`:** now honoured.
- **Worker:** discards messages after a capped number of failed deliveries.
- **Metrics:** redefined so each one measures what its name says. The old "Redundancy Index" was the same for every arm.

## Phase 5: Evaluation redesign

**Goal:** results that are verifiable and reproducible, and that are honest about what they show.

- **Benchmark** (`benchmark/tasks.json`): 136 tasks in 45 families covering all 12 roles.
- **Ground truth** (`benchmark/procedures.json`): 4 steps per family, each with keyword checks. The checks are validated so that no step's keywords are satisfied by the other steps' text.
- **Verifier** (`benchmark/verifier.py`):
  - *strict* (all steps present) is the true outcome;
  - *lenient* (at most one step missing) is what the agent believes, and what it stores.
- **Simulated agent** (`benchmark/simulator.py`, `LLM_PROVIDER=sim`): follows, is misled by, or ignores injected memories according to explicit, documented parameters. It uses common random numbers across arms.
- **Harness** (`benchmark/harness.py`):
  - every arm has its own run ID, and retrieval is scoped to it;
  - identical task order across arms;
  - 5 seeds.
- **Governance refinements found through the evaluation:**
  - confidence is a Beta posterior updated on every counted report, with a 0.45 injection floor;
  - negative self-reports count;
  - outcomes are observed only with a configurable probability.
- **Results:** `results/`, plus `make experiment`.

## Phase 6: Hardening and security

- **Fail-fast startup** (`a4b0ead`): Postgres checks take under a second, and database tests skip cleanly when it's down.
- **Configurable port** (`74b49e4`): `docker-compose.yml` publishes Postgres on `DB_PORT`, and the connection pool closes on exit.
- **Agent identity** (`81f2e89`, migration `003`):
  - per-agent API keys, stored as SHA-256 hashes;
  - identity and role taken from the key;
  - admin-only registration (`make agent`);
  - private records answer `404`.
- **Injection ledger:** `/context` records each injection. A reuse report counts only by consuming an unreported injection of that memory to the reporter, and an author's negative report counts once. This bounds any agent's influence by the injections it receives (paper, Proposition 1).
- **Tests:** `tests/test_auth.py` runs every known attack against the live API.

## Phase 7: Documentation

- The README was rewritten as a tool manual: quickstart with verified commands, API reference, governance model, configuration table, deployment checklist, and evaluation with explicit caveats.

## Phase 8: Research paper and scale

- **Manuscript** (`paper/`): LaTeX for *Future Generation Computer Systems*, with draw.io diagrams, plots generated from data, 29 verified references, and all experiment scripts and raw data.
- **New experiments** (`paper/scripts/paper_experiments.py`):
  - paired McNemar tests;
  - a sensitivity grid over the simulator's assumptions;
  - an adversarial and colluding-agent sweep;
  - latency and scaling measured on real Postgres.
- **Author reputation** (`MEMORY_AUTHOR_PRIOR`, migration `004`): a new memory's prior comes from its author's record. It's off by default: it helps a little under attack but costs about 3 points of success in an honest fleet.
- **Index-backed retrieval** (`MEMORY_ANN_CANDIDATES`): HNSW pre-selection. 17 ms instead of 1.25 s at 50,000 memories, with the same injected memory as exact search in all 300 test queries.

---

## Architecture as built

```
agents / IDEs
   │  X-API-Key                       MCP stdio (bound to one key)
   ▼                                  │
api.py ── auth.py                     mcp_server.py
   └───────────────┬──────────────────┘
                   ▼
              service.py  (MemoryService)
   ┌───────────────┼──────────────────────────┐
   ▼               ▼                          ▼
builder.py      judge.py                  pipeline.py ◄── worker.py ◄── SQS
policy.py       (tiers, confidence)       (tier → extract → embed → 1 txn)
   └───────────────┴────────────┬─────────────┘
                                ▼
                         db.py → PostgreSQL 16 + pgvector
```

### Data model and migrations

| Migration | Adds |
| :--- | :--- |
| `001_init.sql` | `experiences`, `tier_transitions`, enums, indexes, `updated_at` trigger |
| `002_fleet_roles_and_reuse_audit.sql` | 12 roles, `reuse_events`, HNSW index, run index |
| `003_agent_auth_and_injections.sql` | `agents` (hashed keys), `injections` ledger |
| `004_author_prior.sql` | Per-memory `prior` column for author reputation |

Every migration after `001` is idempotent; `make migrate` applies them all.

---

## Design decisions and why

| Decision | Reason |
| :--- | :--- |
| Inject at most one memory (`k = 1`) | At k = 5, precision falls from 95 % to 57 % with no gain in success |
| Inject nothing below the relevance threshold | Irrelevant context costs tokens and misleads; always injecting the top hit costs 2.5 points of success |
| Visibility enforced inside SQL | Filtering after retrieval leaked data (bug 6) |
| Identity from the key, never from the request | Self-asserted identity broke every governance rule |
| Count reports only against injections | Prevents report flooding and reporting on memories never received |
| Negative self-reports count, positive ones don't | Admitting failure is evidence; vouching for yourself isn't |
| One ingest pipeline for queue and sync paths | Two paths had drifted apart (bug 8) |
| Exact retrieval by default, index-backed opt-in | Exact is deterministic for evaluation; index-backed is needed beyond a few thousand memories |
| Author reputation off by default | It costs about 3 points of success when all agents are honest |

---

## Testing

| Suite | What it covers |
| :--- | :--- |
| `test_schema.py` | Data model, tier ordering, validation |
| `test_service.py`, `test_phase2.py`, `test_phase3.py` | Policy, builder, Judge decisions, embedder, extractor, providers |
| `test_integration.py` | Real Postgres: all 12 roles, visibility, run isolation, idempotency, injection accounting, author prior, index-backed mode |
| `test_auth.py` | Each attack against the live REST API, admin endpoints, MCP binding |
| `test_evaluation.py` | Dataset integrity, verifier, simulator, metric definitions, MCP protocol, worker poison handling |

105 tests in total (`make test`). Database tests skip in under a second when Postgres is down.

---

## Current status

**Works and is tested:** the full service (REST, MCP, queue, sync), governance, authentication, both retrieval modes, and the evaluation harness.

**Established only in simulation:** the task-success and token-saving figures. The simulator's benefit from a correct memory is an assumption, and the paper reports how far each conclusion depends on it.

**Known limits:**
- **No real-LLM evaluation yet.** This is the next step.
- **Semantic embeddings untested.** The providers are implemented but not evaluated.
- **Collusion:** with about 30 % colluding agents, governance loses its advantage over having no memory.
- **Keys:** there is no key expiry or rate limiting.
- **Packaging:** no Dockerfile, CI pipeline or license yet.

## Roadmap

1. Real-LLM run: `LLM_PROVIDER=gemini|anthropic`, with rate limiting and resumable runs.
2. Semantic embeddings, then recalibrate the threshold.
3. Reporter-credibility weighting to resist collusion.
4. Dockerfile, CI pipeline (GitHub Actions with a Postgres service) and licence.
