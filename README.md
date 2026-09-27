# ActionCloud

Governed adaptive experience memory for AI agent fleets.

Agents store what they did on past tasks; ActionCloud retrieves the relevant experience for a new task, filters it for relevance, redundancy and trust, and injects a compact, token-budgeted procedure into the agent's prompt. A governance layer (the **Memory Judge**) learns from reuse outcomes which memories actually work, promoting those and holding back those that don't — so one agent's subtly wrong solution doesn't spread through the fleet.

Design principle: **maximise useful memory per prompt token, not retrieved context size.** The evaluation below supports this: injecting one well-chosen memory beats injecting five.

---

## Architecture

```mermaid
graph TD
    subgraph Clients
        MCP[MCP clients<br/>Claude Desktop / Cursor / IDEs]
        REST[REST clients / agents]
    end

    subgraph Service
        MS[MemoryService]
        POL[MemorySelectionPolicy<br/>candidate_k=10 · k_inject=1<br/>threshold=0.30 · min_confidence=0.45]
        CCB[CompactContextBuilder<br/>relevance → trust rank → redundancy → token budget]
        MJ[MemoryJudge<br/>initial tier · confidence posterior · promote / quarantine]
    end

    subgraph Ingest
        SQS[SQS / LocalStack]
        W[Worker]
        P[Ingest pipeline<br/>tier → extract workflow → embed → one transaction]
    end

    DB[(PostgreSQL 16 + pgvector<br/>experiences · tier_transitions · reuse_events)]

    MCP -->|stdio JSON-RPC| MS
    REST -->|HTTP| MS
    MS -->|queued write| SQS --> W --> P
    MS -->|SYNC_WRITE=true| P
    P --> DB
    MS -->|governed hybrid search| DB
    MS --> POL --> CCB
    MS -->|reuse outcome, row-locked| MJ --> DB
```

| Layer | Components |
| :--- | :--- |
| API | FastAPI (`api.py`), MCP stdio server (`mcp_server.py`), both over one `MemoryService` |
| Ingest | `pipeline.py` — shared by the SQS worker and the synchronous path, idempotent per experience id |
| Retrieval | Hybrid pgvector cosine + Postgres full-text score, visibility rules enforced in SQL, optional run scoping |
| Context | `builder.py` — threshold, trust-aware ranking, redundancy filter, token budget, compact procedure format |
| Governance | `judge.py` (pure decision function) applied inside a `SELECT … FOR UPDATE` transaction (`db.apply_reuse`) |
| Embeddings | `hashing` (default: offline lexical feature hashing), `gemini`, `openai` |
| Evaluation | `benchmark/` — 136 tasks, ground-truth procedures, verifier, simulated agent environment, harness |

---

## Governance

### Tiers and visibility

| Tier | Who can retrieve it | How a memory gets here |
| :--- | :--- | :--- |
| `PRIVATE` | Only its author | Self-assessed failure, or quarantined |
| `AGENT` | Author + agents of the **same role** | Self-assessed success (entry point) |
| `SHARED` | Every agent | ≥ 1 successful reuse by *another* agent |
| `VALIDATED` | Every agent | ≥ 3 counted reuses at ≥ 80 % success |
| `ORGANIZATIONAL` | Every agent | ≥ 10 counted reuses at ≥ 90 % success |

A caller with no identity sees only `SHARED` and above. Knowledge therefore spreads **within a role first** and **across roles only after it has proven itself**.

### Outcome feedback

After using injected memories, an agent reports whether its task actually succeeded (`reuse_memory` / `POST /experiences/{id}/reuse`). An author can also report that its *own* memory turned out wrong.

- **Reports count only against a real injection.** A report counts if `get_memory_context` actually put that memory in front of the reporting agent, and each injection earns at most one counted report. Reporting on a memory you were never shown, or reporting ten times, earns nothing.
- **Positive self-reports are ignored.** An agent cannot vouch for its own memory.
- **Negative self-reports count, once.** "My solution failed CI" is exactly the evidence needed.
- **Confidence is updated on every counted report** as a Beta-posterior mean, `(successes + 2·0.6) / (reports + 2)`. Memories below `min_confidence = 0.45` are not injected. One failed report takes a new memory from 0.60 to 0.40.
- **Quarantine.** When a memory has ≥ 3 reports at < 40 % success, it is demoted to `PRIVATE`. Quarantine is terminal for automatic governance.
- **Everything is audited.** Every tier change goes to `tier_transitions`, and every report to `reuse_events`, with who reported it and whether it counted.

In practice, **confidence gating does most of the work**: a bad memory usually stops being injected after one failed report, long before it has the three reports needed for formal quarantine.

---

### Identity

Every agent is registered with a fixed role and gets an API key; only a SHA-256 hash of the key is stored.

- **The REST API takes `agent_id` and role from the key.** A request naming a different agent gets `403`. So an agent cannot read another agent's `PRIVATE` memories, or report outcomes under a second identity to get around the self-report rule.
- **Only an admin can register agents** (`ACTIONCLOUD_ADMIN_KEY` or the CLI), so identities can't be minted at will.
- **The MCP server acts as the agent owning `ACTIONCLOUD_API_KEY`.** It holds database credentials itself, so for MCP this keeps honest clients to one identity; the REST API is the actual security boundary.
- **The in-process experiment client is trusted** and not authenticated.

`tests/test_auth.py` runs each of the old attacks against the live API.

---

## Evaluation

### Setup

- **Workload:** 136 tasks in 45 task families (paraphrases of the same problem) across all 12 roles, run for 3 epochs = 408 tasks per arm. Each family has a ground-truth procedure of 4 verifiable steps (`benchmark/procedures.json`).
- **Grading:** the same `TaskVerifier` grades every arm:
  - *strict* (all steps present) is the reported outcome;
  - *lenient* (≤ 1 step missing) is what the agent itself believes and stores. Some stored "successes" are therefore wrong, which is the realistic failure mode governance must handle.
- **Isolation:** every arm has its own `run_id`, and retrieval is scoped to it. Every arm sees the same task order and the same per-task randomness. Results are mean ± sd over 5 seeds.
- **Arms:**
  - `baseline`: no memory.
  - `flat_memory`: everything shared immediately, no feedback.
  - `actioncloud`: full governance, with 70 % of task outcomes observed.

### How to read these numbers

These runs use `LLM_PROVIDER=sim`, a **simulated agent environment** (`benchmark/simulator.py`), not a real LLM. The simulator's assumptions are explicit parameters:

- With no memory, it solves a task 65 % of the time.
- Following a correct memory succeeds 97 % of the time and uses far fewer output tokens.
- An incomplete memory misleads it 85 % of the time.

So **the size of the benefit of a correct memory is an assumption, not a finding.**

What the simulation *does* measure is everything ActionCloud controls:
- whether retrieval surfaces a memory from the right task family;
- how many prompt tokens the context costs;
- how often flawed memories get injected;
- how those combine under the stated assumptions.

To replace the assumptions with a real model, run the same harness with `LLM_PROVIDER=anthropic|gemini|groq`. The verifier and metrics are unchanged.

### Main result (`results/main.md`)

| Arm | Success % | Retrieval precision % | Misled % | Poisoned injections % | Redundancy index | Tokens / task | $ / success |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 65.0 ± 2.9 | — | — | — | 1.000 | 928 | 0.0198 |
| flat_memory | 75.4 ± 4.9 | 95.2 ± 0.3 | 22.5 ± 5.1 | 27.7 ± 5.3 | 0.017 | 608 | 0.0089 |
| **actioncloud** | **86.8 ± 1.1** | 95.0 ± 1.1 | **7.1 ± 1.2** | **4.4 ± 1.0** | 0.029 | **591** | **0.0075** |

Against baseline, ActionCloud saves 36.4 ± 0.4 % of tokens and 49.4 ± 0.5 % of cost, with +21.8 ± 2.3 pp success. Flat memory gets similar token savings but only +10.5 ± 4.3 pp. The difference is governance: about 28 % of flat memory's same-family injections were flawed memories, versus 4.4 % with ActionCloud.

### Ablations

| Question | Finding | File |
| :--- | :--- | :--- |
| How many memories should be injected? | **One.** Going from k=1 to k=5 drops precision from 95 % to 57 % and raises context from 107 to 162 tokens per task, with no success gain. | `results/k.md` |
| Does the relevance threshold matter? | Yes, moderately. Always injecting the top hit (threshold 0) gives 80 % precision and −2.5 pp success versus threshold 0.30. | `results/threshold.md` |
| How much does governance depend on feedback? | A lot. With 30 / 70 / 100 % of outcomes observed, poisoned injections are 12.3 / 4.4 / 0.0 % and success is 81.5 / 86.8 / 89.8 %. Without observable outcomes, governance can't beat flat memory. | `results/feedback.md` |
| Fleet with one agent per task, 12 roles | All 12 roles end up with memories promoted to SHARED or above. Cross-role reuse is rare (9 of 272 counted reports), because each benchmark task family belongs to one role. | `results/fleet.txt` |

### Retrieval quality (`results/calibration.txt`)

With the default offline `hashing` embedder and the query built as task + technologies, the nearest stored neighbour is from the same task family for **94.1 %** of the 136 tasks. At the 0.30 threshold, recall on same-family pairs is 90.6 % and the false-positive rate on different-family pairs is 1.0 %.

Caveat: tasks in the same family share technology tags, which helps a lexical embedder. Paraphrases with no shared vocabulary need `EMBEDDING_PROVIDER=gemini` or `openai`. Re-run `make calibrate` after switching provider.

### Metric definitions (`metrics.py`)

- **Retrieval precision:** injected memories from the same task family ÷ injected memories.
- **Misled rate:** tasks that received a same-family memory and still failed ÷ tasks that received one.
- **Poisoned injections:** same-family injections of memories whose own task had actually failed.
- **Redundancy index:** of the tasks whose family was *already solved* earlier in the run, the fraction solved again without a same-family memory. The baseline is 1.0 by construction.
- **Injection rate** (formerly "Knowledge Reuse Rate"): tasks that received any memory ÷ tasks.

---

## Quickstart

```bash
make setup                 # venv + deps, copies .env.example -> .env
make up                    # Postgres 16 + pgvector (port 5433) and LocalStack SQS
make migrate               # only for a volume created before sql/003 existed
make test                  # 101 tests (DB-backed ones skip if Postgres is down)
make agent ID=cursor-1 ROLE=coding   # register an agent; prints its API key once

make experiment            # all evaluation presets, 5 seeds -> results/
make fleet                 # 12-role fleet simulation
```

Run the service (queued production path):

```bash
make api       # terminal 1
make worker    # terminal 2
make verify    # terminal 3: end-to-end check
```

Or skip the worker and LocalStack entirely with `SYNC_WRITE=true`: writes go through the same pipeline, in-request.

---

## MCP integration

```json
{
  "mcpServers": {
    "actioncloud": {
      "command": "/path/to/actioncloud/.venv/bin/python",
      "args": ["-m", "actioncloud.mcp_server"],
      "env": {
        "PYTHONPATH": "/path/to/actioncloud/src",
        "DATABASE_URL": "postgresql://actioncloud:actioncloud@localhost:5433/actioncloud",
        "SYNC_WRITE": "true",
        "ACTIONCLOUD_API_KEY": "ac_... (from make agent)"
      }
    }
  }
}
```

| Tool | Arguments | Purpose |
| :--- | :--- | :--- |
| `get_memory_context` | `task`, `agent_id`, `agent_role`, `technologies`, `k_inject`, `token_budget` | Compact memory block to prepend to your prompt, plus the ids to report on |
| `search_memory` | `query`, `agent_id`, `agent_role`, `limit`, `min_tier` | Governed hybrid search |
| `remember_experience` | `task`, `action_taken`, `result`, `success`, `agent_id`, `agent_role`, `problem`, `solution`, `technologies` | Store an experience (numbered steps extract best) |
| `reuse_memory` | `experience_id`, `success`, `agent_id` | Report whether a memory helped; drives governance |
| `get_memory_metrics` | `run_id` | Aggregate metrics |

With `ACTIONCLOUD_API_KEY` set, `agent_id` and `agent_role` come from the key and can be omitted; naming a different agent is an error.

Reuse credit requires that the memory came from `get_memory_context`; memories found through `search_memory` alone earn no credit.

Older argument names (`query`, `role`, `max_memories`, `action`) and tool names (`search_fleet_memory`, `store_experience`, `report_memory_reuse`, `get_fleet_metrics`) are still accepted. Errors are returned as `isError` results without internal details.

## REST API

Every endpoint except `/health` requires an agent key: `X-API-Key: ac_...` (or `Authorization: Bearer ac_...`). `agent_id` and role come from the key, so they can be left out of request bodies.

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/health` | GET | Database and queue status (queue shown as not required when `SYNC_WRITE=true`) |
| `/experiences` | POST | Store an experience (`202` when queued) |
| `/search` | GET | Governed hybrid search (`q`, `agent_id`, `agent_role`, `technologies`, `scope_run_id`, `min_tier`, `limit`) |
| `/context` | POST | Compact, governed context block for a task |
| `/experiences/{id}/reuse` | POST | Outcome report `{success}`. Counted only against a prior injection to this agent. |
| `/experiences/{id}` | GET | Full record, if visible to this agent (otherwise `404`) |
| `/agents` | POST | Admin: register an agent `{agent_id, agent_role}`; returns its key once |
| `/agents/{id}` | DELETE | Admin: revoke an agent's key |
| `/metrics` | GET | Metrics computed from stored experiences. Note: stored `success` is each agent's self-assessment, so it runs higher than verified success. |

---

## Configuration

See `.env.example`. The main settings:

- `EMBEDDING_PROVIDER`: `hashing` (default), `gemini` or `openai`.
- `LLM_PROVIDER`: `sim`, `mock`, `anthropic`, `gemini` or `groq`.
- `SYNC_WRITE`: process writes in-request instead of through the queue.
- `DATABASE_URL`: a full connection URL, overriding the `DB_*` fields.
- `AUTH_REQUIRED` (default `true`), `ACTIONCLOUD_ADMIN_KEY`, `ACTIONCLOUD_API_KEY`: see *Identity*.
- Retrieval policy: the `MEMORY_*` variables.

## Known limitations

- **Simulated results.** The headline numbers come from a simulated environment. A real-LLM run with the included harness is the next step before claiming effect sizes.
- **Lexical default embedder.** Semantic providers are wired in but were not evaluated here, because no API keys or network were available.
- **Brute-force hybrid scoring.** The query scores every visible row, which is fine at thousands of experiences. At larger scale, restrict candidates via the HNSW index first.
- **Almost no cross-role knowledge transfer is exercised**, because the benchmark's task families are role-specific.
- **No rate limiting or key expiry.** A leaked key works until an admin revokes it.
- **No `LICENSE` file is included yet.**
