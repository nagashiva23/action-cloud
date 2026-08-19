# ActionCloud — Phase 1

Cloud-native distributed experience memory for multi-agent systems.

**Phase 1 scope: plumbing only.** No LLM extraction, no knowledge graph, no
embeddings, no ranking. Search is Postgres full-text. This is deliberate — a
dumb search that returns a wrong row is obviously wrong, whereas a wrong row
from a five-term ranking function costs you a week of debugging.

> **Status: written but never run.** This code was authored without a Docker
> environment available, so the loop below is unverified. Expect to fix things
> on first run. That is normal and the failures are listed at the bottom.

---

## Prerequisites

- Docker Desktop (running)
- Python 3.11+
- ~1 GB free disk for container images

No AWS account and no API key needed for Phase 1. The LLM is mocked and SQS is
emulated by LocalStack.

---

## Runbook

### 1. Install

```bash
cd actioncloud
make setup
```

Creates `.venv`, installs dependencies, copies `.env.example` to `.env`.

### 2. Start infrastructure

```bash
make up
```

Expect both containers `healthy` in `docker compose ps`. LocalStack takes
15-30s on first run — if it shows `starting`, wait and re-check.

Confirm the schema loaded:

```bash
make psql
\dt
```

You should see `experiences` and `tier_transitions`. Then `\q`.

### 3. Run the API — terminal 1

```bash
make api
```

Visit http://localhost:8000/health — expect:

```json
{"status":"healthy","database":true,"queue":true,"mode":"local"}
```

`http://localhost:8000/docs` gives you an interactive UI for every endpoint.
This is your main manual testing tool.

### 4. Run the worker — terminal 2

```bash
make worker
```

Expect `worker starting (queue=http://localhost:4566/...)` then silence. Silence
is correct — it is long-polling an empty queue.

### 5. Verify — terminal 3

```bash
make verify
```

This is the Phase 1 exit criterion. It runs an agent, watches the experience
travel through the queue into Postgres, retrieves it, then confirms a *second*
agent can retrieve the first agent's experience — the core mechanism of the
whole project.

Also run the unit tests (no infrastructure needed):

```bash
make test
```

---

## What exists

```
src/actioncloud/
  schema.py      Experience models — the keystone; everything depends on this
  config.py      env-driven settings (same code, LocalStack or real AWS)
  db.py          raw SQL via psycopg; no ORM
  queue.py       SQS wrapper (works against LocalStack unchanged)
  api.py         Agent Memory API — the only door into the system
  worker.py      queue consumer; splits into 3 workers in Phase 2
  client.py      agent-facing library
  llm.py         MockLLM (default) + AnthropicLLM (Phase 3)
  agents/        base agent + coding/research roles
sql/001_init.sql schema, indexes, audit table
scripts/         end-to-end verification
tests/           schema unit tests
```

## Two design decisions worth understanding

**Writes are asynchronous; reads are synchronous.** `POST /experiences` returns
202 and queues the work. In Phase 2 the write path runs an LLM extraction call —
seconds of work. If an agent had to wait for that, using ActionCloud would make
every agent *slower*, and hypothesis H3 would be defeated by the architecture
rather than by the idea. Reads must be fast because the agent is blocked on them.

**Baseline and ActionCloud are one class with a flag.** `Agent(use_memory=False)`
is System A; `use_memory=True` is System B. They share prompt construction,
measurement, and storage. If they were separate implementations, any measured
difference could be an artifact of one being written more carefully — and a
sceptical evaluator would be right to say so.

---

## Known first-run failures

**`FATAL: role "actioncloud" does not exist`, but `make psql` works** — the
giveaway pair. `make psql` runs *inside* the container, so it proves the
container is fine; the app connects over the host port and is reaching a
*different* Postgres installed natively on your machine. That is why the
container publishes on host port **5433**. Check what holds a port with:

```bash
lsof -iTCP:5432 -sTCP:LISTEN -n -P
```

**`port is already allocated`** — something else is on 5433 or 4566. Stop it, or
change the host-side port in `docker-compose.yml` and `.env` (change both).

**Schema changes to `sql/001_init.sql` have no effect** — the init script runs
*only on first boot of a fresh volume*. Use `make reset` (which does `down -v`).
This will confuse you at least once; it confuses everyone once.

**`queue: false` in /health** — LocalStack not ready yet. Wait 20s.
`docker compose logs localstack` if it persists.

**Worker logs nothing** — that is correct when idle. Verify by running
`make verify` in a third terminal and watching the worker print `stored ...`.

**`ModuleNotFoundError: actioncloud`** — run via the Makefile targets, which set
`PYTHONPATH=src`. If running manually, prefix with `PYTHONPATH=src`.

**`NoCredentialsError: Unable to locate credentials`** — `.env` isn't being
loaded, so `AWS_ENDPOINT_URL` is unset and boto3 is trying to reach *real* AWS.
Check `.env` exists (`cp .env.example .env`). The worker now prints
`mode=local (localstack)` on startup; if it says `aws (real)` locally, that's
this problem.

**Row never appears after `store()`** — the worker isn't consuming. Check
terminal 2 is still running and `/health` reports `queue: true`.

---

## Phase 1 remaining

- [ ] Bring the stack up and get `make verify` passing
- [ ] Deploy to AWS: RDS Postgres (enable `pgvector`), SQS queue, API on
      ECS/App Runner. Only env vars change — delete `AWS_ENDPOINT_URL` and the
      same code talks to real AWS.
- [ ] Set a billing alert **before** creating any AWS resource

Then Phase 2: LLM extraction, Neo4j graph, embeddings, hybrid retrieval, and the
Memory Judge.
