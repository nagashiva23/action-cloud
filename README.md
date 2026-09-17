# ActionCloud

Cloud-native distributed experience memory service for multi-agent LLM systems.

ActionCloud enables autonomous agent fleets to share procedural knowledge and eliminate redundant execution while enforcing strict governance to prevent false or unproven memories from corrupting shared fleet state.

---

## Tech Stack & Tools

| Layer | Tools & Technologies | Description |
| :--- | :--- | :--- |
| **API & Core** | Python 3.11+, FastAPI, Uvicorn, Pydantic v2 | High-performance asynchronous REST API and data validation |
| **Storage & Search** | PostgreSQL 16, pgvector, Full-Text Search (FTS) | Relational datastore with dense 1536-dim vector similarity and keyword ranking |
| **Messaging** | Amazon SQS, LocalStack | Decoupled asynchronous queue processing (LocalStack for offline dev) |
| **LLM & Embeddings** | Anthropic Claude API, Google Gemini API, boto3 | Model providers and official AWS SDK integration |
| **DevOps & Hosting** | Docker, Docker Compose, AWS App Runner, AWS ECS Fargate, AWS RDS | Container orchestration, managed cloud web services, and serverless compute |
| **Testing** | Pytest, HTTPX | Unit testing and HTTP client end-to-end verification |

---

## Architecture Overview

```
                              ┌──────────────────────────────────┐
                              │           Agent Fleet            │
                              │ (System A: Baseline / System B)  │
                              └────────┬─────────────────▲───────┘
                                       │                 │
                           POST /experiences (202)  GET /search (Sync)
                                       │                 │
                                       ▼                 │
                              ┌──────────────────┐       │
                              │  FastAPI Server  │───────┤
                              └────────┬─────────┘       │
                                       │                 │
                                 Publish SQS             │
                                       │                 │
                                       ▼                 │
                             ┌──────────────────┐        │
                             │  SQS Queue       │        │
                             │  (LocalStack)    │        │
                             └────────┬─────────┘        │
                                       │                 │
                                  Long-Poll              │
                                       │                 │
                                       ▼                 │
                             ┌──────────────────┐        │
                             │  Queue Worker    │        │
                             │ (LLM Extractor + │        │
                             │ Vector Embedder) │        │
                             └────────┬─────────┘        │
                                       │                 │
                                 Idempotent Insert       │
                                       │                 │
                                       ▼                 │
                             ┌──────────────────┐        │
                             │ PostgreSQL       │────────┘
                             │ (pgvector + FTS) │
                             └──────────────────┘
```

### Core Design Principles

1. **Non-Blocking Asynchronous Writes (`HTTP 202`)**: Memory submission is decoupled from agent execution via SQS message queues and background workers. Agents write memories without incurring task latency overhead.
2. **Synchronous Hybrid Search**: Agents execute fast, governed reads combining `pgvector` 1536-dimensional dense vector similarity with PostgreSQL full-text keyword ranking (`ts_rank`).
3. **Governed Trust Management**: Memories enter untrusted and climb a 5-tier governance ladder based on observed reuse success.
4. **Equalized A/B Benchmark Harness**: System A (Stateless Baseline) and System B (ActionCloud Memory) share identical agent execution paths (`Agent(use_memory=False)` vs `Agent(use_memory=True)`), ensuring rigorous experimental comparisons.

---

## Memory Governance Engine

ActionCloud enforces a 5-tier governance model managed by the `MemoryJudge` engine:

$$\text{PRIVATE} \longrightarrow \text{AGENT} \longrightarrow \text{SHARED} \longrightarrow \text{VALIDATED} \longrightarrow \text{ORGANIZATIONAL}$$

* **PRIVATE**: Visible only to the run that created it. Failed task executions remain private.
* **AGENT**: Visible to the creating agent across runs. Initial successful tasks land here.
* **SHARED**: Visible fleet-wide once promoted through a verified successful reuse.
* **VALIDATED**: Proven by $\ge 3$ recorded reuses with $\ge 80\%$ success rate.
* **ORGANIZATIONAL**: Canonical fleet knowledge ($\ge 10$ reuses with $\ge 90\%$ success rate).
* **Automatic Demotion**: Memories with low reuse success ($< 40\%$ across $\ge 3$ reuses) are automatically demoted to `PRIVATE`.
* **Audit Lineage**: Every tier transition is logged immutably in the `tier_transitions` audit table.

---

## Intelligence & Hybrid Retrieval

* **LLM Extraction Worker (`extractor.py`)**: Asynchronously parses execution logs into procedural workflows (JSON step sequences) and subject-predicate-object knowledge triples.
* **Dense Vector Embeddings (`embeddings.py`)**: Generates 1536-dimensional unit-normalized vector embeddings for PostgreSQL `pgvector` storage.
* **Hybrid Search Engine (`db.py`)**: Fuses full-text keyword search and vector similarity:
  $$\text{Relevance} = 0.5 \times \text{FTS\_Rank} + 0.5 \times \text{Vector\_Similarity}$$
* **Reuse Feedback Endpoint (`api.py`)**: `POST /experiences/{id}/reuse` records agent reuse outcomes and triggers real-time tier promotion or demotion.

---

## Agent Fleet Roles

Implements all 6 proposal agent roles in `ROLE_REGISTRY` (`roles.py`):
- `CodingAgent` (`AgentRole.CODING`)
- `ResearchAgent` (`AgentRole.RESEARCH`)
- `TestingAgent` (`AgentRole.TESTING`)
- `DeploymentAgent` (`AgentRole.DEPLOYMENT`)
- `DocumentationAgent` (`AgentRole.DOCUMENTATION`)
- `DataAnalysisAgent` (`AgentRole.DATA_ANALYSIS`)

---

## System Metrics & Analytics Engine

Accessible via `GET /metrics`:
- **Knowledge Reuse Rate (KRR %)**
- **Redundancy Index (RI)**
- **Cumulative Token Savings (%)**
- **Financial Cost Savings (% USD)**
- **Task Execution Latency (mean ms)**
- **Governance Tier Distribution**

---

## Quickstart & Verification

### Prerequisites

- Docker Desktop
- Python 3.11+

### Installation & Execution

```bash
# 1. Install virtual environment and dependencies
make setup

# 2. Start PostgreSQL (pgvector) and LocalStack SQS
make up

# 3. Start API Server (Terminal 1)
make api

# 4. Start Queue Worker (Terminal 2)
make worker
```

### Running Tests (Terminal 3)

```bash
# Run unit test suite (25 tests)
make test

# Run Phase 1 pipeline verification (19 checks)
./.venv/bin/python scripts/verify_e2e.py

# Run Phase 2 governance & hybrid retrieval verification (16 checks)
./.venv/bin/python scripts/verify_phase2_e2e.py
```

### Inspect Database State

```bash
make psql
```
```sql
SELECT agent_role, task, tier, confidence FROM experiences ORDER BY created_at DESC LIMIT 5;
SELECT experience_id, from_tier, to_tier, reason FROM tier_transitions;
\q
```

---

## API Reference Summary

- `GET /health` : Dependency liveness and readiness probe.
- `POST /experiences` : Submit completed experience (Returns HTTP 202).
- `GET /search` : Perform hybrid vector + full-text search.
- `POST /experiences/{id}/reuse` : Report reuse outcome and trigger governance evaluation.
- `GET /experiences/{id}` : Retrieve full experience record.
- `GET /stats` : Basic row counts.
- `GET /metrics` : Aggregate evaluation metrics and system analytics.

---

## Roadmap — Phase 3

- [ ] Connect production LLM providers (`AnthropicLLM` / `GeminiLLM`) via environment variable `LLM_PROVIDER`.
- [ ] Run 100-task multi-agent benchmark workload across System A vs System B.
- [ ] Deploy production infrastructure to AWS (RDS PostgreSQL `pgvector`, AWS SQS, AWS App Runner / ECS).
