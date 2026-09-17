# ActionCloud — Governed Distributed Experience Memory for Multi-Agent Systems

ActionCloud is a cloud-native distributed experience memory service designed for multi-agent LLM systems. It enables teams of autonomous agents to share procedural knowledge and avoid repeating trial-and-error mistakes, while enforcing strict governance to prevent false or unproven memories from corrupting the fleet.

> **Status: Phase 1 & Phase 2 Complete & Fully Verified.**
> - **Unit Tests**: 25/25 passing (`make test`).
> - **Phase 1 Pipeline E2E**: 19/19 checks passing (`python scripts/verify_e2e.py`).
> - **Phase 2 Governance & Retrieval E2E**: 16/16 checks passing (`python scripts/verify_phase2_e2e.py`).

---

## 🏗 Architecture & Key Principles

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

1. **Asynchronous Writes (HTTP 202)**: Memory creation is decoupled from task execution via SQS queues and background workers. Agents write memories instantly without incurring task latency penalties.
2. **Synchronous Hybrid Retrieval**: Agents perform fast, governed reads combining `pgvector` 1536-dimensional dense vector similarity with PostgreSQL full-text keyword ranking (`ts_rank`).
3. **Strict Memory Governance**: Memories are untrusted by default and must earn fleet-wide trust through observed successful reuse.
4. **Unified A/B Benchmark Harness**: System A (Stateless Baseline) and System B (ActionCloud Memory) share identical agent code paths (`Agent(use_memory=False)` vs `Agent(use_memory=True)`), ensuring clean experimental evaluation.

---

## 🛡 Memory Governance & 5-Tier Ladder

ActionCloud implements a **5-tier governance ladder** managed by the `MemoryJudge` engine:

$$\text{PRIVATE} \longrightarrow \text{AGENT} \longrightarrow \text{SHARED} \longrightarrow \text{VALIDATED} \longrightarrow \text{ORGANIZATIONAL}$$

* **`PRIVATE`**: Visible only to the run that created it. Failed tasks remain private permanently.
* **`AGENT`**: Visible to the same agent across repeated runs. Initial successful tasks land here.
* **`SHARED`**: Visible fleet-wide once an experience earns promotion through a verified successful reuse.
* **`VALIDATED`**: Proven by $\ge 3$ recorded reuses with $\ge 80\%$ success rate.
* **`ORGANIZATIONAL`**: Canonical knowledge ($\ge 10$ reuses with $\ge 90\%$ success rate).
* **Automatic Demotion**: If a memory leads to repeated failures during reuse (success rate $< 40\%$), the Memory Judge automatically demotes it back to `PRIVATE`.
* **Audit Trail**: All tier changes are appended to the `tier_transitions` audit table.

---

## 🧠 Intelligence & Hybrid Retrieval

* **LLM Extraction Worker (`extractor.py`)**: Parses raw episode logs out-of-band into:
  * **Generalized Procedural Workflows**: JSON objects with `prerequisites`, ordered `steps`, and `pitfalls`.
  * **Knowledge Triples**: Subject-Predicate-Object semantic relationships.
* **Dense Vector Embeddings (`embeddings.py`)**: Generates 1536-dimensional unit-normalized vector embeddings for PostgreSQL `pgvector` storage.
* **Hybrid Search (`db.py`)**: Merges keyword ranking with vector similarity:
  $$\text{Relevance} = 0.5 \times \text{FTS\_Rank} + 0.5 \times \text{Vector\_Similarity}$$
* **Reuse Feedback API**: `POST /experiences/{id}/reuse` records agent reuse outcomes and triggers real-time tier promotion/demotion.

---

## 👥 Agent Fleet Roles

Implements all 6 proposal agent roles in `ROLE_REGISTRY` ([roles.py](file:///Users/nagashiva/Desktop/ASAI/S5/PROJECTS/CLOUD/actioncloud/src/actioncloud/agents/roles.py)):
1. `CodingAgent` (`AgentRole.CODING`)
2. `ResearchAgent` (`AgentRole.RESEARCH`)
3. `TestingAgent` (`AgentRole.TESTING`)
4. `DeploymentAgent` (`AgentRole.DEPLOYMENT`)
5. `DocumentationAgent` (`AgentRole.DOCUMENTATION`)
6. `DataAnalysisAgent` (`AgentRole.DATA_ANALYSIS`)

---

## 📊 Metric Evaluation Engine (`metrics.py`)

Provides a built-in metric calculator accessible via **`GET /metrics`**:
* **Knowledge Reuse Rate (KRR %)**
* **Redundancy Index (RI)**
* **Cumulative Token Savings %**
* **Financial Cost Savings % ($ USD)**
* **Task Execution Latency (mean ms)**
* **Governance Tier Distribution**

---

## ⚡ Quickstart & Runbook

### 1. Prerequisites
- Docker Desktop (running)
- Python 3.11+

### 2. Setup Environment
```bash
make setup
```
*(Creates `.venv`, installs dependencies from `requirements.txt`, and copies `.env.example` to `.env`)*

### 3. Start Infrastructure
```bash
make up
```
*(Boots Postgres 16 `pgvector` container on port `5433` and LocalStack SQS on port `4566`)*

### 4. Launch API Server (Terminal 1)
```bash
make api
```
*(Runs FastAPI server on http://localhost:8000. Interactive Swagger UI at http://localhost:8000/docs)*

### 5. Launch Queue Worker (Terminal 2)
```bash
make worker
```
*(Long-polls SQS, extracts workflows & triples, generates embeddings, and persists rows)*

### 6. Run Test Suites & Verification (Terminal 3)
```bash
# Run unit tests
make test

# Run Phase 1 pipeline verification
./.venv/bin/python scripts/verify_e2e.py

# Run Phase 2 governance & hybrid retrieval verification
./.venv/bin/python scripts/verify_phase2_e2e.py
```

### 7. Inspect Live Database
```bash
make psql
```
```sql
SELECT agent_role, task, tier, confidence FROM experiences ORDER BY created_at DESC LIMIT 5;
SELECT experience_id, from_tier, to_tier, reason FROM tier_transitions;
\q
```

---

## 🗺 Roadmap — Phase 3

- [ ] **Real Model Provider**: Wire up `AnthropicLLM` (Claude Sonnet 3.5/4.5) / `GeminiLLM` via environment variable `LLM_PROVIDER`.
- [ ] **100-Task Benchmark Harness**: Execute the standardized multi-agent benchmark workload across System A vs System B.
- [ ] **AWS Cloud Deployment**: Deploy RDS PostgreSQL (`pgvector`), AWS SQS, and API/Worker containers on AWS App Runner / ECS.
- [ ] **Final Empirical Paper & Charts**: Extract metric analytics via `GET /metrics`.
