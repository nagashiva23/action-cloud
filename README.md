# ActionCloud

Cloud-Native Governed Adaptive Experience Memory Platform for Autonomous Agent Fleets.

ActionCloud is a production-ready, governed agent memory tool that captures, extracts, embeds, stores, retrieves, and governs procedural knowledge across heterogeneous AI agent fleets. Built around a core design principle—**maximizing useful memory density per prompt token rather than maximizing retrieved context size**—ActionCloud uses adaptive context selection, redundancy filtering, compact procedural formatting, and automated multi-tier governance to eliminate redundant agent execution without bloating context windows or corrupting fleet memory state.

---

## Technical Architecture

```mermaid
graph TD
    subgraph Clients
        MCP[MCP Clients<br/>Cursor / Antigravity / Claude Desktop]
        REST[REST API Clients<br/>FastAPI / HTTP Agents]
    end

    subgraph Service Layer
        MS[MemoryService Singleton]
        MSP[MemorySelectionPolicy<br/>K_search=10, K_inject=2, Redundancy=0.85]
        CCB[CompactContextBuilder<br/>Token Budgeting & Markdown Formatting]
        MJ[MemoryJudge Governance Engine<br/>5-Tier Ladder & Promotion/Demotion]
    end

    subgraph Infrastructure Layer
        SQS[AWS SQS / LocalStack Queue]
        Worker[Background Worker<br/>LLM Extractor & Unit-Normalized Vector Embedder]
        DB[(PostgreSQL 16<br/>pgvector 1536-dim + Full-Text Search)]
    end

    MCP -->|Stdio Protocol| MS
    REST -->|HTTP REST| MS
    MS -->|Query & Context Req| MSP
    MSP -->|Filter & Budget| CCB
    MS -->|Ingest Async| SQS
    SQS --> Worker
    Worker -->|Idempotent Write| DB
    MS -->|Governed Read| DB
    MS -->|Report Reuse| MJ
    MJ -->|Update Tier & Transition| DB
```

---

## Tech Stack & Core Dependencies

| Layer | Tools & Technologies | Description |
| :--- | :--- | :--- |
| **API & Core** | Python 3.11+, FastAPI, Uvicorn, Pydantic v2 | High-performance asynchronous REST API, context endpoints, and data validation |
| **Service & Policy Layer** | `MemoryService`, `MemorySelectionPolicy`, `CompactContextBuilder` | Unified internal abstraction for adaptive selection, token budgeting, and redundancy filtering |
| **IDE & MCP Server** | Model Context Protocol (MCP Stdio), `mcp_server.py` | Native IDE and desktop assistant integration for Cursor, Antigravity, and Claude Desktop |
| **Storage & Search** | PostgreSQL 16, pgvector, Full-Text Search (FTS) | Relational datastore with 1536-dim dense vector similarity and keyword ranking (`ts_rank`) |
| **Messaging & Workers** | Amazon SQS, LocalStack, Background Async Worker | Decoupled non-blocking ingestion queue and procedural workflow extraction |
| **LLM & Embeddings** | Anthropic Claude API, Google Gemini API, boto3 | Multi-provider LLM support for knowledge extraction and embedding generation |
| **DevOps & Testing** | Docker, Docker Compose, Pytest | Container orchestration and comprehensive unit/integration test suite |

---

## Core Architectural Pillars

### 1. Adaptive Context Selection & Token Budgeting

Instead of injecting arbitrary numbers of memories into an agent prompt, ActionCloud uses a dual-parameter selection policy:
* **Candidate Retrieval Window ($K_{\text{search}} = 10$)**: Fetches top candidate memories via hybrid vector similarity and keyword search.
* **Context Injection Window ($K_{\text{inject}} = 1\dots 2$)**: Limits injected memories to only the most relevant, non-redundant items.
* **Redundancy Filtering ($\text{Similarity} < 0.85$)**: Filters out near-duplicate procedural memories using sequence similarity evaluation.
* **Strict Token Budgeting ($\le 1000\text{ tokens}$)**: Truncates context blocks to fit strict token boundaries, measuring token counts with exact or heuristic estimation.
* **Compact Procedural Formatting**: Formats retrieved experiences into structured, token-efficient Markdown containing explicit Prerequisites, Executed Steps, and Known Pitfalls.

### 2. Governed 5-Tier Memory Ladder

Memories enter the system unvalidated and climb or fall along a strict trust hierarchy managed by `MemoryJudge`:

$$\text{PRIVATE} \longrightarrow \text{AGENT} \longrightarrow \text{SHARED} \longrightarrow \text{VALIDATED} \longrightarrow \text{ORGANIZATIONAL}$$

* **PRIVATE**: Visible only to the run that created it. Failed execution attempts remain private to prevent fleet-wide failure propagation.
* **AGENT**: Visible to the creating agent across runs. Initial successful executions land here.
* **SHARED**: Promoted fleet-wide upon verified successful reuse.
* **VALIDATED**: Proven by $\ge 3$ recorded reuses with $\ge 80\%$ success rate.
* **ORGANIZATIONAL**: Canonical fleet knowledge ($\ge 10$ reuses with $\ge 90\%$ success rate).
* **Automatic Demotion**: Memories with reuse success falling below $40\%$ across $\ge 3$ reuses are automatically demoted back to `PRIVATE`.
* **Immutable Audit Lineage**: Every promotion and demotion event is recorded in the `tier_transitions` audit table.

---

## Model Context Protocol (MCP) Integration

ActionCloud provides a native **MCP Stdio Server** (`actioncloud.mcp_server`), exposing memory capabilities directly to AI IDEs (Cursor, Antigravity, Windsurf, VS Code) and Claude Desktop.

### Configuration

Add ActionCloud to your `.cursor/mcp.json` or `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "actioncloud": {
      "command": "python",
      "args": ["-m", "actioncloud.mcp_server"],
      "env": {
        "PYTHONPATH": "src",
        "DATABASE_URL": "postgresql://actioncloud:actioncloud@localhost:5433/actioncloud"
      }
    }
  }
}
```

### Exposed MCP Tools

| Tool Name | Parameters | Description |
| :--- | :--- | :--- |
| `get_memory_context` | `task` (str), `agent_role` (str, opt), `k_inject` (int, opt) | Formats a compact, token-budgeted memory context block for direct prompt injection. |
| `search_memory` | `query` (str), `agent_role` (str, opt), `limit` (int, opt) | Performs hybrid vector + full-text search across ActionCloud fleet memory. |
| `remember_experience` | `task` (str), `action_taken` (str), `result` (str), `success` (bool), `agent_id` (str), `agent_role` (str) | Submits a completed agent execution experience asynchronously for indexing and governance. |
| `reuse_memory` | `experience_id` (str), `success` (bool), `agent_id` (str) | Reports experience reuse outcome, triggering real-time tier promotion or demotion. |
| `get_memory_metrics` | None | Returns aggregate governance analytics, knowledge reuse rate, and token savings. |

*Note: Backward-compatible aliases (`search_fleet_memory`, `store_experience`, `report_memory_reuse`, `get_fleet_metrics`) are supported.*

---

## Multi-Role Agent Fleet Support

ActionCloud supports a 12-role heterogeneous agent fleet across diverse engineering domains:

1. `CODING` - Application & algorithmic software development
2. `RESEARCH` - Literature, API documentation & pattern analysis
3. `TESTING` - Unit, integration & automated test suite execution
4. `DEPLOYMENT` - CI/CD pipeline automation & release execution
5. `DOCUMENTATION` - Technical writing & architecture documentation
6. `DATA_ANALYSIS` - Data pipeline execution & statistical reporting
7. `SECURITY` - Vulnerability scanning & security policy auditing
8. `DEVOPS` - Infrastructure configuration & environment management
9. `DATABASE` - Schema migrations & query performance optimization
10. `ML` - Machine learning model training & inference pipelines
11. `CLOUD` - Cloud service provisioning & IAM configuration
12. `MONITORING` - Observability, alerting & log analysis

---

## System Analytics & Metrics Engine

Accessible via `GET /metrics` or `get_memory_metrics`:
* **Knowledge Reuse Rate (KRR %)**: Percentage of task executions utilizing existing fleet memory.
* **Redundancy Index (RI)**: Ratio of redundant candidate memories filtered out prior to injection.
* **Cumulative Token Savings (%)**: Token overhead avoided by replacing raw execution histories with compact procedural context.
* **Financial Cost Savings (% USD)**: Calculated LLM API cost savings based on token reduction.
* **Governance Tier Distribution**: Real-time breakdown of experiences across `PRIVATE`, `AGENT`, `SHARED`, `VALIDATED`, and `ORGANIZATIONAL` tiers.

---

## Heterogeneous Fleet Simulation & Verification

ActionCloud includes a 100-agent fleet simulation script (`scripts/run_fleet_simulation.py`) that demonstrates memory propagation, cross-role knowledge sharing, and governance promotion.

```bash
# Execute 100-agent simulation across 10 task iterations
PYTHONPATH=src ./.venv/bin/python scripts/run_fleet_simulation.py --agents 100
```

### Simulated Results Summary (100 Agents / 1000 Tasks)
* **Agents Active**: 100 agents across 12 Fleet Roles
* **Total Tasks Executed**: 1,000 tasks
* **Failures Injected & Recovered**: 10 initial failures isolated to `PRIVATE` tier
* **Memories Injected**: 1,000 relevant context blocks
* **Redundant Memories Filtered**: 1,000 candidate blocks suppressed by policy
* **Cross-Role Knowledge Reuses**: 1,000 recorded governance feedback events
* **Final Governance Distribution**: 9 `AGENT`, 17 `SHARED`, 20 `VALIDATED`

---

## Quickstart & Local Setup

### Prerequisites
* Docker Desktop
* Python 3.11+

### 1. Installation & Environment Setup
```bash
# Create virtual environment and install dependencies
make setup
```

### 2. Start Infrastructure (PostgreSQL pgvector & LocalStack SQS)
```bash
make up
```

### 3. Run Test Suite
```bash
# Execute all 40 unit and integration tests
PYTHONPATH=src ./.venv/bin/pytest tests/
```

### 4. Start Server Infrastructure (Optional for REST API)
```bash
# Terminal 1: Start REST API daemon
make api

# Terminal 2: Start Queue Worker
make worker
```

---

## REST API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/health` | `GET` | Health check for API, PostgreSQL database, and SQS queue |
| `/experiences` | `POST` | Asynchronously ingest an agent experience (Returns `202 Accepted`) |
| `/search` | `GET` | Perform hybrid vector + full-text search over governed experiences |
| `/context` | `POST` | Generate compact, token-budgeted memory context block for task prompt |
| `/experiences/{id}/reuse` | `POST` | Record reuse outcome and evaluate governance tier promotion/demotion |
| `/experiences/{id}` | `GET` | Retrieve full experience details by ID |
| `/metrics` | `GET` | Aggregate governance analytics, token savings, and system metrics |

---

## License

Apache 2.0 License. See `LICENSE` for details.
