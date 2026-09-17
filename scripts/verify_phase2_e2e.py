#!/usr/bin/env python3
"""
Phase 2 end-to-end verification script.

Proves all Phase 2 features:
  1. Asynchronous LLM workflow & knowledge triple extraction.
  2. 1536-dim vector embedding generation (pgvector).
  3. MemoryJudge governance tier assignment (initial AGENT tier).
  4. Hybrid vector + full-text search.
  5. Reuse reporting (POST /experiences/{id}/reuse) and tier promotion to SHARED.
  6. Audit trail logging in tier_transitions.
"""

from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import actioncloud.db as db  # noqa: E402
from actioncloud.agents import TaskSpec, build_agent  # noqa: E402
from actioncloud.client import ActionCloudClient  # noqa: E402
from actioncloud.llm import MockLLM  # noqa: E402
from actioncloud.schema import AgentRole  # noqa: E402

BASE_URL = "http://localhost:8000"
PROBE_TASK = "Deploy FastAPIs on AWS App Runner with Postgres pgvector"

GREEN, RED, YELLOW, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"
passed = failed = 0


def check(label: str, ok: bool, detail: str = "") -> bool:
    global passed, failed
    if ok:
        passed += 1
        print(f"  {GREEN}PASS{RESET}  {label}")
    else:
        failed += 1
        print(f"  {RED}FAIL{RESET}  {label}" + (f"\n        {detail}" if detail else ""))
    return ok


def main() -> int:
    run_id = f"phase2-{uuid.uuid4().hex[:8]}"
    print(f"\nActionCloud Phase 2 Verification  (run_id={run_id})\n")

    client = ActionCloudClient(
        agent_id="deployment-verify", agent_role=AgentRole.DEPLOYMENT, base_url=BASE_URL
    )

    # 1. Submit Experience
    print("1. Submitting experience via DeploymentAgent...")
    agent = build_agent(
        AgentRole.DEPLOYMENT,
        agent_id="deployment-agent-01",
        client=client,
        run_id=run_id,
        use_memory=False,
        llm=MockLLM(simulated_latency_ms=0),
    )
    outcome = agent.run_task(
        TaskSpec(
            task=PROBE_TASK,
            task_key="phase2-probe",
            technologies=["fastapi", "aws", "pgvector"],
            expected_tools=["docker", "aws_cli"],
        )
    )
    check("Agent executed task", outcome.success)
    check("API accepted experience", outcome.experience_id is not None)

    # 2. Wait for worker enrichment
    print("\n2. Waiting for worker to perform LLM extraction & embedding generation...")
    persisted = None
    for _ in range(20):
        try:
            persisted = client.get(outcome.experience_id)
            if persisted and persisted.get("workflow"):
                break
        except Exception:
            pass
        time.sleep(0.5)

    check("Worker persisted record", persisted is not None)
    if not persisted:
        return 1

    check("MemoryJudge assigned initial AGENT tier", persisted["tier"] == "agent")
    check("LLM Extractor generated procedural workflow", persisted.get("workflow") is not None)
    check(
        "Knowledge triples extracted",
        isinstance(persisted.get("knowledge_triples"), list) and len(persisted["knowledge_triples"]) > 0,
    )
    check("Embedding processed", persisted.get("embedded") is True)

    # 3. Hybrid Search Retrieval
    print("\n3. Testing Hybrid Vector + Full-Text Search Retrieval...")
    results = client.search(query="FastAPI AWS App Runner pgvector", limit=5)
    check("Hybrid search returned results", len(results) > 0)
    found_probe = any(r.id == outcome.experience_id for r in results)
    check("Probe experience retrieved via hybrid search", found_probe)
    if found_probe:
        reusable_res = next(r for r in results if r.id == outcome.experience_id)
        check("Retrieved result includes extracted workflow", reusable_res.workflow is not None)

    # 4. Reuse Feedback & Governance Promotion
    print("\n4. Testing Reuse Feedback & Tier Promotion (AGENT -> SHARED)...")
    reuse_resp = client.report_reuse(outcome.experience_id, success=True)
    check("Reuse endpoint acknowledged feedback", reuse_resp.get("reuse_count") == 1)
    check("MemoryJudge promoted experience to SHARED tier", reuse_resp.get("tier") == "shared")
    check("Transition recorded", reuse_resp.get("transition") == "shared")

    # 5. Governance Audit Trail Inspection
    print("\n5. Inspecting tier transition audit trail in DB...")
    with db.get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM tier_transitions WHERE experience_id = %s ORDER BY created_at ASC",
            (outcome.experience_id,),
        )
        transitions = cur.fetchall()

    check("Audit trail has transition records", len(transitions) >= 2)
    if len(transitions) >= 2:
        check("Initial creation logged in audit trail", transitions[0]["to_tier"] == "agent")
        check("Reuse promotion to shared logged in audit trail", transitions[1]["to_tier"] == "shared")

    client.close()
    db.close_pool()

    print(f"\n{'-' * 60}")
    if failed:
        print(f"{RED}{failed} check(s) failed{RESET}, {passed} passed\n")
        return 1
    print(f"{GREEN}All {passed} checks passed.{RESET} Phase 2 development verified successfully!\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
