#!/usr/bin/env python3
"""
Phase 1 end-to-end verification.

This script is the Phase 1 exit criterion made executable. It proves the full
asynchronous loop:

    agent -> API -> queue -> worker -> Postgres -> search -> agent

Run it with the stack up and the worker running:

    python scripts/verify_e2e.py

It exits non-zero on failure, so it works in CI later.
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
from actioncloud.schema import AgentRole, Experience, SystemCondition  # noqa: E402

BASE_URL = "http://localhost:8000"
# Deliberately distinctive so full-text search can't match it by accident.
PROBE_TASK = "Diagnose why the Neo4j bolt connection refuses inside Docker Compose"

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
    run_id = f"verify-{uuid.uuid4().hex[:8]}"
    print(f"\nActionCloud Phase 1 verification  (run_id={run_id})\n")

    # -- 1. Dependencies ---------------------------------------------------
    print("1. Health")
    client = ActionCloudClient(
        agent_id="verify-agent", agent_role=AgentRole.CODING, base_url=BASE_URL
    )
    try:
        health = client.health()
    except Exception as e:  # noqa: BLE001
        check("API reachable", False, f"{e}\n        Is uvicorn running on :8000?")
        return 1

    check("API reachable", True)
    check("database connected", health.get("database") is True,
          "Is Postgres up? `docker compose ps`")
    check("queue reachable", health.get("queue") is True,
          "Is LocalStack up and healthy?")
    if failed:
        return 1

    # -- 2. Write path -----------------------------------------------------
    print("\n2. Write path (agent -> API -> queue -> worker -> Postgres)")
    agent = build_agent(
        AgentRole.CODING,
        agent_id="coding-verify",
        client=client,
        run_id=run_id,
        use_memory=False,          # baseline arm: writes but never reads
        llm=MockLLM(simulated_latency_ms=0),
    )
    outcome = agent.run_task(TaskSpec(task=PROBE_TASK, task_key="verify-probe",
                                      technologies=["neo4j", "docker"]))
    check("agent completed task", outcome.success)
    check("API accepted experience", outcome.experience_id is not None)

    # The worker is asynchronous, so the row does not exist yet. Polling here
    # is not a workaround — it is the correct way to observe an async pipeline,
    # and if this poll times out it means the worker isn't consuming.
    print(f"\n   waiting for worker to persist {outcome.experience_id}...")
    persisted = None
    for attempt in range(20):
        try:
            persisted = client.get(outcome.experience_id)
            break
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    check(
        "worker persisted the experience",
        persisted is not None,
        "Row never appeared after 10s. Is the worker running? "
        "`python -m actioncloud.worker`",
    )
    if persisted is None:
        return 1

    # -- 3. Stored correctly ----------------------------------------------
    print("\n3. Stored record")
    check("task text round-tripped", persisted["task"] == PROBE_TASK)
    check("tagged with run_id", persisted["run_id"] == run_id)
    # Provenance must follow the *agent*, not the shared client. Getting this
    # wrong misattributes every experience and makes the audit trail useless.
    check("attributed to the right agent", persisted["agent_id"] == "coding-verify",
          f"got agent_id={persisted['agent_id']!r}, expected 'coding-verify'")
    check("attributed to the right role", persisted["agent_role"] == "coding",
          f"got agent_role={persisted['agent_role']!r}")
    check("tagged as baseline arm", persisted["system"] == SystemCondition.BASELINE.value)
    # Phase 2 Memory Judge assigns initial successful experiences to 'agent'.
    # Promotion to 'shared' is earned through observed reuse.
    check("Memory Judge assigned initial tier", persisted["tier"] in ("agent", "shared"),
          f"got tier={persisted['tier']!r}, expected 'agent' from the Phase 2 judge")
    check("technologies normalised", set(persisted["technologies"]) == {"neo4j", "docker"})
    check("token counts recorded", persisted["tokens_input"] > 0 and persisted["tokens_output"] > 0)

    # -- 4. Read path ------------------------------------------------------
    print("\n4. Read path (search)")
    results = client.search(query="Neo4j bolt connection Docker", limit=5)
    check("search returned results", len(results) > 0,
          "Full-text search found nothing. Check the GIN index in 001_init.sql.")
    found = any(r.id == outcome.experience_id for r in results)
    check("our experience is retrievable", found)

    # -- 5. Memory actually reaches the second agent -----------------------
    print("\n5. Memory-enabled arm")
    agent_b = build_agent(
        AgentRole.RESEARCH,
        agent_id="research-verify",
        client=client,
        run_id=run_id,
        use_memory=True,           # actioncloud arm
        llm=MockLLM(simulated_latency_ms=0),
    )
    outcome_b = agent_b.run_task(
        TaskSpec(task="Fix Neo4j bolt connection failing in Docker", task_key="verify-probe")
    )
    check(
        "second agent retrieved prior experience",
        outcome_b.retrieved_count > 0,
        "Cross-agent retrieval returned nothing — this is the core mechanism, "
        "so investigate before moving on.",
    )
    check("retrieval was recorded on the new experience", len(outcome_b.retrieved_ids) > 0)

    # Wait for agent_b's experience to land in Postgres before checking idempotency
    for attempt in range(20):
        try:
            if client.get(outcome_b.experience_id) is not None:
                break
        except Exception:
            time.sleep(0.3)

    # -- 6. Idempotency ----------------------------------------------------
    # At-least-once delivery means duplicate messages are expected, not
    # exceptional. If this fails, every metric in Phase 3 is inflated by
    # however many redeliveries happened to occur.
    print("\n6. Idempotency under redelivery")
    before = client._http.get("/stats", params={"run_id": run_id}).json()["experiences"]
    raw = db.get_experience(outcome.experience_id)
    if raw is not None:
        replay = Experience(
            id=raw["id"], agent_id=raw["agent_id"], agent_role=raw["agent_role"],
            task=raw["task"], action=raw["action"], result=raw["result"],
            success=raw["success"], run_id=raw["run_id"], system=raw["system"],
        )
        db.insert_experience(replay)  # simulate SQS redelivering the same event
        after = client._http.get("/stats", params={"run_id": run_id}).json()["experiences"]
        check("duplicate insert did not create a second row", before == after,
              f"count went {before} -> {after}; ON CONFLICT DO NOTHING isn't working")
    else:
        check("duplicate insert did not create a second row", False, "could not re-read row")

    client.close()
    db.close_pool()  # otherwise psycopg's background threads complain on exit

    print(f"\n{'-' * 60}")
    if failed:
        print(f"{RED}{failed} check(s) failed{RESET}, {passed} passed\n")
        return 1
    print(f"{GREEN}All {passed} checks passed.{RESET} Phase 1 loop is working.\n")
    print(f"{YELLOW}Next:{RESET} deploy this same code to AWS (RDS + SQS), "
          f"then start Phase 2.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
