"""Verifier, simulator, metrics definitions, MCP protocol, worker, config."""

from __future__ import annotations

import json

import pytest

from actioncloud.benchmark import load_procedures, load_tasks
from actioncloud.benchmark.simulator import SimulatedAgentLLM, parse_context
from actioncloud.benchmark.verifier import TaskVerifier
from actioncloud.builder import CompactContextBuilder
from actioncloud.metrics import compute_metrics
from actioncloud.policy import MemorySelectionPolicy
from actioncloud.schema import AgentRole


# --- dataset --------------------------------------------------------------

def test_every_task_has_a_valid_role_and_procedure():
    procs = load_procedures()
    for t in load_tasks():
        AgentRole(t["role"])                     # used to crash on 'architecture'
        assert t["task_key"] in procs
    assert {t["role"] for t in load_tasks()} == {r.value for r in AgentRole}


def test_no_step_check_is_satisfied_by_other_steps():
    """Otherwise the verifier could not detect a missing step."""
    for key, steps in load_procedures().items():
        for i, s in enumerate(steps):
            others = " ".join(o["step"] for j, o in enumerate(steps) if j != i).lower()
            assert not any(kw in others for kw in s["any"]), (key, i)


# --- verifier -------------------------------------------------------------

def test_verifier_strict_and_lenient():
    v = TaskVerifier()
    key = "task-docker-pgvector-setup"
    full = "\n".join(f"STEP: {s}" for s in v.steps(key))
    assert v.verify(key, full).strict
    one_missing = "\n".join(f"STEP: {s}" for s in v.steps(key)[1:])
    verdict = v.verify(key, one_missing)
    assert not verdict.strict and verdict.lenient
    assert not v.verify(key, "STEP: nothing useful").lenient


# --- simulator ------------------------------------------------------------

def _prompt_with_memory(key: str, steps: list[str], task_idx: int = 1) -> tuple[str, str]:
    tasks = [t for t in load_tasks() if t["task_key"] == key]
    mem = {"id": "00000000-0000-0000-0000-000000000001", "task": tasks[0]["task"],
           "success": True, "tier": "agent", "relevance": 0.9, "confidence": 0.6,
           "workflow": {"steps": steps}}
    ctx, *_ = CompactContextBuilder.build_context([mem], MemorySelectionPolicy())
    return f"{ctx}\n\n## Your task\n{tasks[task_idx]['task']}", key


def test_parse_context_roundtrip():
    key = "task-docker-pgvector-setup"
    steps = TaskVerifier().steps(key)
    prompt, _ = _prompt_with_memory(key, steps)
    mems, task = parse_context(prompt)
    assert len(mems) == 1 and mems[0].steps == steps
    assert task == [t for t in load_tasks() if t["task_key"] == key][1]["task"]


def test_sim_follows_correct_memory_cheaply():
    key = "task-docker-pgvector-setup"
    v = TaskVerifier()
    llm = SimulatedAgentLLM(seed=1)
    followed = scratch = 0
    for i in range(40):
        llm.set_instance(f"t{i}")
        prompt, _ = _prompt_with_memory(key, v.steps(key))
        r = llm.complete(prompt)
        followed += llm.last_mode == "followed_correct" and v.verify(key, r.text).strict
        llm.set_instance(f"t{i}")
        bare = llm.complete("## Your task\n" + prompt.split("## Your task\n")[1])
        scratch += bare.tokens_output
        assert r.tokens_output < bare.tokens_output
    assert followed >= 36


def test_sim_is_misled_by_incomplete_memory():
    key = "task-docker-pgvector-setup"
    v = TaskVerifier()
    llm = SimulatedAgentLLM(seed=1)
    misled = 0
    for i in range(40):
        llm.set_instance(f"m{i}")
        prompt, _ = _prompt_with_memory(key, v.steps(key)[:-1])
        misled += not v.verify(key, llm.complete(prompt).text).strict
    assert misled >= 28


# --- metrics --------------------------------------------------------------

def _o(key, ok, inj=()):
    return {"task_key": key, "success": ok, "tokens_input": 10, "tokens_output": 10,
            "execution_time_ms": 1, "cost_usd": 0.0, "injected_task_keys": list(inj)}


def test_redundancy_index_distinguishes_arms():
    baseline = [_o("a", True), _o("a", True), _o("a", True)]
    memory = [_o("a", True), _o("a", True, ["a"]), _o("a", True, ["a"])]
    assert compute_metrics(baseline)["redundancy_index"] == 1.0
    assert compute_metrics(memory)["redundancy_index"] == 0.0


def test_precision_and_effective_reuse():
    m = compute_metrics([_o("a", True), _o("a", True, ["a"]), _o("b", False, ["a"])])
    assert m["injection_rate_pct"] == pytest.approx(66.67)
    assert m["retrieval_precision_pct"] == 50.0
    assert m["effective_reuse_rate_pct"] == pytest.approx(33.33)


def test_empty_metrics_do_not_divide_by_zero():
    assert compute_metrics([])["success_rate_pct"] == 0.0


# --- MCP --------------------------------------------------------------------

class FakeService:
    def __init__(self):
        self.calls = []

    def prepare_context(self, **kw):
        self.calls.append(("context", kw))
        return {"context": "", "injected_experience_ids": [], "candidate_count": 0,
                "final_count": 0, "context_tokens": 0}

    def store_experience(self, payload):
        self.calls.append(("store", payload))
        return {"id": "x", "queued": False}

    def record_reuse(self, **kw):
        raise RuntimeError("secret internal detail")


def test_mcp_accepts_readme_argument_names():
    from actioncloud.mcp_server import ActionCloudMCPServer

    svc = FakeService()
    s = ActionCloudMCPServer(service=svc)
    _, err = s.call_tool("get_memory_context", {"task": "t", "agent_role": "coding", "k_inject": 2})
    assert not err and svc.calls[0][1]["policy"].max_context_memories == 2
    _, err = s.call_tool("remember_experience", {"task": "t", "action_taken": "a", "result": "r",
                                                 "success": True, "agent_id": "me"})
    assert not err and svc.calls[1][1].action == "a"


def test_mcp_errors_are_flagged_and_sanitised():
    from actioncloud.mcp_server import ActionCloudMCPServer

    s = ActionCloudMCPServer(service=FakeService())
    text, err = s.call_tool("reuse_memory", {"experience_id": "00000000-0000-0000-0000-000000000001",
                                             "success": True, "agent_id": "a"})
    assert err and "secret" not in text
    text, err = s.call_tool("get_memory_context", {})
    assert err and "task" in text


def test_mcp_jsonrpc_protocol():
    from actioncloud.mcp_server import ActionCloudMCPServer

    s = ActionCloudMCPServer(service=FakeService())
    assert s.handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert s.handle_message({"jsonrpc": "2.0", "id": 1, "method": "ping"})["result"] == {}
    assert s.handle_message({"jsonrpc": "2.0", "id": 2, "method": "nope"})["error"]["code"] == -32601
    init = s.handle_message({"jsonrpc": "2.0", "id": 3, "method": "initialize", "params": {}})
    assert "tools" in init["result"]["capabilities"]
    call = s.handle_message({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                             "params": {"name": "bogus", "arguments": {}}})
    assert call["result"]["isError"] is True


# --- worker & config --------------------------------------------------------

def test_worker_discards_invalid_payload():
    from actioncloud import worker

    msg = {"Body": json.dumps({"type": "experience.created", "payload": {"agent_role": "nope"}})}
    assert worker.process_one(msg) is True


def test_worker_gives_up_after_max_receives(monkeypatch):
    from actioncloud import worker

    def boom(_):
        raise RuntimeError("db down")

    monkeypatch.setitem(worker.HANDLERS, "experience.created", boom)
    body = json.dumps({"type": "experience.created", "payload": {}})
    assert worker.process_one({"Body": body, "Attributes": {"ApproximateReceiveCount": "1"}}) is False
    assert worker.process_one({"Body": body, "Attributes": {"ApproximateReceiveCount": "5"}}) is True


def test_database_url_overrides_fields(monkeypatch):
    from actioncloud.config import Settings

    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h:1/d")
    assert Settings().dsn == "postgresql://u:p@h:1/d"
