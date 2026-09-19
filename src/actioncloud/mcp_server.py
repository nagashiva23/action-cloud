from __future__ import annotations

import json
import logging
import sys
import uuid
from typing import Any, Dict, List, Optional
import httpx

from .policy import MemorySelectionPolicy
from .schema import AgentRole, ExperienceCreate, MemoryTier, SystemCondition
from .service import memory_service

logging.basicConfig(level=logging.ERROR, stream=sys.stderr)
log = logging.getLogger(__name__)

DEFAULT_API_URL = "http://localhost:8000"


class ActionCloudMCPServer:
    """
    MCP Stdio Server wrapper for ActionCloud.
    Directly uses MemoryService for governed memory operations.
    """

    def __init__(self, api_url: str = DEFAULT_API_URL) -> None:
        self.api_url = api_url.rstrip("/")
        self.http = httpx.Client(base_url=self.api_url, timeout=10.0)

    def get_tools_manifest(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "get_memory_context",
                "description": (
                    "Retrieve governed, token-budgeted adaptive context for a task. "
                    "Applies hybrid vector+FTS search, relevance filtering, redundancy elimination, "
                    "and compact procedural formatting."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Task or problem description"},
                        "agent_id": {"type": "string", "description": "Identity of requesting agent"},
                        "role": {"type": "string", "description": "Role of requesting agent"},
                        "max_memories": {"type": "integer", "default": 1, "description": "Max memories to inject"},
                        "token_budget": {"type": "integer", "default": 1000, "description": "Max tokens ceiling"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "search_memory",
                "description": (
                    "Search ActionCloud shared fleet memory for prior agent experiences, "
                    "proven solutions, and extracted procedural workflows."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Technical task, problem, or error"},
                        "agent_id": {"type": "string", "description": "Identity of requesting agent"},
                        "limit": {"type": "integer", "default": 5, "description": "Max candidates"},
                        "min_tier": {"type": "string", "default": "private", "description": "Min trust tier"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "remember_experience",
                "description": (
                    "Submit a completed task experience into ActionCloud shared fleet memory (HTTP 202 async queued)."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string", "description": "Agent identity"},
                        "agent_role": {"type": "string", "description": "Agent role"},
                        "task": {"type": "string", "description": "Task objective"},
                        "action": {"type": "string", "description": "Action executed"},
                        "result": {"type": "string", "description": "Outcome achieved"},
                        "success": {"type": "boolean", "description": "Whether task succeeded"},
                        "problem": {"type": "string", "description": "Obstacle hit"},
                        "solution": {"type": "string", "description": "Resolution"},
                        "technologies": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["task", "action", "result", "success"],
                },
            },
            {
                "name": "reuse_memory",
                "description": (
                    "Report outcome after reusing a prior experience to trigger MemoryJudge governance promotion."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "experience_id": {"type": "string", "description": "UUID of reused experience"},
                        "success": {"type": "boolean", "description": "Whether reuse was successful"},
                        "agent_id": {"type": "string", "description": "Agent reporting reuse"},
                    },
                    "required": ["experience_id", "success"],
                },
            },
            {
                "name": "get_memory_metrics",
                "description": (
                    "Retrieve system-wide token savings, Knowledge Reuse Rate, and governance tier distribution."
                ),
                "inputSchema": {"type": "object", "properties": {}},
            },
            # --- Backward Compatibility Tool Aliases ---
            {
                "name": "search_fleet_memory",
                "description": "Alias for search_memory.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "store_experience",
                "description": "Alias for remember_experience.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string"},
                        "action": {"type": "string"},
                        "result": {"type": "string"},
                        "success": {"type": "boolean"},
                        "problem": {"type": "string"},
                        "solution": {"type": "string"},
                    },
                    "required": ["task", "action", "result", "success"],
                },
            },
            {
                "name": "report_memory_reuse",
                "description": "Alias for reuse_memory.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "experience_id": {"type": "string"},
                        "success": {"type": "boolean"},
                    },
                    "required": ["experience_id", "success"],
                },
            },
            {
                "name": "get_fleet_metrics",
                "description": "Alias for get_memory_metrics.",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]

    def handle_call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        try:
            if name in ("get_memory_context",):
                q = arguments["query"]
                agent_id = arguments.get("agent_id")
                role_str = arguments.get("role")
                role = AgentRole(role_str) if role_str else None
                max_memories = arguments.get("max_memories", 1)
                token_budget = arguments.get("token_budget", 1000)

                pol = MemorySelectionPolicy.from_env()
                pol.max_context_memories = max_memories
                pol.context_token_budget = token_budget

                res = memory_service.prepare_context(query=q, agent_id=agent_id, role=role, policy=pol)
                if not res["context"]:
                    return "No relevant governed prior experience found."

                return (
                    f"{res['context']}\n\n"
                    f"--- ActionCloud Metadata ---\n"
                    f"Candidates evaluated: {res['candidate_count']} | "
                    f"Memories injected: {res['final_count']} | "
                    f"Context tokens: {res['context_tokens']}"
                )

            elif name in ("search_memory", "search_fleet_memory"):
                q = arguments["query"]
                limit = arguments.get("limit", 5)
                agent_id = arguments.get("agent_id")
                min_tier_str = arguments.get("min_tier", "private")
                min_tier = MemoryTier(min_tier_str)

                results = memory_service.search_memory(
                    query=q, limit=limit, min_tier=min_tier, agent_id=agent_id
                )
                if not results:
                    return "No matching fleet experiences found."

                lines = ["## ActionCloud Fleet Memory Results\n"]
                for i, r in enumerate(results, 1):
                    lines.append(f"### {i}. [{r.tier.value.upper()}] {r.task}")
                    if r.problem:
                        lines.append(f"- Problem: {r.problem}")
                    if r.solution:
                        lines.append(f"- Solution: {r.solution}")
                    lines.append(f"- Result: {r.result}")
                    if r.workflow:
                        lines.append(f"- Reusable Workflow: {json.dumps(r.workflow)}")
                    lines.append("")
                return "\n".join(lines)

            elif name in ("remember_experience", "store_experience"):
                agent_id = arguments.get("agent_id", "mcp-agent")
                role_str = arguments.get("agent_role", "coding")
                role = AgentRole(role_str)

                payload = ExperienceCreate(
                    agent_id=agent_id,
                    agent_role=role,
                    task=arguments["task"],
                    action=arguments["action"],
                    result=arguments["result"],
                    success=arguments["success"],
                    problem=arguments.get("problem"),
                    solution=arguments.get("solution"),
                    technologies=arguments.get("technologies", []),
                    run_id="mcp-session",
                    system=SystemCondition.ACTIONCLOUD,
                )
                res = memory_service.store_experience(payload)
                return f"Experience queued successfully. ID: {res['id']}"

            elif name in ("reuse_memory", "report_memory_reuse"):
                exp_id = uuid.UUID(arguments["experience_id"])
                success = arguments["success"]
                agent_id = arguments.get("agent_id")
                res = memory_service.record_reuse(experience_id=exp_id, success=success, agent_id=agent_id)
                return (
                    f"Reuse recorded. Count: {res['reuse_count']}, "
                    f"Current Tier: {res['tier']}, Transition: {res['transition']}"
                )

            elif name in ("get_memory_metrics", "get_fleet_metrics"):
                metrics = memory_service.get_metrics()
                return json.dumps(metrics, indent=2)

            else:
                return f"Unknown tool: {name}"

        except Exception as e:
            log.error("MCP tool execution error (%s): %s", name, e)
            return f"Error executing ActionCloud MCP tool {name}: {e}"

    def run_stdio(self) -> None:
        """Standard JSON-RPC 2.0 Stdio loop for MCP protocol."""
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                req = json.loads(line)
                method = req.get("method")
                msg_id = req.get("id")

                if method == "initialize":
                    res = {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {"tools": {}},
                            "serverInfo": {"name": "ActionCloud MCP Server", "version": "0.3.0"},
                        },
                    }
                    sys.stdout.write(json.dumps(res) + "\n")
                    sys.stdout.flush()

                elif method == "tools/list":
                    res = {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {"tools": self.get_tools_manifest()},
                    }
                    sys.stdout.write(json.dumps(res) + "\n")
                    sys.stdout.flush()

                elif method == "tools/call":
                    params = req.get("params", {})
                    tool_name = params.get("name")
                    arguments = params.get("arguments", {})
                    output = self.handle_call_tool(tool_name, arguments)

                    res = {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {"content": [{"type": "text", "text": output}]},
                    }
                    sys.stdout.write(json.dumps(res) + "\n")
                    sys.stdout.flush()

            except Exception as e:
                log.error("MCP stdio parse error: %s", e)


if __name__ == "__main__":
    server = ActionCloudMCPServer()
    server.run_stdio()

