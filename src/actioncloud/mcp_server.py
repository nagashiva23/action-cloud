"""
ActionCloud — Model Context Protocol (MCP) Server.

Exposes ActionCloud tools over MCP stdio protocol, allowing IDEs
(Cursor, Antigravity, Windsurf, VS Code) and Claude Desktop to search,
store, and govern fleet memory directly within your editor.

Run with:
    python -m actioncloud.mcp_server
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict, List, Optional
import httpx

logging.basicConfig(level=logging.ERROR, stream=sys.stderr)
log = logging.getLogger(__name__)

DEFAULT_API_URL = "http://localhost:8000"


class ActionCloudMCPServer:
    """
    MCP Stdio Server wrapper for ActionCloud.
    """

    def __init__(self, api_url: str = DEFAULT_API_URL) -> None:
        self.api_url = api_url.rstrip("/")
        self.http = httpx.Client(base_url=self.api_url, timeout=10.0)

    def get_tools_manifest(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "search_fleet_memory",
                "description": (
                    "Search ActionCloud shared fleet memory for prior agent experiences, "
                    "proven solutions, and extracted procedural workflows."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Technical task, problem, or error message to search for",
                        },
                        "limit": {
                            "type": "integer",
                            "default": 5,
                            "description": "Maximum number of results to return",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "store_experience",
                "description": (
                    "Store a completed task experience into ActionCloud shared fleet memory."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string", "description": "What you were asked to do"},
                        "action": {"type": "string", "description": "What you actually did"},
                        "result": {"type": "string", "description": "Outcome of the task"},
                        "success": {"type": "boolean", "description": "Whether the task succeeded"},
                        "problem": {"type": "string", "description": "Obstacle hit, if any"},
                        "solution": {"type": "string", "description": "What resolved the problem"},
                        "technologies": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tech stacks used (e.g. docker, fastapi, postgres)",
                        },
                    },
                    "required": ["task", "action", "result", "success"],
                },
            },
            {
                "name": "report_memory_reuse",
                "description": (
                    "Report outcome after reusing a prior experience to trigger MemoryJudge governance promotion."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "experience_id": {"type": "string", "description": "UUID of the experience reused"},
                        "success": {"type": "boolean", "description": "Whether the reuse helped succeed"},
                    },
                    "required": ["experience_id", "success"],
                },
            },
            {
                "name": "get_fleet_metrics",
                "description": (
                    "Retrieve ActionCloud system-wide token savings, cost reduction, and governance tier distribution."
                ),
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]

    def handle_call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        try:
            if name == "search_fleet_memory":
                q = arguments["query"]
                limit = arguments.get("limit", 5)
                resp = self.http.get("/search", params={"q": q, "limit": limit})
                resp.raise_for_status()
                results = resp.json().get("results", [])
                if not results:
                    return "No matching fleet experiences found."

                lines = ["## ActionCloud Fleet Memory Results\n"]
                for i, r in enumerate(results, 1):
                    lines.append(f"### {i}. [{r['tier'].upper()}] {r['task']}")
                    if r.get("problem"):
                        lines.append(f"- Problem: {r['problem']}")
                    if r.get("solution"):
                        lines.append(f"- Solution: {r['solution']}")
                    lines.append(f"- Result: {r['result']}")
                    if r.get("workflow"):
                        lines.append(f"- Reusable Workflow: {json.dumps(r['workflow'])}")
                    lines.append("")
                return "\n".join(lines)

            elif name == "store_experience":
                payload = {
                    "agent_id": "mcp-ide-agent",
                    "agent_role": "coding",
                    "task": arguments["task"],
                    "action": arguments["action"],
                    "result": arguments["result"],
                    "success": arguments["success"],
                    "problem": arguments.get("problem"),
                    "solution": arguments.get("solution"),
                    "technologies": arguments.get("technologies", []),
                    "run_id": "ide-mcp-session",
                    "system": "actioncloud",
                }
                resp = self.http.post("/experiences", json=payload)
                resp.raise_for_status()
                return f"Experience queued successfully. ID: {resp.json().get('id')}"

            elif name == "report_memory_reuse":
                exp_id = arguments["experience_id"]
                success = arguments["success"]
                resp = self.http.post(
                    f"/experiences/{exp_id}/reuse", json={"success": success, "agent_id": "mcp-ide-agent"}
                )
                resp.raise_for_status()
                data = resp.json()
                return (
                    f"Reuse recorded. Count: {data.get('reuse_count')}, "
                    f"Current Tier: {data.get('tier')}, Transition: {data.get('transition')}"
                )

            elif name == "get_fleet_metrics":
                resp = self.http.get("/metrics")
                resp.raise_for_status()
                return json.dumps(resp.json(), indent=2)

            else:
                return f"Unknown tool: {name}"

        except Exception as e:
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
                            "serverInfo": {"name": "ActionCloud MCP Server", "version": "0.2.0"},
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
