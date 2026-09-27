"""
ActionCloud — MCP stdio server.

Newline-delimited JSON-RPC 2.0 over stdin/stdout, as used by Claude Desktop,
Cursor and other MCP clients. stdout carries protocol messages ONLY; all logs
go to stderr.

Tool arguments accept both the documented names and the older ones, e.g.
`task` or `query`, `agent_role` or `role`, `k_inject` or `max_memories`,
`action_taken` or `action`, so existing client configs keep working.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from typing import Any, Dict, List, Optional, Tuple

from .policy import MemorySelectionPolicy
from .schema import AgentRole, ExperienceCreate, MemoryTier, SystemCondition
from .service import memory_service

logging.basicConfig(level=logging.ERROR, stream=sys.stderr)
log = logging.getLogger(__name__)

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "actioncloud", "version": "0.4.0"}

ROLE_ENUM = [r.value for r in AgentRole]
TIER_ENUM = [t.value for t in MemoryTier]

# canonical name -> accepted aliases (first match wins)
ALIASES = {
    "task": ("task", "query"),
    "agent_role": ("agent_role", "role"),
    "k_inject": ("k_inject", "max_memories"),
    "action": ("action_taken", "action"),
}

TOOL_ALIASES = {
    "search_fleet_memory": "search_memory",
    "store_experience": "remember_experience",
    "report_memory_reuse": "reuse_memory",
    "get_fleet_metrics": "get_memory_metrics",
}


class ToolError(Exception):
    """A client-facing error (bad arguments, unknown id). Message is safe to show."""


def _arg(args: Dict[str, Any], canonical: str, default: Any = None, required: bool = False) -> Any:
    for name in ALIASES.get(canonical, (canonical,)):
        if name in args and args[name] is not None:
            return args[name]
    if required:
        raise ToolError(f"missing required argument: {canonical}")
    return default


def _role(args: Dict[str, Any], default: Optional[str] = None) -> Optional[AgentRole]:
    value = _arg(args, "agent_role", default)
    if value is None:
        return None
    try:
        return AgentRole(value)
    except ValueError:
        raise ToolError(f"unknown agent_role {value!r}; expected one of {ROLE_ENUM}") from None


class ActionCloudMCPServer:
    """MCP tools backed directly by MemoryService (no HTTP hop)."""

    def __init__(self, service=memory_service) -> None:
        self.service = service

    # -- manifest ------------------------------------------------------------

    def get_tools_manifest(self) -> List[Dict[str, Any]]:
        identity = {
            "agent_id": {"type": "string", "description": "Identity of the calling agent"},
            "agent_role": {"type": "string", "enum": ROLE_ENUM, "description": "Role of the calling agent"},
        }
        return [
            {
                "name": "get_memory_context",
                "description": (
                    "Get a compact, token-budgeted block of relevant prior experience "
                    "to prepend to your prompt before starting a task. Returns nothing "
                    "if no memory is relevant enough."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string", "description": "The task you are about to do"},
                        **identity,
                        "technologies": {"type": "array", "items": {"type": "string"}},
                        "k_inject": {"type": "integer", "minimum": 1, "maximum": 5, "default": 1},
                        "token_budget": {"type": "integer", "minimum": 100, "default": 1000},
                    },
                    "required": ["task"],
                },
            },
            {
                "name": "search_memory",
                "description": "Hybrid vector + full-text search over governed fleet memory.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        **identity,
                        "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 5},
                        "min_tier": {"type": "string", "enum": TIER_ENUM, "default": "private"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "remember_experience",
                "description": "Store a completed task experience for future reuse.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string"},
                        "action_taken": {"type": "string", "description": "What you did (numbered steps work best)"},
                        "result": {"type": "string"},
                        "success": {"type": "boolean"},
                        **identity,
                        "problem": {"type": "string"},
                        "solution": {"type": "string"},
                        "technologies": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["task", "action_taken", "result", "success", "agent_id"],
                },
            },
            {
                "name": "reuse_memory",
                "description": (
                    "Report whether an injected memory helped. Drives promotion/demotion. "
                    "Reports on your own memories are recorded but not counted."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "experience_id": {"type": "string"},
                        "success": {"type": "boolean"},
                        "agent_id": {"type": "string"},
                    },
                    "required": ["experience_id", "success", "agent_id"],
                },
            },
            {
                "name": "get_memory_metrics",
                "description": "Aggregate governance and efficiency metrics.",
                "inputSchema": {"type": "object", "properties": {"run_id": {"type": "string"}}},
            },
        ]

    # -- tools -----------------------------------------------------------------

    def call_tool(self, name: str, args: Dict[str, Any]) -> Tuple[str, bool]:
        """Run a tool. Returns (text, is_error)."""
        name = TOOL_ALIASES.get(name, name)
        try:
            if name == "get_memory_context":
                pol = MemorySelectionPolicy.from_env()
                pol.max_context_memories = int(_arg(args, "k_inject", pol.max_context_memories))
                pol.context_token_budget = int(args.get("token_budget", pol.context_token_budget))
                res = self.service.prepare_context(
                    query=_arg(args, "task", required=True),
                    agent_id=args.get("agent_id"),
                    role=_role(args),
                    policy=pol,
                    technologies=args.get("technologies") or [],
                )
                if not res["context"]:
                    return "No relevant governed prior experience found.", False
                return (
                    f"{res['context']}\n\n"
                    f"--- ActionCloud ---\n"
                    f"experience_ids: {', '.join(res['injected_experience_ids'])}\n"
                    f"candidates: {res['candidate_count']} | injected: {res['final_count']} | "
                    f"context tokens: {res['context_tokens']}\n"
                    f"After the task, call reuse_memory for each id with success true/false.",
                    False,
                )

            if name == "search_memory":
                try:
                    min_tier = MemoryTier(args.get("min_tier", "private"))
                except ValueError:
                    raise ToolError(f"unknown min_tier; expected one of {TIER_ENUM}") from None
                results = self.service.search_memory(
                    query=_arg(args, "task", required=True),
                    limit=int(args.get("limit", 5)),
                    min_tier=min_tier,
                    agent_id=args.get("agent_id"),
                    agent_role=_role(args),
                )
                if not results:
                    return "No matching fleet experiences found.", False
                lines = ["## ActionCloud Fleet Memory Results\n"]
                for i, r in enumerate(results, 1):
                    lines.append(f"### {i}. [{r.tier.value.upper()}] {r.task}  (id {r.id}, relevance {r.relevance:.2f})")
                    if r.problem:
                        lines.append(f"- Problem: {r.problem}")
                    if r.solution:
                        lines.append(f"- Solution: {r.solution}")
                    lines.append(f"- Result: {r.result}")
                    lines.append("")
                return "\n".join(lines), False

            if name == "remember_experience":
                payload = ExperienceCreate(
                    agent_id=_arg(args, "agent_id", required=True),
                    agent_role=_role(args, default="coding"),
                    task=_arg(args, "task", required=True),
                    action=_arg(args, "action", required=True),
                    result=_arg(args, "result", required=True),
                    success=bool(_arg(args, "success", required=True)),
                    problem=args.get("problem"),
                    solution=args.get("solution"),
                    technologies=args.get("technologies") or [],
                    run_id="mcp-session",
                    system=SystemCondition.ACTIONCLOUD,
                )
                res = self.service.store_experience(payload)
                verb = "queued" if res["queued"] else "stored"
                return f"Experience {verb}. ID: {res['id']}", False

            if name == "reuse_memory":
                try:
                    exp_id = uuid.UUID(str(_arg(args, "experience_id", required=True)))
                except ValueError:
                    raise ToolError("experience_id is not a valid UUID") from None
                res = self.service.record_reuse(
                    experience_id=exp_id,
                    success=bool(_arg(args, "success", required=True)),
                    agent_id=_arg(args, "agent_id", required=True),
                )
                return (
                    f"Reuse recorded (counted={res['counted']}). "
                    f"Reuses: {res['reuse_count']}, tier: {res['tier']}, "
                    f"transition: {res['transition']}. {res['reason']}",
                    False,
                )

            if name == "get_memory_metrics":
                return json.dumps(self.service.get_metrics(args.get("run_id")), indent=2, default=str), False

            return f"Unknown tool: {name}", True

        except ToolError as e:
            return f"Invalid arguments for {name}: {e}", True
        except KeyError as e:
            return f"Not found: {e}", True
        except ValueError as e:
            return f"Invalid arguments for {name}: {e}", True
        except Exception:  # noqa: BLE001
            # Full traceback to stderr only; never leak internals to the client.
            log.exception("MCP tool %s failed", name)
            return f"ActionCloud tool {name} failed with an internal error (see server log).", True

    def handle_call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """Backward-compatible text-only wrapper around call_tool."""
        return self.call_tool(name, arguments)[0]

    # -- JSON-RPC --------------------------------------------------------------

    def handle_message(self, req: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Return the JSON-RPC response for one request, or None for notifications."""
        method = req.get("method")
        msg_id = req.get("id")
        is_notification = "id" not in req

        def ok(result: Dict[str, Any]) -> Dict[str, Any]:
            return {"jsonrpc": "2.0", "id": msg_id, "result": result}

        if is_notification:
            return None  # e.g. notifications/initialized — never answered

        if method == "initialize":
            return ok({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": SERVER_INFO,
            })
        if method == "ping":
            return ok({})
        if method == "tools/list":
            return ok({"tools": self.get_tools_manifest()})
        if method == "tools/call":
            params = req.get("params") or {}
            text, is_error = self.call_tool(params.get("name", ""), params.get("arguments") or {})
            return ok({"content": [{"type": "text", "text": text}], "isError": is_error})

        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }

    def run_stdio(self) -> None:
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError:
                resp = {"jsonrpc": "2.0", "id": None,
                        "error": {"code": -32700, "message": "Parse error"}}
            else:
                resp = self.handle_message(req)
            if resp is not None:
                sys.stdout.write(json.dumps(resp, default=str) + "\n")
                sys.stdout.flush()


def main() -> None:
    ActionCloudMCPServer().run_stdio()


if __name__ == "__main__":
    main()
