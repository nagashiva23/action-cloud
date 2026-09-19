from __future__ import annotations

import json
import pytest

from actioncloud.mcp_server import ActionCloudMCPServer


class TestActionCloudMCPServer:
    def test_tools_manifest_has_required_tools(self):
        server = ActionCloudMCPServer()
        manifest = server.get_tools_manifest()
        tool_names = [t["name"] for t in manifest]

        assert "get_memory_context" in tool_names
        assert "search_memory" in tool_names
        assert "remember_experience" in tool_names
        assert "reuse_memory" in tool_names
        assert "get_memory_metrics" in tool_names

    def test_handle_call_tool_unknown_name(self):
        server = ActionCloudMCPServer()
        res = server.handle_call_tool("invalid_tool_name", {})
        assert "Unknown tool" in res

    def test_get_memory_metrics_tool_call(self):
        server = ActionCloudMCPServer()
        res = server.handle_call_tool("get_memory_metrics", {})
        parsed = json.loads(res)
        assert "systems" in parsed or "governance_tier_distribution" in parsed
