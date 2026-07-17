"""Core 默认路径不应在 import react_loop 时拉起 MCP / Orchestrator / RAG。"""
from __future__ import annotations

import sys


def test_react_loop_import_does_not_load_experimental_modules():
    for name in (
        "react_agent.mcp_client",
        "react_agent.orchestrator",
        "react_agent.rag",
    ):
        sys.modules.pop(name, None)

    import react_agent.react_loop as rl  # noqa: F401

    assert "react_agent.mcp_client" not in sys.modules
    assert "react_agent.orchestrator" not in sys.modules
    assert "react_agent.rag" not in sys.modules
