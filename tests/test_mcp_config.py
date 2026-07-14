"""Tests for MCP config loading (no machine-local paths in defaults)."""
import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from react_agent.mcp_config import (  # noqa: E402
    PORTABLE_DEFAULT_MCP_SERVERS,
    load_mcp_server_commands,
)


def test_portable_default_has_no_drive_letters():
    for cmd in PORTABLE_DEFAULT_MCP_SERVERS:
        joined = " ".join(cmd).lower()
        assert "c:/" not in joined and "d:/" not in joined
        assert "program files" not in joined
        assert "agent_learning" not in joined


def test_load_from_json(tmp_path, monkeypatch):
    cfg = tmp_path / "mcp_servers.json"
    cfg.write_text(
        json.dumps({"servers": [["uvx", "mcp-server-time"], ["npx", "-y", "dummy"]]}),
        encoding="utf-8",
    )
    monkeypatch.delenv("REACT_AGENT_MCP_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    # Also clear accidental project-root hits by pointing env explicitly
    monkeypatch.setenv("REACT_AGENT_MCP_CONFIG", str(cfg))
    cmds = load_mcp_server_commands()
    assert cmds[0] == ["uvx", "mcp-server-time"]
    assert cmds[1][0] == "npx"


def test_default_fallback_when_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("REACT_AGENT_MCP_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    cmds = load_mcp_server_commands(config_path=str(tmp_path / "missing.json"))
    assert cmds == PORTABLE_DEFAULT_MCP_SERVERS
