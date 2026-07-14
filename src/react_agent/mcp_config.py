"""Load MCP server launch commands from config (never hardcode machine paths)."""
from __future__ import annotations

import json
import os
from typing import Optional


# Portable default only — no OS-specific absolute paths.
PORTABLE_DEFAULT_MCP_SERVERS: list[list[str]] = [
    ["uvx", "mcp-server-time"],
]


def _candidate_config_paths() -> list[str]:
    paths: list[str] = []
    env = os.environ.get("REACT_AGENT_MCP_CONFIG", "").strip()
    if env:
        paths.append(env)
    # Package-adjacent / cwd (same pattern as llm_config.json)
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    paths.append(os.path.join(root, "mcp_servers.json"))
    paths.append(os.path.join(os.getcwd(), "mcp_servers.json"))
    return paths


def load_mcp_server_commands(
    config_path: Optional[str] = None,
) -> list[list[str]]:
    """Return list of argv lists for MCP servers.

    Resolution order:
      1. ``config_path`` argument
      2. ``REACT_AGENT_MCP_CONFIG`` env
      3. ``mcp_servers.json`` under project root / cwd
      4. portable default (``uvx mcp-server-time`` only)

    JSON schema::
        {"servers": [["uvx", "mcp-server-time"], ["npx", "-y", "..."]]}
    """
    candidates = [config_path] if config_path else []
    candidates.extend(_candidate_config_paths())

    for path in candidates:
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"  [MCP] 配置不可用 ({path}): {e}")
            continue
        servers = data.get("servers") if isinstance(data, dict) else None
        if not isinstance(servers, list) or not servers:
            print(f"  [MCP] 配置缺少 servers 列表: {path}")
            continue
        out: list[list[str]] = []
        for entry in servers:
            if isinstance(entry, list) and entry and all(isinstance(x, str) for x in entry):
                out.append(list(entry))
            elif isinstance(entry, str) and entry.strip():
                out.append(entry.split())
        if out:
            print(f"  [MCP] 已加载配置: {path} ({len(out)} server)")
            return out

    return [list(cmd) for cmd in PORTABLE_DEFAULT_MCP_SERVERS]
