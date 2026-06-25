
"""
MCP Client — 独立的 MCP 协议实现模块
======================================
纯 Python，仅依赖标准库（json + subprocess）
任何 Agent 都可以 import 使用

用法:
    from mcp_client import MCPClient
    client = MCPClient("uvx", ["mcp-server-time"])
    client.connect()
    tools = client.discover_tools()
    result = client.call_tool("get_current_time", {"timezone": "Asia/Shanghai"})
"""

import json
import subprocess
import os as _os


class MCPClient:
    """通过 stdin/stdout（stdio）连接 MCP Server，实现 JSON-RPC 2.0 通信"""

    def __init__(self, command, args=None, env=None):
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.proc = None
        self._req_id = 0
        self.tools = []

    # ---------------------------------------------------------------
    # 生命周期
    # ---------------------------------------------------------------

    def connect(self, timeout=15):
        """启动 MCP Server 子进程 -> 握手 initialize"""
        env = _os.environ.copy()
        env.update(self.env)
        self.proc = subprocess.Popen(
            [self.command] + self.args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        resp = self._rpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "mcp-client-py", "version": "1.0.0"},
        })
        si = resp.get("serverInfo", {})
        print(f"  [MCP] 已连接: {si.get('name', '?')} v{si.get('version', '?')}")
        self._notify("notifications/initialized")

    def close(self):
        """关闭连接，终止子进程"""
        if self.proc:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            self.proc.terminate()
            self.proc = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ---------------------------------------------------------------
    # MCP 方法
    # ---------------------------------------------------------------

    def discover_tools(self):
        """调用 tools/list -> 返回工具列表"""
        resp = self._rpc("tools/list")
        self.tools = resp.get("tools", [])
        for t in self.tools:
            desc = t.get("description", "")[:60]
            print(f"  [MCP] {t['name']} - {desc}")
        print(f"  [MCP] 共 {len(self.tools)} 个工具")
        return self.tools

    def call_tool(self, name, arguments=None):
        """调用 tools/call -> 返回纯文本结果"""
        resp = self._rpc("tools/call", {
            "name": name,
            "arguments": arguments or {},
        })
        texts = [c["text"] for c in resp.get("content", []) if c.get("type") == "text"]
        return "\n".join(texts)

    def to_tool_definitions(self):
        """转成 OpenAI Function Calling JSON Schema"""
        defs = []
        for t in self.tools:
            defs.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("inputSchema", {
                        "type": "object", "properties": {}
                    }),
                },
            })
        return defs

    # ---------------------------------------------------------------
    # JSON-RPC 2.0 通信原语
    # ---------------------------------------------------------------

    def _rpc(self, method, params=None):
        self._req_id += 1
        req = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": self._req_id,
        }
        line = json.dumps(req) + "\n"
        self.proc.stdin.write(line.encode("utf-8"))
        self.proc.stdin.flush()

        resp_line = self.proc.stdout.readline()
        if not resp_line:
            raise RuntimeError("MCP Server 连接断开")
        resp_line = resp_line.decode("utf-8")
        resp = json.loads(resp_line)


        if "error" in resp:
            e = resp["error"]
            raise RuntimeError(f"MCP 错误 [{e.get('code')}]: {e.get('message')}")
        return resp.get("result", {})

    def _notify(self, method, params=None):
        req = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        line = json.dumps(req) + "\n"
        self.proc.stdin.write(line.encode("utf-8"))
        self.proc.stdin.flush()


# ================================================================
# 命令行测试
# ================================================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python mcp_client.py uvx mcp-server-time")
        sys.exit(1)
    with MCPClient(sys.argv[1], sys.argv[2:]) as c:
        c.connect()
        c.discover_tools()
