"""Sandbox — LangGraph 版工具沙箱隔离

核心思想跟手写版一样：在 subprocess 中执行工具调用，崩溃/超时不拖死主进程。

与手写版 sandbox.py 的区别：
  1. 不硬编码 TOOL_REGISTRY 路径，而是通过参数传入工具的 module path
  2. 支持白名单（某些工具太快了不值得开进程，直接跑）
  3. _sandbox_runner.py 共享手写版的（路径相同），避免两份维护

用法：
    sandbox = Sandbox(timeout=30)
    result = sandbox.run({
        "function": {"name": "calculator", "arguments": '{"expression": "1+1"}'}
    })
    # → "2"

快速工具（get_current_time、calculator）默认不走沙箱，因为开 subprocess 的开销
比工具本身执行还大。web_search 等涉及网络/IO 的工具默认走沙箱。
"""

import subprocess
import json
import sys
import os
import textwrap

# 共享手写版的 _sandbox_runner.py（同一个子进程执行脚本）
_RUNNER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "harness",
    "_sandbox_runner.py",
)


# 默认的快速工具白名单（开进程比执行工具本身还慢的）
_DEFAULT_UNSAFE_TOOLS = {
    "get_current_time",
    "calculator",
}


class Sandbox:
    """工具沙箱——在子进程中执行工具，带超时保护

    用法：
        sandbox = Sandbox(enabled=True, timeout=30)
        result = sandbox.run(tool_call_dict)
    """

    def __init__(self, timeout: int = 30, enabled: bool = True,
                 unsafe_tools: "set[str] | None" = None):
        self.timeout = timeout
        self.enabled = enabled
        # 默认白名单中的工具不走沙箱
        self._unsafe_tools = set(unsafe_tools) if unsafe_tools else _DEFAULT_UNSAFE_TOOLS.copy()
        self._runner_ready = False

    def add_unsafe_tool(self, tool_name: str):
        """添加一个不应在沙箱中运行的工具名

        原因通常是该工具执行极快（几 ms），开 subprocess 的 20-50ms 开销反而更慢。
        典型：get_current_time、calculator。
        """
        self._unsafe_tools.add(tool_name)

    def should_sandbox(self, tool_name: str) -> bool:
        """判断某个工具是否应当在沙箱中执行

        返回 True → 在子进程中执行（安全但开销大）
        返回 False → 直接在当前进程执行（快但崩溃会拖死 Agent）
        """
        if not self.enabled:
            return False
        return tool_name not in self._unsafe_tools

    def run(self, tool_call: dict) -> str:
        """在子进程中执行工具调用

        参数:
            tool_call: 标准的 LLM tool_call 字典
                {"function": {"name": "calculator", "arguments": '{"expression": "1+1"}'}}

        返回:
            工具执行结果的字符串；失败时返回 [沙箱] 前缀的错误消息
        """
        if not self.enabled:
            return "__SANDBOX_DISABLED__"

        tool_name = tool_call.get("function", {}).get("name", "?")
        if tool_name in self._unsafe_tools:
            return "__SANDBOX_DISABLED__"

        self._ensure_runner()
        payload = json.dumps(tool_call, ensure_ascii=False)
        project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        try:
            result = subprocess.run(
                [sys.executable, _RUNNER_PATH, payload],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=project_dir,
                env={
                    **os.environ,
                    "PYTHONIOENCODING": "utf-8",
                },
            )

            if result.returncode != 0:
                stderr = result.stderr.strip()[:300]
                return f"[沙箱] 工具 '{tool_name}' 执行失败: {stderr}"

            output = result.stdout.strip()
            return output if output else f"(工具 '{tool_name}' 无返回)"

        except subprocess.TimeoutExpired:
            return f"[沙箱] 工具 '{tool_name}' 执行超时（{self.timeout}秒）"
        except FileNotFoundError:
            return f"[沙箱] 找不到 Python 解释器"
        except Exception as e:
            return f"[沙箱] 工具 '{tool_name}' 异常: {e}"

    def _ensure_runner(self):
        """确保 _sandbox_runner.py 存在；如果找不到手写版的就自动创建一份"""
        if self._runner_ready:
            return

        if os.path.exists(_RUNNER_PATH):
            self._runner_ready = True
            return

        # 找不到手写版的 runner，在 hand-written harness/ 目录创建
        hand_dir = os.path.dirname(_RUNNER_PATH)
        os.makedirs(hand_dir, exist_ok=True)

        runner_code = textwrap.dedent("""\
        import sys
        import json
        import os

        # 把项目目录加入路径
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # 优先从 LangGraph 版的 tools.py 导入（如果存在）
        try:
            from graph.tools import get_tools
            TOOL_MAP = {t.name: t for t in get_tools()}
        except ImportError:
            # fallback 到手写版的 TOOL_REGISTRY
            from react_loop import TOOL_REGISTRY
            TOOL_MAP = TOOL_REGISTRY

        if len(sys.argv) < 2:
            print("缺少工具调用参数")
            sys.exit(1)

        try:
            tool_call = json.loads(sys.argv[1])
        except json.JSONDecodeError as e:
            print(f"参数解析失败: {e}")
            sys.exit(1)

        name = tool_call.get("function", {}).get("name", "")
        try:
            arguments = json.loads(tool_call["function"].get("arguments", "{}"))
        except (json.JSONDecodeError, KeyError):
            arguments = {}

        if name not in TOOL_MAP:
            print(f"未知工具: {name}")
            sys.exit(1)

        try:
            tool_fn = TOOL_MAP[name]
            result = tool_fn.invoke(arguments) if hasattr(tool_fn, "invoke") else tool_fn(**arguments)
            print(result)
        except Exception as e:
            print(f"工具执行错误: {e}")
            sys.exit(1)
        """)
        with open(_RUNNER_PATH, "w", encoding="utf-8") as f:
            f.write(runner_code)

        self._runner_ready = True
