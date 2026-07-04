"""
沙箱隔离模块 — 在子进程中执行工具调用

每种工具在独立的 Python 子进程中运行，带超时保护。
工具崩溃时不会影响主进程，超时时返回错误提示。

用法:
    from sandbox import Sandbox
    sandbox = Sandbox(timeout=30)
    result = sandbox.run(tool_call_dict)
"""

import subprocess
import json
import sys
import os
import textwrap

# _sandbox_runner.py 的路径（和 sandbox.py 同目录）
_RUNNER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_sandbox_runner.py")


def _ensure_runner():
    """确保子进程运行脚本存在"""
    if not os.path.exists(_RUNNER_PATH):
        runner_code = textwrap.dedent("""\
        import sys
        import json
        import os

        # 把项目目录加入路径
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

        from react_loop import TOOL_REGISTRY

        if len(sys.argv) < 2:
            print("缺少工具调用参数")
            sys.exit(1)

        try:
            tool_call = json.loads(sys.argv[1])
        except json.JSONDecodeError as e:
            print(f"参数解析失败: {e}")
            sys.exit(1)

        name = tool_call["function"]["name"]
        try:
            arguments = json.loads(tool_call["function"]["arguments"])
        except json.JSONDecodeError:
            arguments = {}

        if name not in TOOL_REGISTRY:
            print(f"未知工具: {name}")
            sys.exit(1)

        try:
            result = TOOL_REGISTRY[name](**arguments)
            print(result)
        except Exception as e:
            print(f"工具执行错误: {e}")
            sys.exit(1)
        """)
        with open(_RUNNER_PATH, "w", encoding="utf-8") as f:
            f.write(runner_code)


class Sandbox:
    """工具沙箱——在子进程中执行工具，带超时保护

    用法:
        sandbox = Sandbox(timeout=30)
        result = sandbox.run({
            "function": {"name": "calculator", "arguments": '{"expression": "1+1"}'}
        })
    """

    def __init__(self, timeout: int = 30, enabled: bool = True, prewarm: bool = True):
        self.timeout = timeout
        self.enabled = enabled
        self._prewarmed = False
        _ensure_runner()
        if prewarm and enabled:
            self._prewarm()

    def _prewarm(self):
        """预热子进程：启动一次轻量计算，让 Python 缓存字节码和模块导入"""
        try:
            warmup_payload = json.dumps({
                "function": {"name": "get_time", "arguments": "{}"}
            })
            subprocess.run(
                [sys.executable, _RUNNER_PATH, warmup_payload],
                capture_output=True, text=True, timeout=10,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            self._prewarmed = True
        except Exception:
            self._prewarmed = False

    @property
    def warm_status(self) -> str:
        return "已预热" if self._prewarmed else "未预热"

    def run(self, tool_call: dict) -> str:
        """在子进程中执行工具调用

        参数:
            tool_call: 标准的 LLM tool_call 字典

        返回:
            工具执行结果的字符串
        """
        if not self.enabled:
            # 沙箱关闭时直接返回标识，由 react_loop 自行执行
            return "__SANDBOX_DISABLED__"

        payload = json.dumps(tool_call, ensure_ascii=False)
        project_dir = os.path.dirname(os.path.abspath(__file__))

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
                stderr = result.stderr.strip()[:200]
                return f"[沙箱] 工具执行失败: {stderr}"

            output = result.stdout.strip()
            return output if output else "(工具无返回)"

        except subprocess.TimeoutExpired:
            return f"[沙箱] 工具执行超时（{self.timeout}秒）"
        except FileNotFoundError:
            return f"[沙箱] 找不到 Python 解释器"
        except Exception as e:
            return f"[沙箱] 异常: {e}"


# ============================================================
# 全局实例（默认关闭沙箱，供 react_loop.py 导入后启用）
# ============================================================

SANDBOX = Sandbox(enabled=False)


# ============================================================
# 工具定义 + 工具函数
# ============================================================

SANDBOX_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "toggle_sandbox",
        "description": "开启或关闭工具沙箱隔离。开启后每个工具在独立子进程中执行，崩溃不影响主进程。",
        "parameters": {
            "type": "object",
            "properties": {
                "enabled": {
                    "type": "boolean",
                    "description": "true=开启沙箱，false=关闭"
                }
            },
            "required": ["enabled"],
        },
    },
}


def tool_toggle_sandbox(enabled: bool) -> str:
    """运行时切换沙箱状态"""
    SANDBOX.enabled = enabled
    SANDBOX.timeout = 30
    return f"沙箱已{'开启' if enabled else '关闭'}"

