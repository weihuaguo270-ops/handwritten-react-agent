"""
沙箱隔离模块 — 在子进程中执行工具调用

三策略模式：
  - off:  全部在当前进程执行（最快，适合本地开发）
  - auto: 自动判断——safe 工具直接跑，io/cpu 工具走子进程（默认）
  - on:   全部走子进程（最安全，适合运行不可信代码）

每种工具在独立的 Python 子进程中运行，带超时保护。
工具崩溃时不会影响主进程，超时时返回错误提示。

用法:
    from sandbox import Sandbox
    sandbox = Sandbox(strategy="auto", timeout=30)
    result = sandbox.run(tool_call_dict)
"""

import subprocess
import json
import sys
import os
import textwrap

# _sandbox_runner.py 的路径（和 sandbox.py 同目录）
_RUNNER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_sandbox_runner.py")


# ============================================================
# 工具风险等级标签（按类别判断）
# ============================================================

RISK_SAFE = "safe"  # 0ms 纯本地，不可能崩溃（时间、计算、开关类配置）
RISK_IO = "io"      # 网络/文件 IO，可能超时（搜索、抓取、RAG、清理）
RISK_CPU = "cpu"    # 纯计算但可能耗时长或卡住（ToT 内部多轮调 LLM）


def classify_risk(tool_name: str) -> str:
    """根据工具名判断风险等级

    返回 RISK_SAFE / RISK_IO / RISK_CPU 之一。
    未知工具默认走 io（走沙箱），宁可多隔离也不漏掉。
    """
    safe_tools = {
        "get_time", "get_current_time",
        "calculator",
        "switch_cot_strategy", "switch_role", "switch_context_strategy",
        "toggle_sandbox",
        "start_dashboard",
    }
    io_tools = {
        "web_search", "fetch_page",
        "rag_query",
        "clear_trajectories",
    }
    cpu_tools = {
        "summarize",
        "tot_reasoning",
    }
    if tool_name in safe_tools:
        return RISK_SAFE
    if tool_name in io_tools:
        return RISK_IO
    if tool_name in cpu_tools:
        return RISK_CPU
    # 未知工具: 默认 safe（不走沙箱），因为可能是 MCP/HTTP 等外部工具
    return RISK_SAFE


def should_sandbox_by_risk(tool_name: str, strategy: str) -> bool:
    """根据策略和工具名判断是否走沙箱

    参数:
        tool_name: 工具名
        strategy:  "off" / "auto" / "on"

    返回:
        True=走子进程，False=直接执行
    """
    if strategy == "on":
        return True
    if strategy == "off":
        return False
    # auto 模式：只对 safe 工具跳过沙箱
    risk = classify_risk(tool_name)
    return risk in (RISK_IO, RISK_CPU)


# ============================================================
# 子进程 runner 确保
# ============================================================

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


# ============================================================
# Sandbox 类
# ============================================================

VALID_STRATEGIES = ("off", "auto", "on")


class Sandbox:
    """工具沙箱——在子进程中执行工具，带超时保护

    三策略模式：
      - "off":  全部在当前进程执行（最快）
      - "auto": 自动判断（默认）——safe 工具直接跑，io/cpu 工具走子进程
      - "on":   全部走子进程（最安全）

    用法:
        sandbox = Sandbox(strategy="auto", timeout=30)
        result = sandbox.run({
            "function": {"name": "calculator", "arguments": '{"expression": "1+1"}'}
        })
    """

    def __init__(self, timeout: int = 30, strategy: str = "auto", prewarm: bool = True):
        if strategy not in VALID_STRATEGIES:
            raise ValueError(f"未知沙箱策略: {strategy}，可选: {VALID_STRATEGIES}")
        self.timeout = timeout
        self.strategy = strategy
        self._prewarmed = False
        _ensure_runner()
        if prewarm and strategy != "off":
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
    def enabled(self) -> bool:
        """兼容旧接口：返回当前是否处于可执行子进程的状态"""
        return self.strategy != "off"

    @enabled.setter
    def enabled(self, value: bool):
        """兼容旧接口：set enabled=True → 切 auto，enabled=False → 切 off"""
        self.strategy = "auto" if value else "off"

    @property
    def warm_status(self) -> str:
        return "已预热" if self._prewarmed else "未预热"

    def should_sandbox(self, tool_name: str) -> bool:
        """对外暴露：判断某个工具是否应当在沙箱中执行

        LangGraph 版的 Sandbox 也有同名方法，保持接口一致。
        """
        return should_sandbox_by_risk(tool_name, self.strategy)

    def run(self, tool_call: dict) -> str:
        """在子进程中执行工具调用

        参数:
            tool_call: 标准的 LLM tool_call 字典

        返回:
            工具执行结果的字符串；跳过时返回 "__SANDBOX_DISABLED__"
        """
        tool_name = tool_call.get("function", {}).get("name", "")

        # auto 模式下，safe 工具跳过沙箱
        if not should_sandbox_by_risk(tool_name, self.strategy):
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
                return f"[沙箱] 工具 '{tool_name}' 执行失败: {stderr}"

            output = result.stdout.strip()
            return output if output else f"(工具 '{tool_name}' 无返回)"

        except subprocess.TimeoutExpired:
            return f"[沙箱] 工具 '{tool_name}' 执行超时（{self.timeout}秒）"
        except FileNotFoundError:
            return f"[沙箱] 找不到 Python 解释器"
        except Exception as e:
            return f"[沙箱] 工具 '{tool_name}' 异常: {e}"


# ============================================================
# 全局实例（默认 auto 模式）
# ============================================================

SANDBOX = Sandbox(strategy="auto")


# ============================================================
# 工具定义 + 工具函数
# ============================================================

SANDBOX_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "toggle_sandbox",
        "description": "切换工具沙箱模式: off(全部直接执行)/auto(自动按工具风险决定，推荐)/on(全部子进程隔离)",
        "parameters": {
            "type": "object",
            "properties": {
                "strategy": {
                    "type": "string",
                    "enum": ["off", "auto", "on"],
                    "description": "off=不用沙箱, auto=自动判断(默认), on=全部隔离"
                }
            },
            "required": ["strategy"],
        },
    },
}


def tool_toggle_sandbox(strategy: str = "auto") -> str:
    """运行时切换沙箱策略"""
    if strategy not in VALID_STRATEGIES:
        return f"未知策略: {strategy}，可选: {', '.join(VALID_STRATEGIES)}"
    old = SANDBOX.strategy
    SANDBOX.strategy = strategy
    SANDBOX.timeout = 30
    return f"沙箱策略: {old} → {strategy}{'（自动判断）' if strategy == 'auto' else ''}"
