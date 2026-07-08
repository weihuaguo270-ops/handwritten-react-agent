"""Sandbox — LangGraph 版工具沙箱隔离

核心思想跟手写版一样：在 subprocess 中执行工具调用，崩溃/超时不拖死主进程。

三策略模式（与手写版一致）：
  - off:  全部在当前进程执行（最快）
  - auto: 自动判断——safe 工具直接跑，io/cpu 工具走子进程（默认）
  - on:   全部走子进程（最安全）

与手写版 sandbox.py 的区别：
  1. 不硬编码 TOOL_REGISTRY 路径，而是通过参数传入工具的 module path
  2. _sandbox_runner.py 共享手写版的（路径相同），避免两份维护
  3. 通过 Harness 统一入口 expose

用法：
    sandbox = Sandbox(strategy="auto", timeout=30)
    result = sandbox.run({
        "function": {"name": "calculator", "arguments": '{"expression": "1+1"}'}
    })
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

# 借用手写版的工具风险分类（复用同一份分类逻辑）
_hand_sandbox = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "harness",
    "sandbox.py",
)


def _import_classify():
    """动态导入手写版的 classify_risk / should_sandbox_by_risk"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_hand_sandbox", _hand_sandbox)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.classify_risk, mod.should_sandbox_by_risk


# 导入风险判断逻辑（延迟执行，避免循环导入）
_classify_risk = None
_should_sandbox_by_risk = None


def _lazy_import():
    global _classify_risk, _should_sandbox_by_risk
    if _classify_risk is None:
        _classify_risk, _should_sandbox_by_risk = _import_classify()


VALID_STRATEGIES = ("off", "auto", "on")


class Sandbox:
    """工具沙箱——在子进程中执行工具，带超时保护

    用法：
        sandbox = Sandbox(strategy="auto", timeout=30)
        result = sandbox.run(tool_call_dict)
    """

    def __init__(self, timeout: int = 30, strategy: str = "auto"):
        if strategy not in VALID_STRATEGIES:
            raise ValueError(f"未知沙箱策略: {strategy}，可选: {VALID_STRATEGIES}")
        self.timeout = timeout
        self.strategy = strategy
        self._runner_ready = False
        _lazy_import()

    @property
    def enabled(self) -> bool:
        """兼容旧接口"""
        return self.strategy != "off"

    @enabled.setter
    def enabled(self, value: bool):
        """兼容旧接口"""
        self.strategy = "auto" if value else "off"

    def add_unsafe_tool(self, tool_name: str):
        """兼容旧接口——在 auto 模式下，这个方法是空操作，因为风险判断由 classify_risk 决定

        LangGraph 版的 Harness 旧代码会调用 add_unsafe_tool 注册白名单，
        auto 模式下白名单等价于 safe 工具，由 classify_risk 自动处理。
        """
        pass

    def should_sandbox(self, tool_name: str) -> bool:
        """判断某个工具是否应当在沙箱中执行

        of → False（全部不走）
        auto → safe 工具 False，其他 True
        on → True（全部走）
        """
        if self.strategy == "off":
            return False
        if self.strategy == "on":
            return True
        # auto 模式
        return _should_sandbox_by_risk(tool_name, "auto")

    def run(self, tool_call: dict) -> str:
        """在子进程中执行工具调用

        参数:
            tool_call: 标准的 LLM tool_call 字典
                {"function": {"name": "calculator", "arguments": '{"expression": "1+1"}'}}

        返回:
            工具执行结果的字符串；跳过时返回 "__SANDBOX_DISABLED__"
        """
        tool_name = tool_call.get("function", {}).get("name", "")

        if not self.should_sandbox(tool_name):
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

        hand_dir = os.path.dirname(_RUNNER_PATH)
        os.makedirs(hand_dir, exist_ok=True)

        runner_code = textwrap.dedent("""\
        import sys
        import json
        import os

        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        try:
            from graph.tools import get_tools
            TOOL_MAP = {t.name: t for t in get_tools()}
        except ImportError:
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
