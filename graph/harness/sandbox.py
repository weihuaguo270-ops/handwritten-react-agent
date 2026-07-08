"""Sandbox — LangGraph 版工具沙箱隔离

在 LangGraph 框架下，沙箱是 tools 节点内的逻辑分支，
通过 State 中的 sandbox_map 字段标记每个 tool_call 是否需要隔离执行。

流程：
  call_model 返回 tool_calls
      ↓
  tools_node 遍历 tool_calls：
    ├── sandbox_map[tool_call.id] == True → 在子进程执行
    └── sandbox_map[tool_call.id] == False → 在当前进程直接执行

风险分类逻辑独立为一个纯函数，由 build_agent 在构造时注册。
"""

import subprocess
import json
import sys
import os
import textwrap


# 子进程 runner 脚本
_RUNNER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "harness",
    "_sandbox_runner.py",
)


# ============================================================
# 风险分类（LangGraph 版，独立于手写版）
# ============================================================

RISK_SAFE = "safe"
RISK_IO = "io"
RISK_CPU = "cpu"


def classify_risk(tool_name: str) -> str:
    """判断工具的风险等级

    safe: 纯本地计算，不可能崩溃，不走沙箱（默认值）
    io:   网络/文件 IO，可能超时，走沙箱
    cpu:  纯计算但可能耗时长/卡住，走沙箱

    与手写版不同的地方：
      - 未知工具返回 safe（MCP/HTTP 工具不走沙箱）
      - 不硬编码全量工具名，只标记需要隔离的工具
    """
    io_tools = {
        "web_search", "fetch_page",
        "rag_query",
    }
    cpu_tools = {
        "summarize",
        "tot_reasoning",
    }
    if tool_name in io_tools:
        return RISK_IO
    if tool_name in cpu_tools:
        return RISK_CPU
    return RISK_SAFE


# ============================================================
# Sandbox 类（不依赖手写版，独立实现）
# ============================================================

VALID_STRATEGIES = ("off", "auto", "on")


class Sandbox:
    """工具沙箱——在子进程中执行工具，带超时保护

    三策略模式：
      - off:  全部在当前进程执行（最快）
      - auto: 自动判断——safe 工具直接跑，io/cpu 走子进程（默认）
      - on:   全部走子进程（最安全）

    与手写版的区别：
      - build_agent() 时创建 Sandbox 实例，通过 config 传入节点
      - tools_node 通过 sandbox.should_sandbox(tool_name) 判断
    """

    def __init__(self, timeout: int = 30, strategy: str = "auto"):
        if strategy not in VALID_STRATEGIES:
            raise ValueError(f"未知沙箱策略: {strategy}，可选: {VALID_STRATEGIES}")
        self.timeout = timeout
        self.strategy = strategy
        self._runner_ready = False

    def should_sandbox(self, tool_name: str) -> bool:
        """判断某个工具是否应当在沙箱中执行"""
        if self.strategy == "off":
            return False
        if self.strategy == "on":
            return True
        return classify_risk(tool_name) in (RISK_IO, RISK_CPU)

    def run(self, tool_call: dict) -> str:
        """在子进程中执行工具调用

        返回结果字符串；沙箱关闭或 safe 工具返回 "__SANDBOX_DISABLED__"
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
                capture_output=True, text=True,
                timeout=self.timeout,
                cwd=project_dir,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )

            if result.returncode != 0:
                stderr = result.stderr.strip()[:300]
                return f"[沙箱] 工具 '{tool_name}' 执行失败: {stderr}"

            output = result.stdout.strip()
            return output if output else f"(工具 '{tool_name}' 无返回)"

        except subprocess.TimeoutExpired:
            return f"[沙箱] 工具 '{tool_name}' 执行超时（{self.timeout}秒）"
        except FileNotFoundError:
            return "[沙箱] 找不到 Python 解释器"
        except Exception as e:
            return f"[沙箱] 工具 '{tool_name}' 异常: {e}"

    def _ensure_runner(self):
        """确保 _sandbox_runner.py 存在"""
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
        import importlib.util

        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # 独立加载 graph/ 目录下的工具文件
        _TOOL_MAP = {}
        graph_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "graph")
        if os.path.isdir(graph_dir):
            for fname in sorted(os.listdir(graph_dir)):
                if not fname.endswith(".py") or fname.startswith("_"):
                    continue
                mod_name = fname[:-3]
                filepath = os.path.join(graph_dir, fname)
                try:
                    spec = importlib.util.spec_from_file_location(mod_name, filepath)
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    for attr_name in dir(mod):
                        if attr_name.startswith("_"):
                            continue
                        attr = getattr(mod, attr_name)
                        if callable(attr):
                            _TOOL_MAP[attr_name] = attr
                except Exception:
                    pass

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

        if name not in _TOOL_MAP:
            print(f"未知工具: {name}")
            sys.exit(1)

        try:
            tool_fn = _TOOL_MAP[name]
            result = tool_fn.invoke(arguments) if hasattr(tool_fn, "invoke") else tool_fn(**arguments)
            print(result)
        except Exception as e:
            print(f"工具执行错误: {e}")
            sys.exit(1)
        """)
        with open(_RUNNER_PATH, "w", encoding="utf-8") as f:
            f.write(runner_code)

        self._runner_ready = True
