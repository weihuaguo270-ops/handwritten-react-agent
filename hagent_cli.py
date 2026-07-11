"""hagent — handwritten-react-agent CLI

设计参考 Claude Code 的 Observable Autonomy 原则：
  Agent 自由行动，但每一步都对用户可见。
  用户可以在 Agent 走偏时及时打断。
"""
from __future__ import annotations
import os, sys, json, glob
from contextlib import redirect_stdout
from io import StringIO
import threading
import time

_base = os.path.dirname(os.path.abspath(__file__))
os.chdir(_base)
os.environ.pop("DEEPSEEK_API_KEY", None)
for p in [_base, os.path.join(_base, "src"),
          os.path.join(_base, "experiments", "eval-engine")]:
    sys.path.insert(0, p)

from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich import box
import typer

_console = Console()
_last_traj = [""]
_history: list[dict] = []

# 预加载
with redirect_stdout(StringIO()):
    try:
        import src.handwritten_react_agent.react_loop  # noqa: F401
    except Exception:
        pass


def _import(mod: str, name: str):
    return getattr(__import__(mod, fromlist=[name]), name)


def _recent_traj() -> str:
    for d in [os.path.join(_base, "src", "handwritten_react_agent", "trajectories"),
              os.path.join(_base, "trajectories")]:
        if os.path.exists(d):
            files = sorted(glob.glob(os.path.join(d, "*.json")), reverse=True)
            if files:
                return files[0]
    return ""


# ── 工具调用拦截器（实时展示到终端）──

class ToolMonitor:
    """拦截工具调用，实时显示到终端（通过 stdout 直写）"""
    def __init__(self):
        self._orig = None

    def wrap(self, original):
        self._orig = original
        def wrapped(tc):
            name = tc.get("function", {}).get("name", "")
            args = tc.get("function", {}).get("arguments", "{}")
            # 直接写 stdout，绕过 Console（因为可能在 redirect_stdout 中）
            sys.__stdout__.write(f"  ● {name} {args[:80]}\n")
            sys.__stdout__.flush()
            result = original(tc)
            summary = result.strip()[:80].replace("\n", " ")
            sys.__stdout__.write(f"    → {summary}\n")
            sys.__stdout__.flush()
            return result
        return wrapped


# ── 执行引擎 ──

def _run(query: str) -> str:
    """执行查询，返回最终答案"""
    # 分类
    try:
        task_type = _import("intent.classifier", "IntentClassifier")().classify(query)
    except Exception:
        task_type = ""

    # 权限
    HITL = _import("core.human_in_the_loop", "HumanInTheLoop")
    PW = _import("integration.agent_wrapper", "PermissionWrapper")
    hitl = HITL(ask_fn=_hitl_ask)
    perm = PW(hitl=hitl)

    # 工具监控（实时显示） + 静默执行
    monitor = ToolMonitor()
    import src.handwritten_react_agent.react_loop as rl
    rl.execute_tool_call = monitor.wrap(perm.wrap(rl.execute_tool_call))

    # 多轮上下文
    ctx = query
    if _history:
        ctx = "[历史]\n" + "\n".join(
            f"Q: {m['c'][:150]}" for m in _history[-3:] if m['r']=='u'
        ) + "\n[现在]\n" + query

    # 执行（抑制 Agent 内部 print，工具调用通过 sys.__stdout__ 透出）
    f = StringIO()
    with redirect_stdout(f):
        try:
            result = rl.react_loop(ctx, max_steps=10)
        except Exception as e:
            return f"错误: {e}"

    _last_traj[0] = _recent_traj()
    _history.append({"r": "u", "c": query})
    if result:
        _history.append({"r": "a", "c": result[:500]})
    if len(_history) > 20:
        _history[:] = _history[-20:]

    return result or "（无输出）"


def _hitl_ask(msg: str, choices: list[str]) -> str:
    _console.print(Panel(msg, border_style="yellow", title="确认", box=box.SIMPLE))
    _console.print("  1:允许  2:本次会话  3:拒绝")
    return Prompt.ask("", choices=choices, default="1")


# ── 命令 ──

def _handle(cmd: str) -> bool:
    parts = cmd.strip().split(maxsplit=1)
    c = parts[0].lower()
    if c in ("/exit", "/quit"):
        return True
    elif c == "/clear":
        os.system("cls" if os.name == "nt" else "clear")
    elif c == "/replay":
        _cmd_replay()
    elif c == "/config":
        _cmd_config()
    elif c == "/provider" and len(parts) > 1:
        os.environ["LLM_PROVIDER"] = parts[1]
        _console.print(f"provider: {parts[1]}")
    elif c == "/history":
        if _history:
            for m in _history[-6:]:
                prefix = "你" if m["r"] == "u" else "答"
                _console.print(f"  {prefix}: {m['c'][:100]}")
        else:
            _console.print("（空）")
    return False


def _cmd_replay():
    path = _last_traj[0]
    if not path or not os.path.exists(path):
        _console.print("暂无轨迹")
        return
    try:
        with open(path) as f:
            d = json.load(f)
        _console.print(f"[bold]{d.get('query','')[:100]}[/]")
        for s in d.get("steps", []):
            a = s.get("action", {}) or {}
            if a.get("name"):
                _console.print(f"  [yellow]●[/] {a['name']}")
            if s.get("observation"):
                _console.print(f"    [dim]{s['observation'][:100]}[/]")
            if s.get("thought"):
                t = s["thought"][:80].replace("\n", " ")
                _console.print(f"    [dim]💭 {t}[/]")
    except Exception as e:
        _console.print(f"错误: {e}")


def _cmd_config():
    try:
        from src.handwritten_react_agent.llm import list_providers
        ps = list_providers()
        cur = os.environ.get("LLM_PROVIDER", "default")
        _console.print(f"provider: [cyan]{cur}[/] ({', '.join(ps)})")
    except Exception:
        _console.print("无法读取配置")


# ══════════════════════════════════════════════
#  Shell
# ══════════════════════════════════════════════

app = typer.Typer(name="hagent", no_args_is_help=True)


@app.command()
def shell(provider: str = ""):
    """交互模式 — 仿 Claude Code 风格"""
    if provider:
        os.environ["LLM_PROVIDER"] = provider

    while True:
        try:
            q = Prompt.ask(">")
        except (EOFError, KeyboardInterrupt):
            break

        q = q.strip()
        if not q:
            continue
        if q.startswith("/"):
            if _handle(q):
                break
            continue

        # 执行并显示答案
        answer = _run(q)
        if answer:
            _console.print(Markdown(answer))


# ══════════════════════════════════════════════
#  Run
# ══════════════════════════════════════════════

@app.command()
def run(query: str = typer.Argument(...)):
    """单次执行"""
    answer = _run(query)
    if answer:
        _console.print(Markdown(answer))


# ══════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════

@app.command()
def config():
    _cmd_config()


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        shell()
    else:
        app()
