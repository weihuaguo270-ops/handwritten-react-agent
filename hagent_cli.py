"""hagent — handwritten-react-agent CLI

设计原则：
  1. 无 banner、无 loading、无中间过程
  2. 输入问题 → 得到答案
  3. 想看过程有 /replay
  4. 命令与提示符风格统一
"""
from __future__ import annotations
import os
import sys
import json
import glob
from contextlib import redirect_stdout
from io import StringIO

_base = os.path.dirname(os.path.abspath(__file__))
os.chdir(_base)
os.environ.pop("DEEPSEEK_API_KEY", None)
for p in [_base, os.path.join(_base, "src"),
          os.path.join(_base, "experiments", "eval-engine")]:
    sys.path.insert(0, p)

from rich.console import Console
from rich.prompt import Prompt
import typer

_console = Console()
_last_traj = [""]
_history: list[dict] = []


# ── 预加载 RAG（避免首次输出多余信息）──
with redirect_stdout(StringIO()):
    try:
        import src.handwritten_react_agent.react_loop  # noqa: F401
    except Exception:
        pass


# ── 工具 ──

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


# ── 执行引擎 ──

def _run(query: str) -> str:
    """执行查询，返回最终答案。不打印任何中间信息。"""
    # 分类
    try:
        Classifier = _import("intent.classifier", "IntentClassifier")
    except Exception:
        Classifier = None

    # 权限包装
    HITL = _import("core.human_in_the_loop", "HumanInTheLoop")
    PW = _import("integration.agent_wrapper", "PermissionWrapper")
    hitl = HITL(ask_fn=_hitl_ask)
    perm = PW(hitl=hitl)
    import src.handwritten_react_agent.react_loop as rl
    rl.execute_tool_call = perm.wrap(rl.execute_tool_call)

    # 多轮上下文
    ctx = query
    if _history:
        ctx = "【历史】\n" + "\n".join(
            f"{'Q' if m['r']=='u' else 'A'}: {m['c'][:150]}"
            for m in _history[-3:]
        ) + "\n【现在】\n" + query

    # 执行（静默）
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
    _console.print(f"[yellow]? {msg}[/]")
    _console.print("  1:允许 2:本次 3:拒绝")
    return Prompt.ask("", choices=choices, default="1")


# ── 命令处理 ──

def _handle(cmd: str) -> bool:
    """处理命令。返回 True 表示退出。"""
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
        _console.print(parts[1])
    return False


def _cmd_replay():
    path = _last_traj[0]
    if not path or not os.path.exists(path):
        _console.print("暂无轨迹")
        return
    try:
        with open(path) as f:
            d = json.load(f)
        _console.print(f"[dim]{d.get('query','')[:80]}[/]")
        for s in d.get("steps", []):
            a = s.get("action", {}) or {}
            if a.get("name"):
                _console.print(f"  {a['name']}")
            if s.get("observation"):
                o = s["observation"][:80].replace("\n", " ")
                _console.print(f"  → {o}")
    except Exception:
        _console.print("读取失败")


def _cmd_config():
    try:
        from src.handwritten_react_agent.llm import list_providers
        ps = list_providers()
        cur = os.environ.get("LLM_PROVIDER", "default")
        _console.print(f"provider: {cur} ({', '.join(ps)})")
        key = os.environ.get("DEEPSEEK_API_KEY", "")
        _console.print(f"api-key: {'✅' if key else '❌'}")
    except Exception:
        _console.print("无法读取配置")


# ══════════════════════════════════════════════
#  Shell
# ══════════════════════════════════════════════

app = typer.Typer(name="hagent", no_args_is_help=True)


@app.command()
def shell(provider: str = ""):
    """交互模式"""
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

        _console.print(_run(q))


# ══════════════════════════════════════════════
#  Run
# ══════════════════════════════════════════════

@app.command()
def run(query: str = typer.Argument(...)):
    """单次执行"""
    _console.print(_run(query))


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
