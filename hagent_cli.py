"""hagent — handwritten-react-agent CLI"""
from __future__ import annotations
import os, sys, json, glob
from contextlib import redirect_stdout
from io import StringIO

_base = os.path.dirname(os.path.abspath(__file__))
os.chdir(_base); os.environ.pop("DEEPSEEK_API_KEY", None)
for p in [_base, os.path.join(_base, "src"),
          os.path.join(_base, "experiments", "eval-engine")]:
    sys.path.insert(0, p)

from rich.console import Console
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.text import Text
from rich import box
import typer

_console = Console()
_last_traj = [""]
_history: list[dict] = []

with redirect_stdout(StringIO()):
    try:
        import src.handwritten_react_agent.react_loop
    except Exception:
        pass


def _import(mod: str, name: str):
    return getattr(__import__(mod, fromlist=[name]), name)


def _recent_traj() -> str:
    for d in [os.path.join(_base, "src", "handwritten_react_agent", "trajectories"),
              os.path.join(_base, "trajectories")]:
        if os.path.exists(d):
            fs = sorted(glob.glob(os.path.join(d, "*.json")), reverse=True)
            if fs: return fs[0]
    return ""


# ── 工具监控 ──

class ToolMonitor:
    def __init__(self): self._count = 0
    def wrap(self, original):
        def wrapped(tc):
            self._count += 1
            n = tc.get("function", {}).get("name", "")
            a = tc.get("function", {}).get("arguments", "{}")
            try:
                args = json.loads(a) if isinstance(a, str) else a
            except json.JSONDecodeError:
                args = {}
            parts = []
            if isinstance(args, dict):
                for k, v in list(args.items())[:2]:
                    vs = str(v)
                    if len(vs) > 30: vs = vs[:30] + "..."
                    parts.append(f"{k}={vs}")
            extra = " · " + ", ".join(parts) if parts else ""
            sys.__stdout__.write(f"\033[33m● {n}{extra}\033[0m\n")
            sys.__stdout__.flush()
            r = original(tc)
            s = r.strip()[:80].replace("\n", " ")
            clr = "\033[31m" if "error" in r[:20].lower() else "\033[32m"
            sys.__stdout__.write(f"{clr}  {s}\033[0m\n")
            sys.__stdout__.flush()
            return r
        return wrapped


def _run(q: str) -> str:
    try:
        task_type = _import("intent.classifier", "IntentClassifier")().classify(q)
    except Exception:
        task_type = ""
    HITL = _import("core.human_in_the_loop", "HumanInTheLoop")
    PW = _import("integration.agent_wrapper", "PermissionWrapper")
    hitl = HITL(ask_fn=_hitl_ask)
    perm = PW(hitl=hitl)
    m = ToolMonitor()
    # CLI 模式：注入简洁角色（覆盖 prompt 中默认的客套风格）
    import src.handwritten_react_agent.react_loop as rl
    rl.execute_tool_call = m.wrap(perm.wrap(rl.execute_tool_call))

    # 在查询中注入风格指令，让 Agent 回答简洁直接
    style_prefix = "[指令] 你是CLI工具，不是聊天机器人。回答必须简洁：直接给答案，不说客套话，不解释自己是什么。"

    ctx = q
    if _history:
        ctx = "[历史]\n" + "\n".join(
            f"Q: {h['c'][:150]}" for h in _history[-3:] if h['r']=='u'
        ) + "\n[现在]\n" + q

    # 注入 CLI 风格
    ctx = style_prefix + "\n\n问题: " + ctx

    f = StringIO()
    with redirect_stdout(f):
        try:
            r = rl.react_loop(ctx, max_steps=10)
        except Exception as e:
            return f"错误: {e}"
    _last_traj[0] = _recent_traj()
    _history.append({"r": "u", "c": q})
    if r: _history.append({"r": "a", "c": r[:500]})
    if len(_history) > 20: _history[:] = _history[-20:]
    return r or ""


def _hitl_ask(msg: str, choices: list[str]) -> str:
    sys.__stdout__.write(f"\033[33m? {msg}\033[0m\n")
    _console.print("  1:允许  2:本次  3:拒绝")
    return Prompt.ask("", choices=choices, default="1")


# ── 命令 ──

def _handle(c: str) -> bool:
    p = c.strip().split(maxsplit=1)
    x = p[0].lower()
    if x in ("/exit", "/quit"): return True
    if x == "/clear": os.system("cls" if os.name == "nt" else "clear")
    elif x == "/model" and len(p) > 1:
        os.environ["LLM_MODEL"] = p[1]
        # 清除 LLM 模块缓存，下次调用读新模型
        import src.handwritten_react_agent.llm as llm_mod
        llm_mod._CONFIG = None
        llm_mod.LLM_DEFAULT = None
        _console.print(f"model: {p[1]}")
    elif x == "/replay": _replay()
    elif x == "/provider" and len(p) > 1:
        os.environ["LLM_PROVIDER"] = p[1]
        _console.print(p[1])
    return False


def _replay():
    path = _last_traj[0]
    if not path or not os.path.exists(path): return
    try:
        with open(path) as f:
            d = json.load(f)
        for s in d.get("steps", []):
            a = s.get("action", {}) or {}
            if a.get("name"):
                _console.print(f"  [yellow]●[/] {a['name']}")
                if s.get("observation"):
                    _console.print(f"    [dim]{s['observation'][:100]}[/]")
    except Exception:
        pass


# ══════════════════════════════════════════════
#  Shell
# ══════════════════════════════════════════════

app = typer.Typer(name="agent", no_args_is_help=True)


@app.command()
def shell(provider: str = ""):
    if provider: os.environ["LLM_PROVIDER"] = provider
    pname = os.environ.get("LLM_PROVIDER", "default")
    dash = "─" * 50

    while True:
        # Header — 显示实际模型名而非 provider 标签
        cur = "unknown"
        try:
            from src.handwritten_react_agent.llm import LLM
            cur = LLM().model
        except Exception:
            cur = os.environ.get("LLM_PROVIDER", "default")
        _console.print(f"[bold]handwritten-react-agent[/] [dim]({cur})[/]")
        _console.print(dash)

        try:
            q = Prompt.ask("❯")
        except (EOFError, KeyboardInterrupt):
            break
        q = q.strip()
        if not q: continue
        if q.startswith("/"):
            if _handle(q): break
            continue

        a = _run(q)
        _console.print(dash)
        if a:
            _console.print(Markdown(a))
        else:
            _console.print("[dim]（无输出）[/]")

        # Shortcuts hint
        _console.print(f"\n[dim]/exit  /replay  /provider[/]")


# ══════════════════════════════════════════════
#  Run
# ══════════════════════════════════════════════

@app.command()
def run(query: str = typer.Argument(...)):
    _run(query)


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        shell()
    else:
        app()
