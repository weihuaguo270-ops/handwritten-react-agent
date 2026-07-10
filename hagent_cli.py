"""hagent — handwritten-react-agent CLI

极简交互风格（类似 Claude Code / Codex）。
Agent 的思考过程、工具调用、输出自然融入对话。
"""
from __future__ import annotations
import os
import sys
import importlib

_base = os.path.dirname(os.path.abspath(__file__))
for p in [_base, os.path.join(_base, "src"),
          os.path.join(_base, "experiments", "eval-engine")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.style import Style
import typer

_console = Console()
_err_console = Console(stderr=True)


def _import(mod: str, name: str):
    return getattr(importlib.import_module(mod), name)


app = typer.Typer(name="hagent", no_args_is_help=True)


# ══════════════════════════════════════════════
#  shell — 交互模式
# ══════════════════════════════════════════════

@app.command()
def shell(
    provider: str = "",
):
    """启动交互模式（类似 Claude Code）"""
    if provider:
        os.environ["LLM_PROVIDER"] = provider
    provider_name = os.environ.get("LLM_PROVIDER", "default")

    _console.print(f"hagent [dim]({provider_name})[/dim]  — 输入 /help 查看命令")
    _console.print()

    while True:
        try:
            query = Prompt.ask(">")
        except (EOFError, KeyboardInterrupt):
            _console.print()
            break

        query = query.strip()
        if not query:
            continue
        if query.startswith("/"):
            _handle_cmd(query)
            continue

        _execute(query)
        _console.print()


def _handle_cmd(cmd: str):
    c = cmd.split()[0].lower()
    if c in ("/exit", "/quit"):
        raise SystemExit(0)
    elif c == "/help":
        _console.print("  /exit  退出")
        _console.print("  /clear 清屏")
        _console.print("  /config 配置")
    elif c == "/clear":
        os.system("cls" if os.name == "nt" else "clear")
    elif c == "/config":
        _show_config()
    else:
        _console.print(f"[red]? 未知命令: {c}[/]")


def _execute(query: str):
    """执行查询，类似 Claude Code 的对话流"""
    # ── 意图分类 ──
    try:
        Classifier = _import("intent.classifier", "IntentClassifier")
        task_type = Classifier().classify(query)
    except Exception:
        task_type = ""

    # ── 权限包装 ──
    HITL = _import("core.human_in_the_loop", "HumanInTheLoop")
    PW = _import("integration.agent_wrapper", "PermissionWrapper")
    hitl = HITL(ask_fn=_hitl_ask)
    perm = PW(hitl=hitl)
    from src.handwritten_react_agent.react_loop import execute_tool_call as orig
    import src.handwritten_react_agent.react_loop as rl_mod
    rl_mod.execute_tool_call = perm.wrap(orig)

    # ── 执行（实时显示 Agent 输出）───
    _console.print(f"[dim]● {task_type}[/dim]" if task_type else "")

    # 重定向 print 到 rich，让 Agent 的中间输出也显示在对话中
    class _PrintRedirect:
        def write(self, s):
            if s.strip():
                _console.print(s.rstrip())
        def flush(self):
            pass

    old_stdout = sys.stdout
    sys.stdout = _PrintRedirect()

    try:
        result = rl_mod.react_loop(query, max_steps=10)
    except Exception as e:
        sys.stdout = old_stdout
        _console.print(f"[red]✗ {e}[/]")
        return

    sys.stdout = old_stdout

    # ── 复盘 ──
    report = perm.watch.summary()
    if report.has_issues:
        for ev in report.events:
            icon = {"tool_blocked": "🔒", "tool_error": "✗",
                    "approach_switch": "→", "search_fail": "⚠",
                    "limit_hit": "⛔", "complete": ""}.get(ev.type, "•")
            _console.print(f"  {icon} [dim]{ev.description}[/]")
        if hitl.check_direction("重试失败路径", details=report.one_liner):
            _console.print("↻ 重试...")
            _execute(query)


def _hitl_ask(msg: str, choices: list[str]) -> str:
    """HITL 确认（极简风格）"""
    _console.print(f"[yellow]? {msg}[/]")
    labels = {"1": "允许", "2": "本次会话", "3": "拒绝"}
    _console.print("  " + " ".join(
        f"[dim]{k}[/]:{v}" for k, v in labels.items()
    ))
    return Prompt.ask("", choices=choices, default="1")


# ══════════════════════════════════════════════
#  run — 单次执行
# ══════════════════════════════════════════════

@app.command()
def run(
    query: str = typer.Argument(...),
    provider: str = typer.Option("", "--provider", "-p"),
):
    """单次执行 Agent"""
    if provider:
        os.environ["LLM_PROVIDER"] = provider
    _execute(query)


# ══════════════════════════════════════════════
#  config — 配置
# ══════════════════════════════════════════════

@app.command()
def config():
    """查看配置"""
    _show_config()


def _show_config():
    from src.handwritten_react_agent.llm import list_providers
    _console.print(f"provider: [cyan]{os.environ.get('LLM_PROVIDER', 'default')}[/]")
    _console.print(f"可用: {', '.join(list_providers())}")
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    _console.print(f"api-key: {'✅ 已配置' if key else '❌ 未配置'}")
    Perm = _import("core.permissions", "TOOL_PERMISSIONS")
    Arg = _import("core.permissions", "ARG_RULES")
    levels = {}
    for _, v in Perm.items():
        levels[v.value] = levels.get(v.value, 0) + 1
    _console.print(f"权限规则: {levels}")
    _console.print(f"参数规则: {len(Arg)} 条")


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        shell()
    else:
        app()
