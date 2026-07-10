"""hagent — handwritten-react-agent CLI

极简交互风格（类似 Claude Code / Codex）。
"""
from __future__ import annotations
import os, sys, importlib, json, glob

_base = os.path.dirname(os.path.abspath(__file__))
for p in [_base, os.path.join(_base, "src"),
          os.path.join(_base, "experiments", "eval-engine")]:
    sys.path.insert(0, p)

from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
import typer

_console = Console()
_last_traj_file = [""]
_session_history: list[dict] = []  # 多轮对话历史 [{role, content}, ...]
_MAX_HISTORY = 10


def _import(mod: str, name: str):
    return getattr(importlib.import_module(mod), name)


app = typer.Typer(name="hagent", no_args_is_help=True)


# ══════════════════════════════════════════════
#  交互模式
# ══════════════════════════════════════════════

@app.command()
def shell(provider: str = ""):
    """启动交互模式"""
    if provider:
        os.environ["LLM_PROVIDER"] = provider
    pname = os.environ.get("LLM_PROVIDER", "default")
    _console.print(f"hagent [dim]({pname})[/dim]  — /help")
    _console.print()

    while True:
        try:
            query = Prompt.ask(">")
        except (EOFError, KeyboardInterrupt):
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
    args = cmd.split(maxsplit=1)

    if c in ("/exit", "/quit"):
        raise SystemExit(0)
    elif c == "/help":
        _console.print("  /exit        退出")
        _console.print("  /clear       清屏")
        _console.print("  /config      查看配置")
        _console.print("  /memory      查看对话历史")
        _console.print("  /provider    切换 Provider（例：/provider openai）")
        _console.print("  /replay      查看上一条轨迹")
        _console.print("  /eval        评测上一条回答")
        _console.print("  /orch        多 Agent 编排（例：/orch 查时间和算50*30）")
    elif c == "/clear":
        os.system("cls" if os.name == "nt" else "clear")
    elif c == "/config":
        _show_config()
    elif c == "/provider" and len(args) > 1:
        os.environ["LLM_PROVIDER"] = args[1]
        _console.print(f"→ provider: {args[1]}")
    elif c == "/replay":
        _cmd_replay()
    elif c == "/eval":
        _cmd_eval_last()
    elif c == "/memory":
        if _session_history:
            _console.print(f"[dim]对话历史 ({len(_session_history)} 条):[/]")
            for m in _session_history[-5:]:
                role = "你" if m["role"] == "user" else "Agent"
                _console.print(f"  {role}: {m['content'][:100]}")
        else:
            _console.print("[dim]当前会话无历史[/]")
    elif c in ("/orch", "/orchestrator") and len(args) > 1:
        _execute_orchestrator(args[1])
    else:
        _console.print(f"[red]? {cmd}[/]")


def _execute(query: str):
    """执行查询"""
    # ── 分类 ──
    try:
        task_type = _import("intent.classifier", "IntentClassifier")().classify(query)
    except Exception:
        task_type = ""

    # ── 权限 ──
    HITL = _import("core.human_in_the_loop", "HumanInTheLoop")
    PW = _import("integration.agent_wrapper", "PermissionWrapper")
    hitl = HITL(ask_fn=_hitl_ask)
    perm = PW(hitl=hitl)
    from src.handwritten_react_agent.react_loop import execute_tool_call as orig
    import src.handwritten_react_agent.react_loop as rl_mod
    rl_mod.execute_tool_call = perm.wrap(orig)

    _console.print(f"[dim]{task_type}[/]" if task_type else "")

    # ── 多轮对话：注入历史上下文 ──
    if _session_history:
        context_summary = "\n".join(
            f"{'你' if m['role']=='user' else '我'}: {m['content'][:200]}"
            for m in _session_history[-3:]  # 最近 3 轮
        )
        query_with_context = (
            f"【对话历史】\n{context_summary}\n\n"
            f"【当前问题】\n{query}"
        )
    else:
        query_with_context = query

    # ── 执行 ──
    try:
        result = rl_mod.react_loop(query_with_context, max_steps=10)
    except Exception as e:
        _console.print(f"[red]✗ {e}[/]")
        return

    # ── 轨迹文件路径（供 /replay 使用）───
    _last_traj_file[0] = _find_latest_traj()

    # ── 复盘 ──
    report = perm.watch.summary()
    if report.has_issues:
        for ev in report.events:
            icon = {"tool_blocked": "🔒", "tool_error": "✗",
                    "approach_switch": "→", "search_fail": "⚠",
                    "limit_hit": "⛔"}.get(ev.type, "•")
            _console.print(f"  {icon} [dim]{ev.description}[/]")
        if hitl.check_direction("重试失败路径", details=report.one_liner):
            _console.print("↻ 重试...")
            _execute(query)

    # ── 保存对话历史 ──
    _session_history.append({"role": "user", "content": query})
    if result:
        _session_history.append({"role": "assistant", "content": result[:500]})
    if len(_session_history) > _MAX_HISTORY * 2:
        _session_history[:-(_MAX_HISTORY * 2)] = []

    # ── 快速 Eval 评分 ──
    if task_type != "functional_test" and result:
        _quick_eval(query, result)


def _find_latest_traj() -> str:
    """找最新轨迹文件"""
    for d in [os.path.join(_base, "src", "handwritten_react_agent", "trajectories"),
              os.path.join(_base, "trajectories")]:
        if os.path.exists(d):
            files = sorted(glob.glob(os.path.join(d, "*.json")), reverse=True)
            if files:
                return files[0]
    return ""


# ══════════════════════════════════════════════
#  Eval 评分
# ══════════════════════════════════════════════

def _quick_eval(query: str, response: str):
    """快速评测"""
    try:
        report_mod = _import("report", "run_eval")
        fmt = _import("report", "format_text")
        report = report_mod(response=response, context=query)
        for line in fmt(report).split("\n")[:4]:
            if "总体评分" in line or "评测维度" in line or "通过" in line:
                _console.print(f"  [dim]{line.strip()}[/]")
    except Exception:
        pass


def _cmd_eval_last():
    """评测上一次回答"""
    from src.handwritten_react_agent.llm import LLM_DEFAULT
    try:
        _console.print("[dim]请输入要评测的回答（多行输入，空行结束）:[/]")
        lines = []
        while True:
            line = input()
            if not line:
                break
            lines.append(line)
        response = "\n".join(lines)
        if not response:
            return
        query = Prompt.ask("参考上下文（可选）", default="")
        _quick_eval(query, response)
    except Exception:
        pass


# ══════════════════════════════════════════════
#  Replay
# ══════════════════════════════════════════════

def _cmd_replay():
    """查看上一条轨迹"""
    path = _last_traj_file[0]
    if not path or not os.path.exists(path):
        _console.print("[dim]没有最近的轨迹[/]")
        return
    try:
        import json
        with open(path) as f:
            d = json.load(f)
        _console.print(f"[dim]会话: {d.get('session_id','')}[/]")
        _console.print(f"[dim]查询: {d.get('query','')[:100]}[/]")
        _console.print(f"[dim]步骤: {d.get('total_steps',0)} 步[/]")
        for s in d.get("steps", []):
            act = s.get("action", {}) or {}
            name = act.get("name", "")
            args = str(act.get("arguments", act.get("args", "")))[:60]
            obs = (s.get("observation") or "")[:80]
            if name:
                _console.print(f"  [dim]Step {s['step']}: {name}({args})[/]")
            elif obs:
                _console.print(f"  [dim]Step {s['step']}: → {obs}[/]")
    except Exception as e:
        _console.print(f"[red]✗ {e}[/]")


# ══════════════════════════════════════════════
#  HITL
# ══════════════════════════════════════════════

# ══════════════════════════════════════════════
#  Orchestrator
# ══════════════════════════════════════════════

def _execute_orchestrator(query: str):
    """多 Agent 编排执行"""
    _console.print("[dim]orchestrator[/]")
    try:
        from src.handwritten_react_agent.orchestrator import Orchestrator
        from src.handwritten_react_agent.react_loop import react_loop, call_llm
        from src.handwritten_react_agent.tools import TOOL_DEFINITIONS

        orch = Orchestrator(
            call_llm_func=call_llm,
            react_loop_func=react_loop,
            tool_definitions=TOOL_DEFINITIONS,
        )
        tasks = orch.plan(query)
        if not tasks:
            _console.print("[red]Orchestrator 无法分解任务[/]")
            return

        result = orch.execute(query)
        _console.print(f"\n[green]{result[:500]}[/]" if result else "[red]无结果[/]")

        _session_history.append({"role": "user", "content": f"[Orchestrator] {query}"})
        if result:
            _session_history.append({"role": "assistant", "content": result[:500]})
    except Exception as e:
        _console.print(f"[red]✗ {e}[/]")
        import traceback
        traceback.print_exc()


# ══════════════════════════════════════════════
#  HITL
# ══════════════════════════════════════════════

def _hitl_ask(msg: str, choices: list[str]) -> str:
    _console.print(f"[yellow]? {msg}[/]")
    _console.print("  1:允许 2:本次会话 3:拒绝")
    return Prompt.ask("", choices=choices, default="1")


# ══════════════════════════════════════════════
#  Run
# ══════════════════════════════════════════════

@app.command()
def run(query: str = typer.Argument(...),
        provider: str = typer.Option("", "--provider", "-p")):
    """单次执行 Agent"""
    if provider:
        os.environ["LLM_PROVIDER"] = provider
    _execute(query)


# ══════════════════════════════════════════════
#  Config
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
