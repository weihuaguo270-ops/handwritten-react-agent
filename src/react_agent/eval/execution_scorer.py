"""execution_scorer — Execution 任务集打分（offline_tools + agent）

- offline_tools: 不经 LLM，直接执行预置工具脚本
- agent: 经 react_loop 子进程（需 DEEPSEEK_API_KEY 等），按答案/工具验收

用法：
    from react_agent.eval.execution_scorer import run_execution_suite
    report = run_execution_suite(modes=["offline_tools"])
    report = run_execution_suite(modes=["agent"])  # live
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from react_agent.tools import TOOL_REGISTRY

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_EXECUTION_DATASET = os.path.join(_EVAL_DIR, "execution_dataset.json")

AgentRunner = Callable[..., tuple]  # (stdout, traj, exit_code, duration)


def load_execution_dataset(path: Optional[str] = None) -> list[dict]:
    filepath = path or DEFAULT_EXECUTION_DATASET
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"execution dataset must be a list: {filepath}")
    return data


def _check_expectation(result: str, step: dict) -> tuple[bool, str]:
    """对照 expect_* 字段判定单步是否通过。"""
    if "expect_equals" in step:
        exp = str(step["expect_equals"])
        if result.strip() == exp:
            return True, "equals"
        return False, f"expected equals {exp!r}, got {result.strip()!r}"

    if "expect_contains" in step:
        needle = str(step["expect_contains"])
        if needle in result:
            return True, "contains"
        return False, f"missing {needle!r} in {result[:200]!r}"

    if "expect_regex" in step:
        pat = str(step["expect_regex"])
        if re.search(pat, result.strip()):
            return True, "regex"
        return False, f"regex {pat!r} not matched: {result.strip()!r}"

    low = result.lower()
    if '"error"' in low or result.startswith("[错误]") or "错误：" in result:
        return False, f"tool error-like output: {result[:200]!r}"
    return True, "no_explicit_expect"


def execute_tool_step(tool: str, arguments: dict) -> str:
    if tool not in TOOL_REGISTRY:
        return json.dumps({"error": f"未知工具: {tool}"})
    try:
        return str(TOOL_REGISTRY[tool](**(arguments or {})))
    except Exception as e:
        return json.dumps({"error": f"执行错误: {e}"})


def _collect_tools(stdout: str, trajectory: Optional[dict]) -> set[str]:
    tools = set(re.findall(r"\[调工具\] (\w+)\(", stdout or ""))
    if trajectory:
        for step in trajectory.get("steps") or []:
            action = step.get("action") or {}
            if action.get("name"):
                tools.add(action["name"])
            for a in step.get("actions") or []:
                if a.get("name"):
                    tools.add(a["name"])
    return tools


def _final_answer_text(stdout: str, trajectory: Optional[dict]) -> str:
    parts = [stdout or ""]
    if trajectory:
        parts.append(str(trajectory.get("final_answer") or ""))
    return "\n".join(parts)


def score_offline_task(task: dict) -> dict[str, Any]:
    task_id = str(task.get("id") or "unknown")
    step_results = []
    all_ok = True
    t0 = time.time()
    for i, step in enumerate(task.get("steps") or []):
        tool = step.get("tool", "")
        args = step.get("arguments") or {}
        raw = execute_tool_step(tool, args)
        ok, reason = _check_expectation(raw, step)
        step_results.append({
            "index": i + 1,
            "tool": tool,
            "arguments": args,
            "result": raw[:500],
            "passed": ok,
            "reason": reason,
        })
        if not ok:
            all_ok = False
    duration = round(time.time() - t0, 3)
    return {
        "id": task_id,
        "name": task.get("name", ""),
        "tags": list(task.get("tags") or []),
        "mode": "offline_tools",
        "passed": all_ok and bool(step_results),
        "skipped": False,
        "duration_seconds": duration,
        "steps": step_results,
        "reason": "all steps ok" if all_ok and step_results else "step failure",
    }


def score_agent_task(
    task: dict,
    *,
    agent_runner: Optional[AgentRunner] = None,
) -> dict[str, Any]:
    """跑 agent 子进程并做执行验收打分。"""
    task_id = str(task.get("id") or "unknown")
    question = (task.get("question") or task.get("query") or "").strip()
    if not question:
        return {
            "id": task_id,
            "name": task.get("name", ""),
            "tags": list(task.get("tags") or []),
            "mode": "agent",
            "passed": False,
            "skipped": True,
            "reason": "missing question",
            "steps": [],
        }

    if agent_runner is None:
        from react_agent.eval.runner import run_single_case
        agent_runner = run_single_case

    timeout = int(task.get("timeout") or 90)
    max_steps = task.get("max_steps")
    t0 = time.time()
    stdout, trajectory, exit_code, duration = agent_runner(
        question,
        timeout=timeout,
        max_steps=max_steps,
    )
    # runner may return its own duration; keep wall clock as backup
    if not duration:
        duration = round(time.time() - t0, 3)

    text = _final_answer_text(stdout, trajectory)
    tools = _collect_tools(stdout, trajectory)
    checks: list[dict[str, Any]] = []
    ok = True

    if exit_code == -1:
        ok = False
        checks.append({"name": "timeout", "passed": False, "reason": "subprocess timeout"})
    else:
        checks.append({"name": "timeout", "passed": True, "reason": f"exit={exit_code}"})

    expected_tools = list(task.get("expected_tools") or [])
    if expected_tools:
        hit = tools & set(expected_tools)
        tool_ok = bool(hit)
        checks.append({
            "name": "tools",
            "passed": tool_ok,
            "reason": f"called={sorted(tools)} expected={expected_tools}",
        })
        ok = ok and tool_ok

    expected_answer = task.get("expected_answer")
    if expected_answer is not None:
        needle = str(expected_answer)
        ans_ok = needle in text
        checks.append({
            "name": "expected_answer",
            "passed": ans_ok,
            "reason": f"need {needle!r}",
        })
        ok = ok and ans_ok

    for needle in task.get("must_contain") or []:
        c_ok = str(needle) in text
        checks.append({
            "name": "must_contain",
            "passed": c_ok,
            "reason": f"need {needle!r}",
        })
        ok = ok and c_ok

    any_list = list(task.get("must_contain_any") or [])
    if any_list:
        any_ok = any(str(n) in text for n in any_list)
        checks.append({
            "name": "must_contain_any",
            "passed": any_ok,
            "reason": f"need any of {any_list}",
        })
        ok = ok and any_ok

    # 至少要有非空最终答案/实质输出
    has_answer = False
    if trajectory and str(trajectory.get("final_answer") or "").strip():
        has_answer = True
    elif re.search(r"FINAL ANSWER:\s*\S", stdout or "", re.I):
        has_answer = True
    elif len((stdout or "").strip()) > 20 and exit_code == 0:
        has_answer = True
    checks.append({
        "name": "has_answer",
        "passed": has_answer,
        "reason": "final_answer present" if has_answer else "empty answer",
    })
    ok = ok and has_answer

    self_repair = "[Harness自修]" in (stdout or "")
    return {
        "id": task_id,
        "name": task.get("name", ""),
        "tags": list(task.get("tags") or []),
        "mode": "agent",
        "passed": ok,
        "skipped": False,
        "duration_seconds": duration,
        "exit_code": exit_code,
        "tools_called": sorted(tools),
        "self_repair_seen": self_repair,
        "checks": checks,
        "stdout_preview": (stdout or "")[:400],
        "final_answer_preview": str(
            (trajectory or {}).get("final_answer") or ""
        )[:300],
        "reason": "agent outcome ok" if ok else "agent outcome failed",
    }


def score_task(
    task: dict,
    *,
    agent_runner: Optional[AgentRunner] = None,
) -> dict[str, Any]:
    """评测单条 execution 任务。"""
    task_id = str(task.get("id") or "unknown")
    mode = task.get("mode", "offline_tools")
    if mode == "offline_tools":
        return score_offline_task(task)
    if mode == "agent":
        return score_agent_task(task, agent_runner=agent_runner)
    return {
        "id": task_id,
        "name": task.get("name", ""),
        "tags": list(task.get("tags") or []),
        "mode": mode,
        "passed": False,
        "skipped": True,
        "reason": f"unsupported mode: {mode}",
        "steps": [],
    }


def run_execution_suite(
    path: Optional[str] = None,
    *,
    only_ids: Optional[set[str]] = None,
    modes: Optional[list[str]] = None,
    agent_runner: Optional[AgentRunner] = None,
) -> dict[str, Any]:
    """跑 execution 集，返回可归档 JSON 报告。

    modes: 默认 ["offline_tools"]。传 ["agent"] 或 ["offline_tools","agent"]。
    """
    wanted = set(modes or ["offline_tools"])
    tasks = load_execution_dataset(path)
    if only_ids:
        tasks = [t for t in tasks if str(t.get("id")) in only_ids]
    tasks = [t for t in tasks if (t.get("mode") or "offline_tools") in wanted]

    results = []
    for task in tasks:
        results.append(score_task(task, agent_runner=agent_runner))

    scored = [r for r in results if not r.get("skipped")]
    passed = sum(1 for r in scored if r["passed"])
    total = len(scored)
    rate = round(100.0 * passed / total, 1) if total else 0.0

    by_tag: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "passed": 0})
    by_mode: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "passed": 0})
    for r in scored:
        mode = r.get("mode") or "unknown"
        by_mode[mode]["total"] += 1
        if r["passed"]:
            by_mode[mode]["passed"] += 1
        for tag in r.get("tags") or ["untagged"]:
            by_tag[tag]["total"] += 1
            if r["passed"]:
                by_tag[tag]["passed"] += 1

    def _rate_map(d: dict) -> dict:
        return {
            k: {
                **v,
                "pass_rate": round(100.0 * v["passed"] / v["total"], 1) if v["total"] else 0.0,
            }
            for k, v in sorted(d.items())
        }

    report = {
        "report_id": f"execution_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": os.path.basename(path or DEFAULT_EXECUTION_DATASET),
        "modes": sorted(wanted),
        "summary": {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": rate,
            "skipped": sum(1 for r in results if r.get("skipped")),
        },
        "by_mode": _rate_map(by_mode),
        "by_tag": _rate_map(by_tag),
        "results": results,
    }
    return report


def report_to_markdown(report: dict, *, title: Optional[str] = None) -> str:
    s = report.get("summary") or {}
    modes = report.get("modes") or [report.get("mode", "offline_tools")]
    title = title or f"Execution 公开快照（{report.get('report_id', 'exec')}）"
    mode_note = ", ".join(f"`{m}`" for m in modes)
    lines = [
        f"# {title}",
        "",
        f"- **report_id:** `{report.get('report_id', '')}`",
        f"- **timestamp:** `{report.get('timestamp', '')}`",
        f"- **dataset:** `{report.get('dataset', '')}`",
        f"- **modes:** {mode_note}",
        f"- **通过率:** **{s.get('passed', 0)}/{s.get('total', 0)}（{s.get('pass_rate', 0)}%）**",
        "",
        "## 按 mode",
        "",
        "| mode | passed | total | rate |",
        "|------|--------|-------|------|",
    ]
    for mode, info in (report.get("by_mode") or {}).items():
        lines.append(
            f"| `{mode}` | {info.get('passed', 0)} | {info.get('total', 0)} "
            f"| {info.get('pass_rate', 0)}% |"
        )
    lines.extend([
        "",
        "## 按 tag",
        "",
        "| tag | passed | total | rate |",
        "|-----|--------|-------|------|",
    ])
    for tag, info in (report.get("by_tag") or {}).items():
        lines.append(
            f"| `{tag}` | {info.get('passed', 0)} | {info.get('total', 0)} "
            f"| {info.get('pass_rate', 0)}% |"
        )
    lines.extend(["", "## 用例明细", ""])
    for r in report.get("results") or []:
        icon = "PASS" if r.get("passed") else ("SKIP" if r.get("skipped") else "FAIL")
        lines.append(
            f"- **{icon}** `{r.get('id')}` [{r.get('mode', '')}] — {r.get('name', '')} "
            f"({r.get('duration_seconds', 0)}s) — {r.get('reason', '')}"
        )
    honesty = [
        "",
        "## 诚实边界",
        "",
    ]
    if "agent" in modes and "offline_tools" not in modes:
        honesty.extend([
            "- 本套为 **端到端 Agent（LLM 规划 + 工具）** 执行验收，绑定模型与日期",
            "- 样本量有限；失败需区分模型失误 / 评分过严 / 工具环境",
            "",
        ])
    elif "agent" in modes:
        honesty.extend([
            "- `offline_tools` = 工具层验收；`agent` = LLM 端到端验收；勿混谈为同一指标",
            "- agent 数字绑定具体模型与 `report_id`",
            "",
        ])
    else:
        honesty.extend([
            "- 本套为 **工具执行验收**，不是端到端 Agent（LLM 规划）成功率",
            "- 数字绑定具体 `report_id`；改工具语义后需重跑",
            "",
        ])
    lines.extend(honesty)
    return "\n".join(lines)
