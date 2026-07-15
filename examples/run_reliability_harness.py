"""Harness 长跑可靠性对照：ToolGuard / 自修 ON vs OFF 可测指标。

不依赖 LLM：注入故障工具，度量重试恢复、超时阻断、自修提示附着率。

用法：
  python examples/run_reliability_harness.py
  python examples/run_reliability_harness.py --publish
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("REACT_AGENT_SKIP_RAG", "1")
os.environ.setdefault("REACT_AGENT_SANDBOX_PREWARM", "0")


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _scenario_flaky_timeout_recover() -> dict:
    """SAFE 工具暂时失败（timeout），Guard ON 应重试后成功。"""
    from react_agent.resilience import ToolGuard

    results = {}
    for guard_on in (True, False):
        calls = {"n": 0}

        def flaky(tc):
            calls["n"] += 1
            if calls["n"] < 3:
                raise Exception("timeout")
            return "recovered_ok"

        if guard_on:
            guard = ToolGuard()
            fn = guard.wrap(flaky)
        else:
            fn = flaky
        calls["n"] = 0
        try:
            out = fn({"function": {"name": "get_time", "arguments": "{}"}})
            recovered = out == "recovered_ok"
            err = None
        except Exception as e:
            recovered = False
            out = ""
            err = str(e)[:120]
        results["on" if guard_on else "off"] = {
            "recovered": recovered,
            "calls": calls["n"],
            "output": (out or "")[:80],
            "error": err,
        }
    return {
        "id": "flaky_timeout_recover",
        "description": "timeout 可恢复：前 2 次失败，第 3 次成功",
        "metric": "recovered",
        "guard_on": results["on"],
        "guard_off": results["off"],
        "delta": {
            "on_better": results["on"]["recovered"] and not results["off"]["recovered"],
            "on_calls": results["on"]["calls"],
            "off_calls": results["off"]["calls"],
        },
    }


def _scenario_hard_timeout_block() -> dict:
    """硬超时：慢工具被 Guard 截断；OFF 则一直睡。"""
    from react_agent.resilience import ToolGuard

    results = {}
    for guard_on in (True, False):
        def slow(tc):
            time.sleep(3)
            return "too_late"

        t0 = time.time()
        if guard_on:
            guard = ToolGuard()
            guard._TOOL_TIMEOUTS = {**ToolGuard._TOOL_TIMEOUTS, "execute_python": 1}
            fn = guard.wrap(slow)
            out = fn({"function": {"name": "execute_python", "arguments": "{}"}})
            elapsed = round(time.time() - t0, 2)
            blocked = "超时" in out or "error" in out.lower()
            results["on"] = {
                "blocked_or_timed_out": blocked,
                "elapsed_s": elapsed,
                "output": out[:120],
            }
        else:
            # OFF：不等满 3s，用短睡表示「无保护会跑满」——用 0.2s 代理，避免拖垮套件
            def slow_short(tc):
                time.sleep(0.25)
                return "too_late"

            out = slow_short({"function": {"name": "execute_python", "arguments": "{}"}})
            elapsed = round(time.time() - t0, 2)
            results["off"] = {
                "blocked_or_timed_out": False,
                "elapsed_s": elapsed,
                "output": out[:120],
                "note": "OFF 路径用短睡代理，表示无超时截断会返回结果",
            }
    return {
        "id": "hard_timeout_block",
        "description": "ToolGuard ON 时对慢工具做超时截断",
        "metric": "blocked_or_timed_out",
        "guard_on": results["on"],
        "guard_off": results["off"],
        "delta": {
            "on_blocks": results["on"]["blocked_or_timed_out"],
            "off_returns_result": results["off"]["output"] == "too_late",
        },
    }


def _scenario_self_repair_attach() -> dict:
    """错误观测上附着自修提示。"""
    from react_agent.react_loop import looks_like_tool_error, self_repair_hint

    err = '{"error": "超时 (30s)", "retry_exhausted": true}'
    ok = looks_like_tool_error(err)
    hint = self_repair_hint("web_search", err) if ok else ""
    attached = "[Harness自修]" in hint and "web_search" in hint
    clean = "323"
    no_false = not looks_like_tool_error(clean)
    return {
        "id": "self_repair_attach",
        "description": "失败结果附着自修提示；正常结果不误触发",
        "metric": "attach_precision",
        "guard_on": {
            "error_detected": ok,
            "hint_attached": attached,
            "clean_not_flagged": no_false,
            "passed": ok and attached and no_false,
        },
        "guard_off": {
            "note": "关闭自修时仅透传原始错误，无提示层（逻辑上 attach=False）",
            "hint_attached": False,
            "passed": True,
        },
        "delta": {"self_repair_available": attached},
    }


def _scenario_rate_limit_guard() -> dict:
    """频率限制：短时间超限被阻断。"""
    from react_agent.resilience import ToolGuard

    guard = ToolGuard()
    guard._max_rate = 2

    def ok_tool(tc):
        return "ok"

    wrapped = guard.wrap(ok_tool)
    tc = {"function": {"name": "web_search", "arguments": "{}"}}
    r1 = wrapped(tc)
    r2 = wrapped(tc)
    r3 = wrapped(tc)
    blocked = '"blocked"' in r3 and "频繁" in r3
    return {
        "id": "rate_limit_guard",
        "description": "1 分钟内同工具超限调用被阻断",
        "metric": "third_call_blocked",
        "guard_on": {
            "first": r1,
            "second": r2,
            "third": r3[:80],
            "blocked": blocked,
            "passed": r1 == "ok" and r2 == "ok" and blocked,
        },
        "guard_off": {
            "note": "无 Guard 时三次均可成功（此处不实际调用）",
            "would_block": False,
        },
        "delta": {"guard_blocks_burst": blocked},
    }


def run_harness() -> dict:
    scenarios = [
        _scenario_flaky_timeout_recover(),
        _scenario_hard_timeout_block(),
        _scenario_self_repair_attach(),
        _scenario_rate_limit_guard(),
    ]
    # pass = scenario demonstrates intended Guard/self-repair property
    passed = 0
    for s in scenarios:
        sid = s["id"]
        if sid == "flaky_timeout_recover":
            s["passed"] = bool(s["delta"]["on_better"])
        elif sid == "hard_timeout_block":
            s["passed"] = bool(s["delta"]["on_blocks"])
        elif sid == "self_repair_attach":
            s["passed"] = bool(s["guard_on"]["passed"])
        elif sid == "rate_limit_guard":
            s["passed"] = bool(s["guard_on"]["passed"])
        else:
            s["passed"] = False
        if s["passed"]:
            passed += 1

    total = len(scenarios)
    return {
        "report_id": f"reliability_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kind": "harness_reliability_injected",
        "summary": {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round(100.0 * passed / total, 1) if total else 0.0,
        },
        "scenarios": scenarios,
        "meta": {
            "git": _git_sha(),
            "note": (
                "注入故障对照，非线上长跑 SLA；"
                "证明 ToolGuard/自修机制可测，不等于全链路 Agent 稳定性"
            ),
        },
    }


def to_markdown(report: dict, *, title: str) -> str:
    s = report["summary"]
    lines = [
        f"# {title}",
        "",
        f"- **report_id:** `{report.get('report_id', '')}`",
        f"- **timestamp:** `{report.get('timestamp', '')}`",
        f"- **kind:** `{report.get('kind', '')}`",
        f"- **场景通过:** **{s['passed']}/{s['total']}（{s['pass_rate']}%）**",
        f"- **git:** `{(report.get('meta') or {}).get('git', '')}`",
        "",
        "## 对照表",
        "",
        "| scenario | Guard/自修效果 | passed |",
        "|----------|----------------|:------:|",
    ]
    for sc in report.get("scenarios") or []:
        icon = "yes" if sc.get("passed") else "no"
        lines.append(
            f"| `{sc.get('id')}` | {sc.get('description', '')} | {icon} |"
        )
    lines.extend([
        "",
        "## 明细",
        "",
    ])
    for sc in report.get("scenarios") or []:
        lines.append(f"### `{sc.get('id')}`")
        lines.append("")
        lines.append(f"- ON: `{json.dumps(sc.get('guard_on'), ensure_ascii=False)[:300]}`")
        lines.append(f"- OFF: `{json.dumps(sc.get('guard_off'), ensure_ascii=False)[:300]}`")
        lines.append(f"- delta: `{json.dumps(sc.get('delta'), ensure_ascii=False)[:300]}`")
        lines.append("")
    lines.extend([
        "## 复现",
        "",
        "```bash",
        "python examples/run_reliability_harness.py --publish",
        "```",
        "",
        "## 诚实边界",
        "",
        "- 本报告是 **注入故障单元对照**，不是数百步线上长跑",
        "- 与 live Agent execution 成功率是不同指标，勿合并宣传",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--stem", default=None)
    args = parser.parse_args()

    report = run_harness()
    s = report["summary"]
    print("=" * 55)
    print("  Harness Reliability (injected)")
    print(f"  {s['passed']}/{s['total']}  pass_rate={s['pass_rate']}%")
    print("=" * 55)
    for sc in report["scenarios"]:
        print(f"  [{'OK' if sc['passed'] else 'FAIL'}] {sc['id']}")

    if args.publish:
        stem = args.stem or f"reliability_snapshot_{datetime.now().strftime('%Y%m%d')}"
        docs = ROOT / "docs"
        snap = docs / "snapshots"
        docs.mkdir(exist_ok=True)
        snap.mkdir(exist_ok=True)
        (snap / f"{stem}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (docs / f"{stem}.md").write_text(
            to_markdown(report, title=f"Harness 可靠性对照（{stem}）"),
            encoding="utf-8",
        )
        print(f"\nPublished docs/{stem}.md")

    return 0 if s["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
