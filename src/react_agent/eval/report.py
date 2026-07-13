"""report — 评测报告生成与保存

对 runner 输出的原始结果列表 + scorer 的评分，汇总为完整报告：
  1. 每条用例的得分明细
  2. 总体统计（总数/通过率/平均分/平均步数/平均耗时）
  3. 按 tag / capability 分组统计
  4. 失败案例列表（含回放路径）
  5. 保存到 eval/reports/ 目录
"""

import json
import os
import time
from typing import Optional

from .scorer import score_result
from .capability_scorer import score_capability

REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


def generate_report(raw_results: list, cases: list,
                    provider: Optional[str] = None) -> dict:
    """为批量执行结果生成完整评测报告"""
    scored = []
    passed_count = 0

    for raw, case in zip(raw_results, cases):
        stdout = raw.get("stdout", "")
        trajectory = raw.get("trajectory")
        run_results = raw.get("run_results")  # consistency 多次运行

        if getattr(case, "capability", None):
            score = score_capability(
                case, stdout, trajectory, run_results=run_results
            )
        else:
            score = score_result(case, stdout, trajectory)

        entry = {
            "case_id": case.id or f"case_{len(scored)+1}",
            "question": case.question[:100],
            "tag": case.tag,
            "capability": getattr(case, "capability", None),
            "timed_out": raw.get("timed_out", False),
            "exit_code": raw.get("exit_code", 0),
            "duration_seconds": raw.get("duration_seconds", 0),
            "trajectory_file": trajectory.get("session_id", "")
                                 if trajectory else "",
            "total_steps": (
                trajectory.get("total_steps", len(trajectory.get("steps", [])))
                if trajectory else 0
            ),
            "total_tokens": (
                trajectory.get("total_tokens_estimated", 0) if trajectory else 0
            ),
            "score": score,
            "metrics": score.get("metrics", {}),
        }
        if score.get("passed"):
            passed_count += 1
        scored.append(entry)

    total = len(scored)
    total_score = sum(s["score"].get("total", 0) for s in scored)
    max_score = sum(s["score"].get("max_score", 0) for s in scored) or 1

    by_tag = _group_stats(scored, key="tag")
    by_capability = _group_capability_stats(scored)
    capability_summary = _capability_summary(scored)

    report = {
        "report_id": time.strftime("eval_%Y%m%d_%H%M%S"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "provider": provider or "default",
        "summary": {
            "total": total,
            "passed": passed_count,
            "failures": total - passed_count,
            "pass_rate": round(passed_count / total, 3) if total else 0,
            "total_score": total_score,
            "max_score": max_score,
            "score_rate": round(total_score / max_score, 3) if max_score else 0,
            "avg_duration": round(
                sum(s["duration_seconds"] for s in scored) / total, 2
            ) if total else 0,
            "avg_steps": round(
                sum(s["total_steps"] for s in scored) / total, 1
            ) if total else 0,
            "avg_tokens": round(
                sum(s["total_tokens"] for s in scored) / total, 0
            ) if total else 0,
            **capability_summary,
        },
        "by_tag": by_tag,
        "by_capability": by_capability,
        "results": scored,
        "failures": [s for s in scored if not s["score"].get("passed")],
    }

    return report


def _group_stats(scored: list, key: str = "tag") -> dict:
    by = {}
    for s in scored:
        tag = s.get(key) or "unknown"
        if tag not in by:
            by[tag] = {"total": 0, "passed": 0, "total_duration": 0.0}
        by[tag]["total"] += 1
        if s["score"].get("passed"):
            by[tag]["passed"] += 1
        by[tag]["total_duration"] += s["duration_seconds"]
    for tag, stats in by.items():
        stats["pass_rate"] = (
            round(stats["passed"] / stats["total"], 3) if stats["total"] else 0
        )
    return by


def _group_capability_stats(scored: list) -> dict:
    by = {}
    for s in scored:
        cap = s.get("capability")
        if not cap:
            continue
        if cap not in by:
            by[cap] = {
                "total": 0,
                "passed": 0,
                "total_duration": 0.0,
                "metric_sums": {},
                "metric_counts": {},
            }
        by[cap]["total"] += 1
        if s["score"].get("passed"):
            by[cap]["passed"] += 1
        by[cap]["total_duration"] += s["duration_seconds"]
        for mk, mv in (s.get("metrics") or {}).items():
            if isinstance(mv, (int, float)):
                by[cap]["metric_sums"][mk] = by[cap]["metric_sums"].get(mk, 0) + float(mv)
                by[cap]["metric_counts"][mk] = by[cap]["metric_counts"].get(mk, 0) + 1

    for cap, stats in by.items():
        stats["pass_rate"] = (
            round(stats["passed"] / stats["total"], 3) if stats["total"] else 0
        )
        avgs = {}
        for mk, total in stats["metric_sums"].items():
            cnt = stats["metric_counts"].get(mk, 1)
            avgs[mk] = round(total / cnt, 3)
        stats["avg_metrics"] = avgs
        del stats["metric_sums"]
        del stats["metric_counts"]
    return by


def _capability_summary(scored: list) -> dict:
    """顶层五项能力摘要（仅对有对应 capability 的子集计算）。"""
    out = {}

    def _subset(cap):
        return [s for s in scored if s.get("capability") == cap]

    acc = _subset("accuracy")
    if acc:
        out["accuracy_rate"] = round(
            sum(1 for s in acc if s["score"].get("passed")) / len(acc), 3
        )

    tools = _subset("tool_selection")
    if tools:
        f1s = [s.get("metrics", {}).get("tool_f1") for s in tools]
        f1s = [x for x in f1s if isinstance(x, (int, float))]
        out["tool_selection_f1"] = round(sum(f1s) / len(f1s), 3) if f1s else 0.0

    reason = _subset("reasoning")
    if reason:
        out["reasoning_rate"] = round(
            sum(1 for s in reason if s["score"].get("passed")) / len(reason), 3
        )

    cons = _subset("consistency")
    if cons:
        rates = [s.get("metrics", {}).get("consistency_rate") for s in cons]
        rates = [x for x in rates if isinstance(x, (int, float))]
        out["consistency_rate"] = round(sum(rates) / len(rates), 3) if rates else 0.0

    hall = _subset("hallucination")
    if hall:
        hs = [s.get("metrics", {}).get("hallucination_rate") for s in hall]
        hs = [x for x in hs if isinstance(x, (int, float))]
        out["hallucination_rate"] = round(sum(hs) / len(hs), 3) if hs else 0.0

    return out


def save_report(report: dict, directory: Optional[str] = None) -> str:
    """将评测报告保存为 JSON 文件，返回文件路径"""
    save_dir = directory or REPORT_DIR
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{report['report_id']}.json"
    filepath = os.path.join(save_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return filepath


def load_report(filepath: str) -> dict:
    """加载已保存的评测报告"""
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def list_reports(directory: Optional[str] = None) -> list[dict]:
    """列出所有评测报告（按时间倒序）。"""
    import glob
    save_dir = directory or REPORT_DIR
    os.makedirs(save_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(save_dir, "eval_*.json")), reverse=True)
    result = []
    for f in files[:50]:
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            result.append({
                "report_id": data.get("report_id", ""),
                "timestamp": data.get("timestamp", ""),
                "provider": data.get("provider", ""),
                "summary": data.get("summary", {}),
                "by_tag": data.get("by_tag", {}),
                "by_capability": data.get("by_capability", {}),
                "filepath": f,
            })
        except (json.JSONDecodeError, OSError):
            continue
    return result
