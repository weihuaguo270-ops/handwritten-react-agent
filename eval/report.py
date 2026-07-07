"""report — 评测报告生成与保存

对 runner 输出的原始结果列表 + scorer 的评分，汇总为完整报告：
  1. 每条用例的得分明细
  2. 总体统计（总数/通过率/平均分/平均步数/平均耗时）
  3. 按 tag 分组统计
  4. 失败案例列表（含回放路径）
  5. 保存到 eval/reports/ 目录
"""

import json
import os
import time
from typing import Optional

from .scorer import score_result

REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


def generate_report(raw_results: list, cases: list,
                    provider: Optional[str] = None) -> dict:
    """为批量执行结果生成完整评测报告

    参数:
        raw_results: runner.run_batch 返回的结果列表
        cases: 对应的 TestCase 列表
        provider: LLM provider（记录在报告中）

    返回:
        报告字典
    """
    scored = []
    passed_count = 0

    for raw, case in zip(raw_results, cases):
        stdout = raw.get("stdout", "")
        trajectory = raw.get("trajectory")
        score = score_result(case, stdout, trajectory)

        entry = {
            "case_id": case.id or f"case_{len(scored)+1}",
            "question": case.question[:100],
            "tag": case.tag,
            "timed_out": raw.get("timed_out", False),
            "exit_code": raw.get("exit_code", 0),
            "duration_seconds": raw.get("duration_seconds", 0),
            "trajectory_file": trajectory.get("session_id", "")
                                 if trajectory else "",
            "total_steps": trajectory.get("total_steps", len(trajectory.get("steps", [])))
                           if trajectory else 0,
            "total_tokens": trajectory.get("total_tokens_estimated", 0)
                            if trajectory else 0,
            "score": score,
        }
        if score["passed"]:
            passed_count += 1
        scored.append(entry)

    total = len(scored)
    total_score = sum(s["score"]["total"] for s in scored)
    max_score = sum(s["score"]["max_score"] for s in scored)

    # 统计
    by_tag = {}
    for s in scored:
        tag = s.get("tag") or "unknown"
        if tag not in by_tag:
            by_tag[tag] = {"total": 0, "passed": 0, "total_duration": 0.0}
        by_tag[tag]["total"] += 1
        if s["score"]["passed"]:
            by_tag[tag]["passed"] += 1
        by_tag[tag]["total_duration"] += s["duration_seconds"]
    for tag, stats in by_tag.items():
        stats["pass_rate"] = round(stats["passed"] / stats["total"], 3) if stats["total"] else 0

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
            "avg_duration": round(sum(s["duration_seconds"] for s in scored) / total, 2) if total else 0,
            "avg_steps": round(sum(s["total_steps"] for s in scored) / total, 1) if total else 0,
            "avg_tokens": round(sum(s["total_tokens"] for s in scored) / total, 0) if total else 0,
        },
        "by_tag": by_tag,
        "results": scored,
        "failures": [s for s in scored if not s["score"]["passed"]],
    }

    return report


def save_report(report: dict, directory: Optional[str] = None) -> str:
    """将评测报告保存为 JSON 文件，返回文件路径

    文件名: eval_YYYYMMDD_HHMMSS.json
    trajectory_file 字段记录了 session_id，dashboard 可以通过
    /api/trajectories/<traj_xxx.json> 直接回放。
    """
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
    """列出所有评测报告

    返回按时间倒序排列的摘要列表。
    """
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
                "filepath": f,
            })
        except (json.JSONDecodeError, OSError):
            continue
    return result
