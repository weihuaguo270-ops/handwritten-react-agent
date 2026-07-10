"""reporter — 复盘报告生成

将 Analyzer 的分析结果转为：
  - 终端可读的复盘报告
  - 结构化的 JSON 数据
  - 供 LLM Judge 进一步分析的 prompt
"""
from __future__ import annotations
import json
import time
from typing import Optional

from .reader import Trajectory
from .analyzer import TrajectoryAnalysis, FailureType


def format_report(analysis: TrajectoryAnalysis) -> str:
    """格式化为终端可读的复盘报告"""
    lines = []
    lines.append("=" * 55)
    lines.append("  🔍 Trace Debugger — 执行复盘报告")
    lines.append("=" * 55)
    lines.append(f"  会话:    {analysis.session_id}")
    lines.append(f"  模型:    {analysis.model}")
    lines.append(f"  耗时:    {analysis.total_duration:.1f}s")
    lines.append(f"  步骤:    {analysis.total_steps} 步 / {analysis.num_paths} 条路径")
    lines.append("")

    # 用户需求
    lines.append("  ── 用户需求 ──")
    lines.append(f"  {analysis.query[:120]}")
    lines.append("")

    # 总体评估
    lines.append(f"  ── 总体评估 ──")
    lines.append(f"  {analysis.overall_assessment}")
    lines.append("")

    # 路径详情
    for pa in analysis.paths:
        icon = "✅" if pa.success else "❌"
        label = "（主路径）" if pa.is_main else "（备选路径）"
        lines.append(f"  ── 路径 {pa.path_index} {label} {icon} ──")
        lines.append(f"  {pa.summary}")

        for sa in pa.step_analyses:
            step_icon = "✅" if sa.success else "❌"
            action_info = f" [{sa.action}]" if sa.action else ""
            dur = f"  {sa.duration:.1f}s" if sa.duration else ""
            lines.append(f"    {step_icon} Step {sa.step_index}{action_info}{dur}")
            if sa.failure_detail:
                lines.append(f"      原因: {sa.failure_detail}")
            if sa.suggestion:
                lines.append(f"      建议: {sa.suggestion}")
        lines.append("")

    # 轨迹级问题
    if analysis.traj_issues:
        lines.append("  ── 轨迹级问题 ──")
        for issue in analysis.traj_issues:
            lines.append(f"  ⚠ {issue}")
        lines.append("")

    # 失败路径汇总
    if any(not pa.success and not pa.is_main for pa in analysis.paths):
        lines.append("  ── 失败路径汇总 ──")
        lines.append(f"  {analysis.failed_paths_summary}")
        lines.append("")

    # 修复建议
    if analysis.needs_fix and analysis.fix_suggestions:
        lines.append("  ── 修复建议 ──")
        for i, s in enumerate(analysis.fix_suggestions, 1):
            lines.append(f"  {i}. {s}")
        lines.append("")
        lines.append("  🔄 是否需要修复这些问题后重新输出？")
        lines.append(f"  输入 y 确认修复，n 忽略，v 查看详细步骤")
        lines.append("")

    lines.append("=" * 55)
    return "\n".join(lines)


def format_json(analysis: TrajectoryAnalysis) -> str:
    """格式化为 JSON"""
    return json.dumps({
        "session_id": analysis.session_id,
        "query": analysis.query,
        "model": analysis.model,
        "total_duration": analysis.total_duration,
        "total_steps": analysis.total_steps,
        "overall_assessment": analysis.overall_assessment,
        "needs_fix": analysis.needs_fix,
        "fix_suggestions": analysis.fix_suggestions,
        "paths": [
            {
                "index": pa.path_index,
                "success": pa.success,
                "is_main": pa.is_main,
                "num_steps": pa.num_steps,
                "tools_used": pa.tools_used,
                "failures": pa.failure_types,
                "details": pa.failure_details,
            }
            for pa in analysis.paths
        ],
    }, ensure_ascii=False, indent=2)


def build_judge_prompt(analysis: TrajectoryAnalysis) -> str:
    """生成供 LLM Judge 进一步分析的 prompt

    让 Judge 判断：
    1. 失败路是否真的无法修复？还是可以换参数重试？
    2. 主输出是否有受失败路径影响的风险？
    3. 整体上 Agent 的决策是否合理？
    """
    return f"""你是一个 Agent 执行质量的深度分析专家。请分析以下执行复盘报告：

## 用户需求
{analysis.query[:200]}

## 总体评估
{analysis.overall_assessment}

## 路径详情
{chr(10).join(f"{'✅' if pa.success else '❌'} 路径{pa.path_index}{'（主）' if pa.is_main else ''}: {pa.summary}" for pa in analysis.paths)}

## 你的任务
1. 主路径的最终答案是否可靠？有没有潜在风险？
2. 失败路径是真的无法修复，还是可以简单调整后重试？
3. 如果修复，先修复哪条路径？
4. Agent 的执行策略是否合理？

请输出 JSON：
{{"main_path_risk": "low/medium/high", "fixable_paths": [...], "priority": "...", "strategy_assessment": "..."}}
"""
