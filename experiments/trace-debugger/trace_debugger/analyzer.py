"""analyzer — 路径分析与失败原因分类

对轨迹中的每条路径进行深度分析：
  - 工具调用是否成功/失败
  - 搜索是否返回有效结果
  - LLM 是否偏离用户意图
  - 是否存在重复尝试相同方案
  - 最终方案的可靠性评估
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional

from .reader import Trajectory, Path, Step


# ── 失败分类 ──

class FailureType:
    """失败原因分类"""
    TOOL_ERROR = "tool_error"           # 工具调用报错
    SEARCH_EMPTY = "search_empty"       # 搜索无结果
    SEARCH_TIMEOUT = "search_timeout"   # 搜索超时
    SEARCH_LIMIT = "search_limit"       # 搜索次数达上限
    LLM_OFFTRACK = "llm_offtrack"      # LLM 跑偏（答非所问）
    CONTEXT_OVERFLOW = "context_overflow"  # 上下文溢出
    DUPLICATE_ATTEMPT = "duplicate"     # 重复相同尝试
    NO_FINAL_ANSWER = "no_answer"       # 没给出最终答案
    STEP_LIMIT = "step_limit"           # 达到最大步数
    UNKNOWN = "unknown"                 # 无法分类

    LABELS = {
        "tool_error": "工具调用报错",
        "search_empty": "搜索无有效结果",
        "search_timeout": "搜索超时",
        "search_limit": "搜索次数达上限",
        "llm_offtrack": "LLM 偏离用户意图",
        "context_overflow": "上下文窗口溢出",
        "duplicate": "重复相同尝试",
        "no_answer": "未给出最终答案",
        "step_limit": "达到最大步数",
        "unknown": "未知原因",
    }


# ── 分析结果 ──

@dataclass
class StepAnalysis:
    """单步分析结果"""
    step_index: int
    action: str
    success: bool
    duration: float
    failure_type: str = ""
    failure_detail: str = ""
    suggestion: str = ""


@dataclass
class PathAnalysis:
    """单条路径的分析结果"""
    path_index: int
    num_steps: int
    tools_used: list[str]
    success: bool
    is_main: bool
    has_errors: bool
    failure_types: list[str]
    failure_details: list[str]
    step_analyses: list[StepAnalysis]
    summary: str


@dataclass
class TrajectoryAnalysis:
    """完整分析报告"""
    session_id: str
    query: str
    model: str
    total_duration: float
    total_steps: int
    num_paths: int
    paths: list[PathAnalysis]
    main_path_summary: str
    failed_paths_summary: str
    overall_assessment: str
    needs_fix: bool
    fix_suggestions: list[str]
    traj_issues: list[str] = field(default_factory=list)  # 轨迹级问题


# ── 分析器 ──

class Analyzer:
    """轨迹分析器

    用法：
        analyzer = Analyzer()
        analysis = analyzer.analyze(trajectory)
    """

    def analyze(self, traj: Trajectory) -> TrajectoryAnalysis:
        """分析完整轨迹"""
        path_analyses = []
        for i, path in enumerate(traj.paths):
            pa = self._analyze_path(path, i)
            path_analyses.append(pa)

        # 轨迹级分析
        traj_level_issues = self._analyze_trajectory_level(traj)

        # 主路径摘要
        main = traj.main_path
        main_summary = self._summarize_main(main, path_analyses) if main else "无主路径"

        # 失败路径摘要
        failed = traj.failed_paths
        failed_summary = self._summarize_failed(failed, path_analyses) if failed else "无失败路径"

        # 总体评估
        all_failures = []
        for pa in path_analyses:
            all_failures.extend(pa.failure_details)
        all_failures.extend(traj_level_issues)
        needs_fix = len(all_failures) > 0

        return TrajectoryAnalysis(
            session_id=traj.session_id,
            query=traj.query,
            model=traj.model,
            total_duration=traj.total_duration,
            total_steps=traj.num_steps,
            num_paths=traj.num_paths,
            paths=path_analyses,
            main_path_summary=main_summary,
            failed_paths_summary=failed_summary,
            overall_assessment=self._assess_overall(traj, path_analyses, traj_level_issues),
            needs_fix=needs_fix,
            fix_suggestions=self._generate_suggestions(traj, path_analyses, traj_level_issues),
            traj_issues=traj_level_issues,
        )

    def _analyze_trajectory_level(self, traj: Trajectory) -> list[str]:
        """轨迹级问题检测"""
        issues = []

        # 1. 检测搜索次数达上限
        search_count = 0
        web_search_steps = []
        for s in traj.steps:
            if s.action_name == "web_search":
                search_count += 1
                web_search_steps.append(s.index)
        if search_count >= 4:
            issues.append(f"搜索次数达到上限（{search_count}次），第4次及之后的搜索被跳过")

        # 2. 检测达到最大步数
        if traj.num_steps >= 10 and not traj.final_answer:
            issues.append(f"达到最大步数（{traj.num_steps}步）仍未给出最终答案，执行被强制终止")
        elif traj.num_steps >= 10:
            issues.append(f"接近最大步数限制（{traj.num_steps}步），最终答案可能不够充分")

        # 3. 检测重复搜索（相同关键词搜了多次）
        search_queries = []
        for s in traj.steps:
            if s.action_name == "web_search" and s.action_args:
                # 提取搜索关键词
                query = s.action_args[:60]
                search_queries.append(query)
        if len(search_queries) != len(set(search_queries)):
            issues.append("存在重复搜索相同关键词的情况")

        return issues

    def _analyze_path(self, path: Path, index: int) -> PathAnalysis:
        """分析单条路径"""
        step_analyses = []
        failure_types = set()
        failure_details = []

        for step in path.steps:
            sa = self._analyze_step(step)
            step_analyses.append(sa)
            if not sa.success and sa.failure_type:
                failure_types.add(sa.failure_type)
                if sa.failure_detail:
                    failure_details.append(f"Step {step.index}: {sa.failure_detail}")

        summary_parts = []
        if path.success:
            summary_parts.append("成功")
        else:
            summary_parts.append("失败")
        summary_parts.append(f"{len(path.steps)} 步")
        if path.tools_used:
            summary_parts.append(f"工具: {', '.join(path.tools_used)}")
        if failure_types:
            labels = [FailureType.LABELS.get(ft, ft) for ft in failure_types]
            summary_parts.append(f"问题: {'/'.join(labels)}")

        return PathAnalysis(
            path_index=index,
            num_steps=path.num_steps,
            tools_used=path.tools_used,
            success=path.success,
            is_main=path.is_main_path,
            has_errors=path.has_errors,
            failure_types=list(failure_types),
            failure_details=failure_details,
            step_analyses=step_analyses,
            summary=" | ".join(summary_parts),
        )

    def _analyze_step(self, step: Step) -> StepAnalysis:
        """分析单步"""
        failure_type = ""
        failure_detail = ""
        suggestion = ""

        if step.is_action:
            if step.has_error:
                failure_type = FailureType.TOOL_ERROR
                # 错误信息精简
                err = step.error_message[:80] if step.error_message else ""
                failure_detail = f"{step.action_name} 调用失败: {err}"
                if "参数解析" in err:
                    suggestion = f"传参格式有误，检查引号嵌套或 JSON 格式"
                elif "timeout" in err.lower() or "超时" in err:
                    suggestion = "任务耗时过长，考虑分步执行或增加超时"
                elif "stderr" in err.lower() or "traceback" in err.lower():
                    suggestion = f"代码执行报错，检查语法或依赖"
                else:
                    suggestion = f"检查 {step.action_name} 的参数或重试"

            elif "搜索" in step.action_name:
                if not step.observation or len(step.observation.strip()) < 20:
                    failure_type = FailureType.SEARCH_EMPTY
                    # 从参数中提取搜索词
                    query = step.action_args[:60] if step.action_args else ""
                    failure_detail = f"搜索 '{query}' 无有效结果"
                    suggestion = "换搜索关键词或尝试其他来源"
                # 搜索成功但无后续检查
            elif step.action_name in ("web_search",):
                # 检查是否靠近搜索限制（后续通过轨迹级分析补充）
                pass

        if not failure_type:
            # 达到最大步数检测
            if not step.observation and not step.action_name:
                pass  # 在轨迹级分析中处理

        return StepAnalysis(
            step_index=step.index,
            action=step.action_name,
            success=not bool(failure_type),
            duration=step.duration,
            failure_type=failure_type,
            failure_detail=failure_detail,
            suggestion=suggestion,
        )

    def _summarize_main(self, main: Path, analyses: list[PathAnalysis]) -> str:
        """生成主路径摘要"""
        for pa in analyses:
            if pa.is_main:
                if pa.success:
                    return f"最终通过 {pa.num_steps} 步完成"
                else:
                    return f"已执行 {pa.num_steps} 步但可能不够理想"
        return ""

    def _summarize_failed(self, failed: list[Path], analyses: list[PathAnalysis]) -> str:
        """生成失败路径摘要"""
        parts = []
        for pa in analyses:
            if pa.is_main:
                continue
            parts.append(f"路径 {pa.path_index}: {pa.summary}")
        return "\n".join(parts) if parts else "无"

    def _assess_overall(self, traj: Trajectory, analyses: list[PathAnalysis],
                        traj_level_issues: list[str] = None) -> str:
        """总体质量评估"""
        if traj_level_issues is None:
            traj_level_issues = []
        total_failures = sum(
            1 for pa in analyses for sa in pa.step_analyses if not sa.success
        )
        total_issues = total_failures + len(traj_level_issues)
        total_steps = sum(pa.num_steps for pa in analyses)

        if total_issues == 0:
            return f"✅ 执行顺利，{total_steps} 步无错误"
        elif total_failures == 0 and traj_level_issues:
            return f"⚠️ 步骤未报错，但存在轨迹级问题（{len(traj_level_issues)} 项）"
        elif total_issues <= max(total_steps * 0.3, 2):
            return f"⚠️ 有少量问题（{total_issues} 项/{total_steps} 步），可考虑优化"
        else:
            return f"❌ 执行问题较多（{total_issues} 项/{total_steps} 步），建议检查"

    def _generate_suggestions(self, traj: Trajectory, analyses: list[PathAnalysis],
                              traj_level_issues: list[str] = None) -> list[str]:
        """生成修复建议"""
        if traj_level_issues is None:
            traj_level_issues = []
        suggestions = []
        seen_types = set()

        # 步骤级建议
        for pa in analyses:
            for ft in pa.failure_types:
                if ft not in seen_types:
                    seen_types.add(ft)
                    label = FailureType.LABELS.get(ft, ft)
                    suggestions.append(f"修复 {label}：{self._suggestion_for(ft)}")

        # 轨迹级建议
        if any("搜索次数达到上限" in i for i in traj_level_issues):
            suggestions.append("搜索次数过多：先整合已有搜索结果再决定是否继续搜索")
        if any("达到最大步数" in i for i in traj_level_issues):
            suggestions.append("任务超步数限制：尝试将复杂任务拆分为多个子任务")
        if any("重复搜索" in i for i in traj_level_issues):
            suggestions.append("存在重复搜索：建议搜索前先检查是否已有相关结果")

        return suggestions

    def _suggestion_for(self, failure_type: str) -> str:
        mapping = {
            FailureType.TOOL_ERROR: "检查工具参数是否正确，或增加参数校验",
            FailureType.SEARCH_EMPTY: "调整搜索词策略，先确认需求再搜索",
            FailureType.SEARCH_TIMEOUT: "限制搜索范围或添加缓存层",
            FailureType.LLM_OFFTRACK: "在 system prompt 中强化约束",
            FailureType.DUPLICATE_ATTEMPT: "添加状态追踪，避免重复相同尝试",
        }
        return mapping.get(failure_type, "检查执行环境和输入")
