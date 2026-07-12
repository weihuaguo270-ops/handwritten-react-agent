"""trace_watch — Agent 执行路径监控与复盘

与 PermissionWrapper 配合使用，不是独立模块：
  ┌─────────────────────────────────────────────┐
  │  Agent 执行                                  │
  │    │                                         │
  │    ├── PermissionWrapper.check_tool_call()   │ ← 权限检查
  │    │       │                                 │
  │    │       ▼                                 │
  │    │   TraceWatch.record()                   │ ← 记录路径事件
  │    │       │                                 │
  │    │       ▼                                 │
  │    │   execute_tool_call()                   │ ← 执行工具
  │    │                                         │
  │    └── 执行完成                              │
  │          │                                   │
  │          ▼                                   │
  │    TraceWatch.summary()                      │ ← 生成复盘摘要
  │          │                                   │
  │          ▼                                   │
  │    随最终输出一起展示给用户                    │
  │    + HITL 询问是否重试失败路径               │
  └─────────────────────────────────────────────┘
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Callable

from .human_in_the_loop import HumanInTheLoop


# ── 事件类型 ──

class EventType:
    TOOL_BLOCKED = "tool_blocked"       # 工具被 HITL 拒绝
    TOOL_ERROR = "tool_error"           # 工具执行报错
    APPROACH_SWITCH = "approach_switch" # 切换执行路径
    SEARCH_FAIL = "search_fail"         # 搜索无结果
    LIMIT_HIT = "limit_hit"             # 达到限制（搜索/步数）
    COMPLETE = "complete"               # 执行完成


@dataclass
class TraceEvent:
    """单条监控事件"""
    type: str
    timestamp: float
    tool_name: str = ""
    description: str = ""
    suggestion: str = ""


@dataclass
class TraceReport:
    """执行复盘摘要（紧凑型，attach 到输出）"""
    session_id: str = ""
    events: list[TraceEvent] = field(default_factory=list)
    total_tool_calls: int = 0
    blocked_count: int = 0
    error_count: int = 0
    switch_count: int = 0

    @property
    def has_issues(self) -> bool:
        return self.blocked_count > 0 or self.error_count > 0

    @property
    def one_liner(self) -> str:
        """一行摘要"""
        parts = []
        if self.blocked_count:
            parts.append(f"{self.blocked_count} 次操作需用户确认")
        if self.error_count:
            parts.append(f"{self.error_count} 次工具报错")
        if self.switch_count:
            parts.append(f"切换了 {self.switch_count} 次执行路径")
        if not parts:
            return "执行顺利，无异常"
        return "；".join(parts)

    def to_text(self) -> str:
        """紧凑的复盘文本（2-5 行，attach 到输出末尾）"""
        if not self.has_issues and self.switch_count == 0:
            return ""

        lines = ["", "━━━ 执行复盘 ━━━"]
        for ev in self.events:
            icon = {"tool_blocked": "🔒", "tool_error": "❌",
                    "approach_switch": "🔄", "search_fail": "⚠",
                    "limit_hit": "⛔", "complete": "✅"}.get(ev.type, "•")
            lines.append(f"  {icon} {ev.description}")
            if ev.suggestion:
                lines.append(f"    建议: {ev.suggestion}")
        lines.append("━━━━━━━━━━━━━")
        return "\n".join(lines)

    def to_hitl_prompt(self) -> str:
        """生成给 HITL 的询问 prompt"""
        lines = [f"执行完成，期间遇到 {self.blocked_count + self.error_count} 个问题。"]
        for ev in self.events:
            if ev.type in ("tool_blocked", "tool_error"):
                lines.append(f"  · {ev.description}")
        lines.append("\n是否要修复这些问题后重新执行？")
        return "\n".join(lines)


# ── TraceWatch 监控器 ──

class TraceWatch:
    """执行路径监控器

    用法：
        # 1. 创建（需 HITL 实例）
        watch = TraceWatch(hitl=hitl)

        # 2. 在 PermissionWrapper 拦截回调中记录
        watch.record("tool_blocked", tool_name="write_file",
                     desc="用户拒绝了写文件操作")

        # 3. 执行完成后获取复盘摘要
        report = watch.summary()
        output += report.to_text()  # attach 到输出

        # 4. 询问用户是否重试
        if report.has_issues and watch.hitl:
            if watch.hitl.check_direction("重试失败路径",
                                          details=report.one_liner):
                # 用户同意 → 重新执行
    """

    def __init__(self, hitl: Optional[HumanInTheLoop] = None):
        self.hitl = hitl
        self.events: list[TraceEvent] = []
        self._tool_call_count = 0
        self._last_tool = ""

    def record(
        self,
        event_type: str,
        tool_name: str = "",
        description: str = "",
        suggestion: str = "",
    ) -> None:
        """记录一次事件"""
        self.events.append(TraceEvent(
            type=event_type,
            timestamp=time.time(),
            tool_name=tool_name,
            description=description[:200],
            suggestion=suggestion[:200],
        ))

    def on_tool_call(self, tool_name: str, args: dict, result: str) -> None:
        """工具调用完成后的回调

        自动检测：
        - 工具执行报错
        - 路径切换
        """
        self._tool_call_count += 1

        # 检测工具报错
        if isinstance(result, str) and ("error" in result.lower()
                                         or "blocked" in result.lower()):
            try:
                import json
                err = json.loads(result).get("error", "")
                if "blocked" in result:
                    self.record("tool_blocked", tool_name=tool_name,
                                description=f"{tool_name} 被用户拒绝",
                                suggestion="如需使用此工具，请联系管理员开通权限")
                else:
                    self.record("tool_error", tool_name=tool_name,
                                description=f"{tool_name} 执行报错: {err[:60]}",
                                suggestion="检查参数后重试")
            except json.JSONDecodeError:
                pass
            self._last_tool = tool_name
            return

        # 检测路径切换
        if self._last_tool and tool_name != self._last_tool:
            # 如果前一个工具是搜索类，切到其他工具是正常的
            # 但如果前一个工具报错后切到不同工具，记录路径切换
            pass  # 暂不自动触发 approach_switch

        self._last_tool = tool_name

    def on_approach_switch(self, from_tool: str, to_tool: str, reason: str = "") -> None:
        """记录路径切换

        当 Agent 从一条路换到另一条路时调用：
          from_tool: 之前的工具
          to_tool: 切换后的工具
          reason: 切换原因（如"搜索无结果"、"工具报错"）
        """
        self.record("approach_switch",
                    tool_name=f"{from_tool}→{to_tool}",
                    description=f"切换执行路径: {from_tool} → {to_tool}",
                    suggestion=reason or "检查前一步的失败原因")

    def on_limit_hit(self, limit_type: str, detail: str = "") -> None:
        """记录限制达到"""
        labels = {"search": "搜索次数达上限", "step": "达到最大步数",
                  "timeout": "执行超时"}
        label = labels.get(limit_type, limit_type)
        self.record("limit_hit", description=f"{label}: {detail}" if detail else label,
                    suggestion="尝试将复杂任务拆分为多个子任务")

    def summary(self) -> TraceReport:
        """生成复盘报告"""
        report = TraceReport()
        for ev in self.events:
            if ev.type == "tool_blocked":
                report.blocked_count += 1
            elif ev.type == "tool_error":
                report.error_count += 1
            elif ev.type == "approach_switch":
                report.switch_count += 1
        report.events = self.events
        report.total_tool_calls = self._tool_call_count
        return report

    def ask_user_for_retry(self) -> bool:
        """询问用户是否要重试失败路径

        返回: True=用户同意重试, False=用户拒绝
        """
        report = self.summary()
        if not report.has_issues:
            return False
        if not self.hitl:
            return False

        return self.hitl.check_direction(
            "重试失败路径",
            details=report.one_liner,
        )

    @property
    def has_issues(self) -> bool:
        return self.summary().has_issues
