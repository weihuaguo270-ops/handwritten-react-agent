"""human_in_the_loop — 人工审批管理器

在 Agent 执行高风险操作或 Eval Loop 要调整方向前，暂停并询问用户。

优化 v2：
  1. 临时授权过期（"仅此一次"5分钟后自动失效）
  2. 上下文展示（附 Agent 当前思考过程）
  3. 同类型操作静默延续（避免频繁打断）
"""
from __future__ import annotations
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .permissions import (
    PermissionLevel,
    get_tool_permission,
    get_direction_permission,
    is_high_risk,
    describe_action,
)


# ── 审批记录 ──

@dataclass
class ApprovalRecord:
    timestamp: str
    operation: str
    category: str
    level: str
    approved: bool
    reason: str = ""
    timeout: bool = False
    context: str = ""  # Agent 上下文


# ── HITL 管理器 ──

class HumanInTheLoop:
    """人工审批管理器（v2 — 临时授权过期 + 上下文展示）"""

    def __init__(
        self,
        ask_fn: Optional[Callable[[str, list[str]], str]] = None,
        auto_approve_safe: bool = True,
        default_timeout: float = 60.0,
        audit_log_size: int = 100,
        temp_auth_minutes: float = 5.0,
    ):
        self.ask_fn = ask_fn
        self.auto_approve_safe = auto_approve_safe
        self.default_timeout = default_timeout
        self.audit_log: list[ApprovalRecord] = []
        self._max_log = audit_log_size
        self._temp_auth_ttl = temp_auth_minutes * 60  # 秒
        # 临时授权缓存: {tool_key: expiry_timestamp}
        self._temp_approvals: dict[str, float] = {}
        # 永久授权: {tool_key: True}
        self._perm_approvals: dict[str, bool] = {}

    # ── 公开 API ──

    def check_tool_call(
        self,
        tool_name: str,
        tool_args: Optional[dict] = None,
        reason: str = "",
    ) -> bool:
        level = get_tool_permission(tool_name, tool_args)  # 参数级权限
        desc = describe_action(tool_name, tool_args)
        return self._check(
            operation=desc,
            category="tool_call",
            level=level,
            reason=reason,
            context_hint=f"Agent 思考: {reason[:200]}" if reason else "",
        )

    def check_direction(
        self,
        action: str,
        details: str = "",
    ) -> bool:
        level = get_direction_permission(action)
        desc = f"{action}：{details}" if details else action
        return self._check(
            operation=desc,
            category="direction_change",
            level=level,
        )

    # ── 内部逻辑 ──

    def _tool_key(self, tool_name: str) -> str:
        """生成工具的唯一 key（用于授权缓存）"""
        return f"tool:{tool_name}"

    def _has_temp_auth(self, key: str) -> bool:
        """检查临时授权是否有效"""
        if key not in self._temp_approvals:
            return False
        if time.time() > self._temp_approvals[key]:
            del self._temp_approvals[key]
            return False
        return True

    def _check(
        self,
        operation: str,
        category: str,
        level: PermissionLevel,
        reason: str = "",
        context_hint: str = "",
    ) -> bool:
        approved = True
        timeout = False

        if level == PermissionLevel.SAFE:
            if self.auto_approve_safe:
                return True
            approved = True

        elif level == PermissionLevel.NOTIFY:
            self._notify(operation, reason)
            approved = True

        elif level == PermissionLevel.CONFIRM:
            if self.ask_fn is None:
                approved = True
            else:
                # 检查是否有同类型操作的临时授权
                tool_key = self._tool_key(operation.split("：")[0] if "：" in operation else operation)
                if self._has_temp_auth(tool_key):
                    approved = True
                else:
                    approved, timeout = self._ask_confirm(operation, reason, context_hint)

        elif level == PermissionLevel.DENY:
            if self.ask_fn is None:
                approved = False
            else:
                tool_key = self._tool_key(operation.split("：")[0] if "：" in operation else operation)
                # 永久授权覆盖
                if self._perm_approvals.get(tool_key):
                    approved = True
                elif self._has_temp_auth(tool_key):
                    approved = True
                else:
                    approved, timeout = self._ask_override(operation, reason, context_hint)

        self._log(ApprovalRecord(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            operation=operation[:200],
            category=category,
            level=level.value,
            approved=approved,
            reason=reason[:100],
            timeout=timeout,
            context=context_hint[:100],
        ))

        return approved

    # ── 用户交互 ──

    def _build_msg(self, operation: str, reason: str, context_hint: str) -> str:
        """构建带上下文的询问消息"""
        parts = [f"🔐 Agent 请求执行：\n{operation}"]
        if context_hint:
            parts.append(f"\n{context_hint}")
        if reason and reason not in context_hint:
            parts.append(f"\n理由：{reason}")
        return "\n".join(parts)

    def _ask_confirm(self, operation: str, reason: str, context_hint: str = "") -> tuple[bool, bool]:
        msg = self._build_msg(operation, reason, context_hint)
        msg += "\n\n是否允许？\n1) ✅ 允许\n2) ⏱ 允许本次会话\n3) ❌ 拒绝\n\n请输入编号 (1-3):"

        try:
            choice = self.ask_fn(msg, ["1", "2", "3"])
            if "2" in choice or "本次会话" in choice:
                tool_key = self._tool_key(
                    operation.split("：")[0] if "：" in operation else operation)
                self._temp_approvals[tool_key] = time.time() + self._temp_auth_ttl
                return True, False
            if "1" in choice or "允许" in choice:
                return True, False
            return False, False
        except Exception:
            return False, True

    def _ask_override(self, operation: str, reason: str, context_hint: str = "") -> tuple[bool, bool]:
        msg = self._build_msg(operation, reason, context_hint)
        msg += "\n\n此操作默认拒绝。请选择：\n1) 🚫 保持拒绝\n2) ⏱ 仅此一次（5分钟有效）\n3) 🔓 永久允许此类操作\n\n请输入编号 (1-3):"

        try:
            choice = self.ask_fn(msg, ["1", "2", "3"])
            tool_key = self._tool_key(
                operation.split("：")[0] if "：" in operation else operation)

            if "3" in choice or "永久" in choice:
                self._perm_approvals[tool_key] = True
                return True, False
            if "2" in choice or "仅此一次" in choice:
                self._temp_approvals[tool_key] = time.time() + self._temp_auth_ttl
                return True, False
            return False, False
        except Exception:
            return False, True

    def _notify(self, operation: str, reason: str = "") -> None:
        if self.ask_fn is None:
            return
        msg = f"ℹ️ Agent 正在执行：{operation}"
        if reason:
            msg += f"\n\n原因：{reason}\n\n输入 1 继续"
        try:
            self.ask_fn(msg, ["1"])
        except Exception:
            pass

    def _log(self, record: ApprovalRecord) -> None:
        self.audit_log.append(record)
        if len(self.audit_log) > self._max_log:
            self.audit_log.pop(0)

    # ── 公开接口 ──

    def recent_approvals(self, n: int = 10) -> list[ApprovalRecord]:
        return self.audit_log[-n:]

    def stats(self) -> dict[str, Any]:
        total = len(self.audit_log)
        approved = sum(1 for r in self.audit_log if r.approved)
        denied = sum(1 for r in self.audit_log if not r.approved)
        timed_out = sum(1 for r in self.audit_log if r.timeout)
        return {
            "total_checks": total,
            "approved": approved,
            "denied": denied,
            "timed_out": timed_out,
            "approval_rate": round(approved / total, 3) if total else 0,
            "temp_approvals_active": len(self._temp_approvals),
            "perm_approvals_count": len(self._perm_approvals),
        }

    def tool_call_blocker(self, tool_name: str, tool_args: dict) -> Optional[str]:
        if self.check_tool_call(tool_name, tool_args):
            return None
        return f"操作被用户拒绝：{tool_name}"

    def __repr__(self) -> str:
        return (
            f"HumanInTheLoop("
            f"ask_fn={'✓' if self.ask_fn else '✗'}, "
            f"temp_auth={len(self._temp_approvals)}, "
            f"perm_auth={len(self._perm_approvals)})"
        )
