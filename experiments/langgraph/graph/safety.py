"""
安全与权限 — 为 LangGraph Agent 提供工具调用权限检查和 HITL

从手写版 src/handwritten_react_agent/safety/ 迁移，适配 LangGraph 的 tool 调用方式。
"""
from __future__ import annotations
import time
from typing import Any, Optional, Callable
from langchain_core.tools import tool as langchain_tool


class PermissionLevel:
    SAFE = "safe"
    NOTIFY = "notify"
    CONFIRM = "confirm"
    DENY = "deny"


# ── 工具权限表 ──

TOOL_PERMISSIONS: dict[str, str] = {
    # SAFE
    "get_current_time": PermissionLevel.SAFE,
    "calculator": PermissionLevel.SAFE,
    "web_search": PermissionLevel.SAFE,
    "fetch_page": PermissionLevel.SAFE,
    "rag_query": PermissionLevel.SAFE,
    "web_rag": PermissionLevel.SAFE,
    # NOTIFY
    "get_memory": PermissionLevel.NOTIFY,
    "search_memory": PermissionLevel.NOTIFY,
    # CONFIRM
    "execute_python": PermissionLevel.CONFIRM,
    "write_file": PermissionLevel.CONFIRM,
    "patch_file": PermissionLevel.CONFIRM,
    "delete_file": PermissionLevel.CONFIRM,
    # DENY
    "delete_directory": PermissionLevel.DENY,
    "shutdown": PermissionLevel.DENY,
    "install_package": PermissionLevel.DENY,
}


def get_tool_permission(tool_name: str) -> str:
    """获取工具的默认权限等级"""
    return TOOL_PERMISSIONS.get(tool_name, PermissionLevel.SAFE)


def is_high_risk(level: str) -> bool:
    return level in (PermissionLevel.CONFIRM, PermissionLevel.DENY)


# ── 审批管理器 ──

class HumanInTheLoop:
    """人工审批管理器（简化版，适配 LangGraph 的工具调用模式）"""

    def __init__(
        self,
        ask_fn: Optional[Callable[[str, list[str]], str]] = None,
        auto_approve_safe: bool = True,
    ):
        self.ask_fn = ask_fn
        self.auto_approve_safe = auto_approve_safe
        self._temp_approvals: dict[str, float] = {}
        self._perm_approvals: dict[str, bool] = {}
        self.audit_log: list[dict] = []
        self._temp_auth_ttl = 300  # 5 分钟

    def check_tool_call(self, tool_name: str, tool_args: Optional[dict] = None) -> tuple[bool, str]:
        """检查工具调用是否允许。返回 (允许, 理由)"""
        level = get_tool_permission(tool_name)
        key = f"tool:{tool_name}"

        if level == PermissionLevel.SAFE and self.auto_approve_safe:
            return True, ""

        if key in self._perm_approvals:
            return True, "已永久授权"

        if key in self._temp_approvals and time.time() < self._temp_approvals[key]:
            return True, "临时授权有效"

        if level == PermissionLevel.DENY:
            if self.ask_fn is None:
                return False, "该操作默认拒绝"
            return self._ask_override(tool_name, tool_args)

        if level == PermissionLevel.CONFIRM:
            if self.ask_fn is None:
                return True, ""
            return self._ask_confirm(tool_name, tool_args)

        return True, ""

    def _ask_confirm(self, tool_name: str, tool_args: Optional[dict]) -> tuple[bool, str]:
        if not self.ask_fn:
            return True, ""
        msg = f"🔐 Agent 请求调用工具：{tool_name}"
        if tool_args:
            msg += f"\n参数：{tool_args}"
        msg += "\n\n1) ✅ 允许\n2) ⏱ 本次会话\n3) ❌ 拒绝"
        try:
            choice = self.ask_fn(msg, ["1", "2", "3"])
            if "2" in choice:
                self._temp_approvals[f"tool:{tool_name}"] = time.time() + self._temp_auth_ttl
                return True, "临时授权（5分钟）"
            if "1" in choice:
                return True, "用户确认"
            return False, "用户拒绝"
        except Exception:
            return False, "审批超时"

    def _ask_override(self, tool_name: str, tool_args: Optional[dict]) -> tuple[bool, str]:
        if not self.ask_fn:
            return False, ""
        msg = f"🚫 高风险操作：{tool_name}\n参数：{tool_args}\n\n1) 🚫 保持拒绝\n2) ⏱ 仅此一次\n3) 🔓 永久允许"
        try:
            choice = self.ask_fn(msg, ["1", "2", "3"])
            if "3" in choice:
                self._perm_approvals[f"tool:{tool_name}"] = True
                return True, "永久授权"
            if "2" in choice:
                self._temp_approvals[f"tool:{tool_name}"] = time.time() + self._temp_auth_ttl
                return True, "临时授权（5分钟）"
            return False, "用户拒绝"
        except Exception:
            return False, "审批超时"

    def wrap_tool(self, fn: Callable, tool_name: str) -> Callable:
        """包装工具函数，调用前先做权限检查"""
        def wrapped(**kwargs):
            allowed, reason = self.check_tool_call(tool_name, kwargs)
            self.audit_log.append({
                "tool": tool_name,
                "allowed": allowed,
                "reason": reason,
                "timestamp": time.strftime("%H:%M:%S"),
            })
            if not allowed:
                return f"操作被拒绝：{reason}"
            return fn(**kwargs)
        return wrapped

    def stats(self) -> dict:
        total = len(self.audit_log)
        approved = sum(1 for r in self.audit_log if r["allowed"])
        return {
            "total": total,
            "approved": approved,
            "denied": total - approved,
            "temp_active": len(self._temp_approvals),
            "perm_count": len(self._perm_approvals),
        }
