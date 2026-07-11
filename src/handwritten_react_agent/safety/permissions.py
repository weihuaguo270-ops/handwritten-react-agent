"""permissions — 操作权限等级与工具分类

为 Agent 的每次操作定义风险等级，控制哪些操作需要人工审批。

等级划分：
  SAFE     — 纯读取，无需审批
  NOTIFY   — 读取敏感信息，通知用户但不阻塞
  CONFIRM  — 写入/执行，需用户确认
  DENY     — 破坏性操作，默认拒绝（除非用户明确覆盖）

v2 新增：参数级权限（Argument Rules）
  根据工具参数动态调整权限等级：
    write_file /tmp/* → SAFE（临时文件）
    write_file /etc/* → CONFIRM（系统配置）
    execute_python 含 os.system → 自动提升为 CONFIRM
"""
from __future__ import annotations
from enum import Enum
from typing import Any, Callable, Optional


class PermissionLevel(Enum):
    """操作权限等级"""
    SAFE = "safe"
    NOTIFY = "notify"
    CONFIRM = "confirm"
    DENY = "deny"


class Category(Enum):
    """操作分类"""
    TOOL_CALL = "tool_call"
    DIRECTION_CHANGE = "direction"
    RETRY = "retry"
    CORRECT = "correct"


# ── 工具权限表（工具名 → 默认等级） ──

TOOL_PERMISSIONS: dict[str, PermissionLevel] = {
    # SAFE
    "get_time": PermissionLevel.SAFE,
    "get_current_time": PermissionLevel.SAFE,
    "convert_time": PermissionLevel.SAFE,
    "calculator": PermissionLevel.SAFE,
    "web_search": PermissionLevel.SAFE,
    "fetch_page": PermissionLevel.SAFE,
    "summarize": PermissionLevel.SAFE,
    "rag_query": PermissionLevel.SAFE,
    "search_files": PermissionLevel.SAFE,
    "read_text_file": PermissionLevel.SAFE,
    "list_directory": PermissionLevel.SAFE,
    "directory_tree": PermissionLevel.SAFE,
    "list_allowed_directories": PermissionLevel.SAFE,

    # NOTIFY
    "get_file_info": PermissionLevel.NOTIFY,
    "read_env": PermissionLevel.NOTIFY,
    "mcp_list_tools": PermissionLevel.NOTIFY,
    "trajectory_replay": PermissionLevel.NOTIFY,
    "dashboard_query": PermissionLevel.NOTIFY,
    "get_memory": PermissionLevel.NOTIFY,
    "search_memory": PermissionLevel.NOTIFY,

    # CONFIRM
    "write_file": PermissionLevel.CONFIRM,
    "patch_file": PermissionLevel.CONFIRM,
    "execute_python": PermissionLevel.CONFIRM,
    "execute_command": PermissionLevel.CONFIRM,
    "send_email": PermissionLevel.CONFIRM,
    "mcp_call_tool": PermissionLevel.CONFIRM,
    "save_memory": PermissionLevel.CONFIRM,
    "update_memory": PermissionLevel.CONFIRM,
    "delete_file": PermissionLevel.CONFIRM,
    "create_file": PermissionLevel.CONFIRM,

    # DENY
    "delete_directory": PermissionLevel.DENY,
    "format_disk": PermissionLevel.DENY,
    "shutdown": PermissionLevel.DENY,
    "restart": PermissionLevel.DENY,
    "modify_system_config": PermissionLevel.DENY,
    "install_package": PermissionLevel.DENY,
    "uninstall_package": PermissionLevel.DENY,
}

# ── 方向调整权限 ──

DIRECTION_CHANGE_LEVELS: dict[str, PermissionLevel] = {
    "修正指令注入": PermissionLevel.CONFIRM,
    "重新执行步骤": PermissionLevel.CONFIRM,
    "更换 Provider 重试": PermissionLevel.CONFIRM,
    "跳过失败步骤": PermissionLevel.CONFIRM,
    "终止执行": PermissionLevel.CONFIRM,
    "调整温度参数": PermissionLevel.NOTIFY,
    "调整最大步数": PermissionLevel.NOTIFY,
    "增加测试用例": PermissionLevel.SAFE,
}


# ── 参数级规则（v2） ──
# (工具名, 参数匹配函数, 覆盖后的等级)
# 匹配函数返回 True 时，使用该等级覆盖默认等级
# 从上到下匹配，返回第一个匹配的规则

ArgChecker = Callable[[dict[str, Any]], bool]


def _path_contains(substring: str) -> ArgChecker:
    """参数中的 path 包含指定子串"""
    def check(args: dict) -> bool:
        path = str(args.get("path", "") or args.get("filepath", "") or "")
        return substring in path
    return check


def _code_contains(keyword: str) -> ArgChecker:
    """代码参数包含指定关键词"""
    def check(args: dict) -> bool:
        code = str(args.get("code", "") or args.get("command", "") or "")
        return keyword in code
    return check


def _key_contains(keyword: str) -> ArgChecker:
    """任意参数值包含指定关键词"""
    def check(args: dict) -> bool:
        for v in args.values():
            if keyword in str(v).lower():
                return True
        return False
    return check


# 参数级规则表
ARG_RULES: list[tuple[str, ArgChecker, PermissionLevel]] = [
    # --- 敏感内容优先检测（高于路径规则）---
    ("write_file", _key_contains("password"), PermissionLevel.DENY),
    ("write_file", _key_contains("secret"), PermissionLevel.DENY),
    ("write_file", _key_contains("api_key"), PermissionLevel.DENY),

    # write_file 路径级
    ("write_file", _path_contains("/tmp/"), PermissionLevel.SAFE),
    ("write_file", _path_contains("/Temp/"), PermissionLevel.SAFE),
    ("write_file", _path_contains("temp"), PermissionLevel.SAFE),
    ("write_file", _path_contains("/etc/"), PermissionLevel.CONFIRM),
    ("write_file", _path_contains("/usr/"), PermissionLevel.CONFIRM),

    # execute_python 内容级
    ("execute_python", _code_contains("os.system"), PermissionLevel.CONFIRM),
    ("execute_python", _code_contains("subprocess"), PermissionLevel.CONFIRM),
    ("execute_python", _code_contains("shutil.rmtree"), PermissionLevel.DENY),
    ("execute_python", _code_contains("os.remove"), PermissionLevel.CONFIRM),
    ("execute_python", _code_contains("__import__"), PermissionLevel.CONFIRM),
]


# ── 查询函数 ──


def get_tool_permission(
    tool_name: str,
    tool_args: Optional[dict] = None,
) -> PermissionLevel:
    """获取工具的权限等级（支持参数级覆盖）

    先检查参数级规则，未命中则返回默认等级。
    不在表中的工具默认 SAFE。
    """
    if tool_args:
        for name, checker, level in ARG_RULES:
            if name == tool_name and checker(tool_args):
                return level

    return TOOL_PERMISSIONS.get(tool_name, PermissionLevel.SAFE)


def get_direction_permission(action: str) -> PermissionLevel:
    return DIRECTION_CHANGE_LEVELS.get(action, PermissionLevel.CONFIRM)


def is_high_risk(level: PermissionLevel) -> bool:
    return level in (PermissionLevel.CONFIRM, PermissionLevel.DENY)


def describe_action(tool_name: str, args: Optional[dict] = None) -> str:
    """生成操作的可读描述"""
    desc = f"调用工具：{tool_name}"
    if args:
        args_summary = {}
        for k, v in (args or {}).items():
            if isinstance(v, str) and len(v) > 100:
                args_summary[k] = v[:50] + "..."
            elif k.lower() in ("password", "secret", "key", "token", "api_key"):
                args_summary[k] = "******"
            else:
                args_summary[k] = v
        desc += f"\n  参数：{args_summary}"
    return desc
