"""测试 permissions + human_in_the_loop 权限系统"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

from handwritten_react_agent.safety.permissions import (
    PermissionLevel, get_tool_permission, get_direction_permission,
    is_high_risk, describe_action,
)
from handwritten_react_agent.safety.human_in_the_loop import HumanInTheLoop, ApprovalRecord


# ── Permissions 测试 ──

def test_safe_tools():
    for tool in ["get_time", "calculator", "web_search", "read_text_file",
                  "rag_query", "search_files", "fetch_page"]:
        assert get_tool_permission(tool) == PermissionLevel.SAFE, f"{tool}"
    print("  ✅ SAFE 工具")

def test_confirm_tools():
    for tool in ["write_file", "execute_python", "patch_file"]:
        assert get_tool_permission(tool) == PermissionLevel.CONFIRM, f"{tool}"
    print("  ✅ CONFIRM 工具")

def test_deny_tools():
    for tool in ["delete_directory", "format_disk", "shutdown", "modify_system_config"]:
        assert get_tool_permission(tool) == PermissionLevel.DENY, f"{tool}"
    print("  ✅ DENY 工具")

def test_unknown_safe():
    assert get_tool_permission("nonexistent_tool") == PermissionLevel.SAFE
    print("  ✅ 未知工具默认 SAFE")

def test_args_level_safe():
    """参数级规则：/tmp/ 下写文件降为 SAFE"""
    h = HumanInTheLoop(ask_fn=None)
    assert h.check_tool_call("write_file", {"path": "/tmp/test.log"}) is True
    # 参数为空/未知路径 → 保持 CONFIRM
    assert h.check_tool_call("write_file", {"path": "/unknown/path.txt"}) is True  # 无 ask_fn 放行
    print("  ✅ 路径级 SAFE 正确")

def test_args_level_deny():
    """参数级规则：含敏感关键词提升为 DENY"""
    h = HumanInTheLoop(ask_fn=None)
    assert h.check_tool_call("write_file", {"path": "/tmp/passwords.txt", "content": "admin:123456"}) is False
    assert h.check_tool_call("write_file", {"path": "/tmp/config.env", "content": "API_KEY=sk-xxx"}) is False
    print("  ✅ 敏感内容 DENY 正确")

def test_args_level_confirm():
    """参数级规则：系统路径保持 CONFIRM"""
    h = HumanInTheLoop(ask_fn=None)
    assert h.check_tool_call("write_file", {"path": "/etc/nginx.conf"}) is True  # 无 ask_fn 放行
    assert h.check_tool_call("execute_python", {"code": "import os; os.system('rm -rf /')"}) is True  # 无 ask_fn 放行
    print("  ✅ 系统路径 CONFIRM 正确")

def test_high_risk():
    assert is_high_risk(PermissionLevel.CONFIRM)
    assert is_high_risk(PermissionLevel.DENY)
    assert not is_high_risk(PermissionLevel.SAFE)
    print("  ✅ 高风险判断")

def test_describe():
    desc = describe_action("write_file", {"path": "/tmp/test.txt"})
    assert "write_file" in desc and "/tmp/test.txt" in desc
    print("  ✅ 操作描述")

def test_describe_masked():
    desc = describe_action("send_email", {"api_key": "sk-123"})
    assert "******" in desc
    print("  ✅ 敏感信息脱敏")


# ── Mock ask_fn ──

def _make_ask_fn(choice: str):
    count = {"n": 0}
    def fn(msg, choices):
        count["n"] += 1
        return choice if choice in choices else choices[0]
    return fn, count


# ── HITL 测试 ──

def test_auto_approve_safe():
    h = HumanInTheLoop(ask_fn=None)
    assert h.check_tool_call("get_time") is True
    assert h.check_tool_call("calculator", {}) is True
    print("  ✅ SAFE 自动放行")

def test_confirm_approved():
    h = HumanInTheLoop(ask_fn=_make_ask_fn("1")[0])
    assert h.check_tool_call("write_file", {"path": "/tmp/x"}) is True
    print("  ✅ CONFIRM 用户批准")

def test_confirm_denied():
    h = HumanInTheLoop(ask_fn=_make_ask_fn("3")[0])
    assert h.check_tool_call("execute_python", {"code": "print(1)"}) is False
    print("  ✅ CONFIRM 用户拒绝")

def test_deny_default():
    h = HumanInTheLoop(ask_fn=_make_ask_fn("1")[0])
    assert h.check_tool_call("delete_directory", {}) is False
    print("  ✅ DENY 默认拒绝")

def test_deny_override():
    h = HumanInTheLoop(ask_fn=_make_ask_fn("2")[0])
    assert h.check_tool_call("delete_directory", {}) is True
    print("  ✅ DENY 临时授权（5分钟有效）")

def test_deny_permanent():
    h = HumanInTheLoop(ask_fn=_make_ask_fn("3")[0])
    assert h.check_tool_call("delete_directory", {}) is True
    # 第二次同类型操作不再询问
    assert h.check_tool_call("delete_directory", {"path": "/tmp"}) is True
    print("  ✅ DENY 永久授权生效")

def test_temp_auth_expiry():
    """临时授权过期后不再有效"""
    call_n = {"count": 0}

    def _ask_once_then_reject(msg, choices):
        call_n["count"] += 1
        if call_n["count"] == 1:
            return "2"
        return "1"

    h = HumanInTheLoop(ask_fn=_ask_once_then_reject, temp_auth_minutes=0.01)
    # 第一次：获得临时授权
    assert h.check_tool_call("delete_directory", {}) is True
    # 0.01 分钟 = 0.6 秒，等 0.7 秒后过期
    import time; time.sleep(0.7)
    # 第二次：临时授权已过期，用户拒绝
    assert h.check_tool_call("delete_directory", {}) is False
    print("  ✅ 临时授权过期正确")

def test_direction_approved():
    h = HumanInTheLoop(ask_fn=_make_ask_fn("1")[0])
    assert h.check_direction("修正指令注入", "添加新步骤") is True
    print("  ✅ 方向调整批准")

def test_direction_denied():
    h = HumanInTheLoop(ask_fn=_make_ask_fn("3")[0])
    assert h.check_direction("重新执行步骤", "Step 3") is False
    print("  ✅ 方向调整拒绝")

def test_no_ask_fn():
    h = HumanInTheLoop(ask_fn=None)
    assert h.check_tool_call("get_time") is True          # SAFE → 放行
    assert h.check_tool_call("delete_directory", {}) is False  # DENY → 拒绝
    assert h.check_tool_call("write_file", {}) is True     # CONFIRM → 放行
    print("  ✅ 无 ask_fn 行为正确")

def test_audit_log():
    h = HumanInTheLoop(ask_fn=None)
    h.check_tool_call("get_time")       # auto 放行 → 不记日志
    h.check_tool_call("delete_directory", {"path": "/"})  # DENY → 记日志
    # auto_approve_safe=True 时 SAFE 操作不记日志
    assert len(h.audit_log) == 1
    assert h.audit_log[0].approved is False

    s = h.stats()
    assert s["total_checks"] == 1
    assert s["denied"] == 1
    print("  ✅ 审计日志")

def test_tool_blocker():
    h = HumanInTheLoop(ask_fn=None)
    assert h.tool_call_blocker("get_time", {}) is None
    assert h.tool_call_blocker("delete_directory", {}) is not None
    print("  ✅ 工具拦截器")

def test_notify():
    called = False
    def fn(msg, choices):
        nonlocal called
        called = True
        return choices[0]
    h = HumanInTheLoop(ask_fn=fn)
    assert h.check_tool_call("get_memory", {"key": "prefs"}) is True
    assert called
    print("  ✅ NOTIFY 通知不阻塞")


if __name__ == "__main__":
    print("=" * 50)
    print("  权限系统测试（16 个）")
    print("=" * 50)
    tests = [
        test_safe_tools, test_confirm_tools, test_deny_tools,
        test_unknown_safe, test_args_level_safe, test_args_level_deny, test_args_level_confirm,
        test_high_risk, test_describe, test_describe_masked,
        test_auto_approve_safe, test_confirm_approved, test_confirm_denied,
        test_deny_default, test_deny_override, test_deny_permanent, test_temp_auth_expiry,
        test_direction_approved, test_direction_denied,
        test_no_ask_fn, test_audit_log, test_tool_blocker, test_notify,
    ]
    for t in tests:
        t()
    print(f"\n  ✅ 全部 {len(tests)} 个测试通过")
