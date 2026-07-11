"""测试 integration 集成层 — 权限拦截 + 自动 Eval 触发

验证：
  1. PermissionWrapper 能正确拦截/放行工具调用
  2. AutoEvalWrapper 能自动判断触发条件
  3. create_guarded_agent 一站式工厂能正确组装
  4. 触发条件查询接口正确
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integration.agent_wrapper import (
    PermissionWrapper,
    AutoEvalWrapper,
    create_guarded_agent,
)


# ══════════════════════════════════════════════
# 辅助
# ══════════════════════════════════════════════


def _mock_ask_approve(msg: str, choices: list[str]) -> str:
    """模拟用户批准"""
    return "1"


def _mock_ask_reject(msg: str, choices: list[str]) -> str:
    """模拟用户拒绝"""
    return "3"


def _mock_tool_call(name: str, args: dict = None) -> dict:
    return {
        "id": "call_001",
        "function": {
            "name": name,
            "arguments": json.dumps(args or {}),
        }
    }


def _original_execute(tc: dict) -> str:
    """模拟原始 execute_tool_call"""
    return json.dumps({"result": "ok"})


# ══════════════════════════════════════════════
# 测试
# ══════════════════════════════════════════════


def test_wrapper_safe_passthrough():
    """SAFE 工具直接放行"""
    pw = PermissionWrapper(ask_fn=_mock_ask_approve)
    wrapped = pw.wrap(_original_execute)

    tc = _mock_tool_call("get_time")
    result = json.loads(wrapped(tc))
    assert result["result"] == "ok"
    print(f"  ✅ SAFE 工具放行")


def test_wrapper_confirm_approved():
    """CONFIRM 工具用户批准后放行"""
    pw = PermissionWrapper(ask_fn=_mock_ask_approve)
    wrapped = pw.wrap(_original_execute)

    tc = _mock_tool_call("write_file", {"path": "/tmp/test.txt"})
    result = json.loads(wrapped(tc))
    assert result["result"] == "ok"
    print(f"  ✅ CONFIRM 工具批准后放行")


def test_wrapper_confirm_rejected():
    """CONFIRM 工具用户拒绝后拦截"""
    pw = PermissionWrapper(ask_fn=_mock_ask_reject)
    wrapped = pw.wrap(_original_execute)

    tc = _mock_tool_call("execute_python", {"code": "print(1)"})
    result = json.loads(wrapped(tc))
    assert result.get("blocked") is True
    assert pw.blocked_count == 1
    print(f"  ✅ CONFIRM 工具拒绝后拦截")


def test_wrapper_deny_default():
    """DENY 工具默认拒绝（无 ask_fn）"""
    pw = PermissionWrapper(ask_fn=None)
    wrapped = pw.wrap(_original_execute)

    tc = _mock_tool_call("delete_directory", {"path": "/"})
    result = json.loads(wrapped(tc))
    assert result.get("blocked") is True
    print(f"  ✅ DENY 工具默认拒绝")


def test_wrapper_no_ask_fn():
    """无 ask_fn 时 SAFE 放行 DENY 拒绝"""
    pw = PermissionWrapper(ask_fn=None)
    wrapped = pw.wrap(_original_execute)

    r1 = json.loads(wrapped(_mock_tool_call("get_time")))
    assert r1["result"] == "ok"

    r2 = json.loads(wrapped(_mock_tool_call("delete_directory", {})))
    assert r2.get("blocked") is True

    r3 = json.loads(wrapped(_mock_tool_call("write_file", {})))
    assert r3["result"] == "ok"

    print(f"  ✅ 无 ask_fn 正确放行/拒绝")


def test_wrapper_stats():
    """权限拦截统计"""
    pw = PermissionWrapper(ask_fn=_mock_ask_reject)
    wrapped = pw.wrap(_original_execute)

    wrapped(_mock_tool_call("get_time"))          # SAFE → 放行
    wrapped(_mock_tool_call("execute_python"))    # CONFIRM → mock 返回 3 → 拒绝

    stats = pw.stats
    assert stats["blocked_calls"] == 1
    assert stats["trace_errors"] == 0
    print(f"  ✅ 权限统计正确（拦截 {stats['blocked_calls']} 次）")


def test_auto_eval_should_eval():
    """AutoEvalWrapper.should_eval 分类正确"""
    # 生成式任务
    assert AutoEvalWrapper.should_eval("帮我写一份报告") is True
    assert AutoEvalWrapper.should_eval("分析一下Python和Java的优缺点") is True
    assert AutoEvalWrapper.should_eval("写一个排序算法") is True

    # 功能测试
    assert AutoEvalWrapper.should_eval("现在几点了？") is False
    assert AutoEvalWrapper.should_eval("计算 1+1") is False
    assert AutoEvalWrapper.should_eval("搜索今天的天气") is False
    print(f"  ✅ should_eval 分类正确")


def test_trigger_conditions():
    """触发条件接口"""
    conditions = AutoEvalWrapper.trigger_conditions()
    assert len(conditions) == 4
    gen = [c for c in conditions if c["condition"] == "生成式任务"]
    assert gen[0]["trigger"] is True
    func = [c for c in conditions if c["condition"] == "功能测试类任务"]
    assert func[0]["trigger"] is False
    print(f"  ✅ 触发条件接口返回正确（{len(conditions)} 条）")


def test_create_guarded_agent():
    """一站式工厂函数"""
    def dummy_agent(query: str) -> str:
        return f"结果: {query}"

    guarded = create_guarded_agent(
        agent_fn=dummy_agent,
        execute_tool_fn=_original_execute,
        ask_fn=_mock_ask_approve,
        enable_permissions=True,
        enable_auto_eval=False,  # 不传 judge_fn 时自动禁用
    )
    result = guarded("测试查询")
    assert "结果: 测试查询" in result
    print(f"  ✅ create_guarded_agent 工厂函数正常")


def test_wrapper_malformed_tool_call():
    """畸形的工具调用不应崩溃"""
    pw = PermissionWrapper(ask_fn=None)
    wrapped = pw.wrap(_original_execute)

    # 没有 function 字段
    result = wrapped({"id": "bad_call"})
    assert isinstance(result, str)

    # 参数不是 JSON
    tc = {"id": "1", "function": {"name": "get_time", "arguments": "not-json"}}
    result = json.loads(wrapped(tc))
    assert "error" in result or "result" in result
    print(f"  ✅ 畸形调用不崩溃")


if __name__ == "__main__":
    print("=" * 50)
    print("  Integration 集成层测试")
    print("=" * 50)

    tests = [
        ("SAFE 放行", test_wrapper_safe_passthrough),
        ("CONFIRM 批准", test_wrapper_confirm_approved),
        ("CONFIRM 拒绝", test_wrapper_confirm_rejected),
        ("DENY 默认拒绝", test_wrapper_deny_default),
        ("无 ask_fn", test_wrapper_no_ask_fn),
        ("权限统计", test_wrapper_stats),
        ("auto_eval 分类", test_auto_eval_should_eval),
        ("触发条件", test_trigger_conditions),
        ("工厂函数", test_create_guarded_agent),
        ("畸形调用", test_wrapper_malformed_tool_call),
    ]

    for name, test_fn in tests:
        test_fn()

    print(f"\n  ✅ 全部 {len(tests)} 个测试通过")
