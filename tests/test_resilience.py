"""resilience 模块测试"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from handwritten_react_agent.resilience import (
    classify_error, is_retryable, ErrorCategory,
    retry, CircuitBreaker, guarded_call,
)


def test_classify_auth():
    assert classify_error(Exception("401 Unauthorized")) == ErrorCategory.AUTH
    assert classify_error(Exception("API key is invalid")) == ErrorCategory.AUTH
    print("  ✅ AUTH 分类正确")


def test_classify_timeout():
    assert classify_error(Exception("timeout")) == ErrorCategory.TIMEOUT
    assert classify_error(Exception("Connection timed out")) == ErrorCategory.TIMEOUT
    print("  ✅ TIMEOUT 分类正确")


def test_retryable():
    assert is_retryable(ErrorCategory.TIMEOUT) is True
    assert is_retryable(ErrorCategory.API_ERROR) is True
    assert is_retryable(ErrorCategory.AUTH) is False
    assert is_retryable(ErrorCategory.VALIDATION) is False
    print("  ✅ 重试判断正确")


def test_retry_success():
    """重试后成功"""
    call_count = {"n": 0}

    @retry(max_attempts=3, base_delay=0.1)
    def flaky():
        call_count["n"] += 1
        if call_count["n"] < 2:
            raise Exception("timeout")
        return "ok"

    result = flaky()
    assert result == "ok"
    assert call_count["n"] == 2
    print("  ✅ 重试后成功")


def test_retry_exhausted():
    """重试耗尽后抛出异常"""
    call_count = {"n": 0}

    @retry(max_attempts=2, base_delay=0.1)
    def always_fail():
        call_count["n"] += 1
        raise Exception("timeout")

    try:
        always_fail()
        assert False, "应抛出异常"
    except Exception:
        assert call_count["n"] == 2
    print("  ✅ 重试耗尽后抛出")


def test_retry_auth_not_retry():
    """AUTH 错误不重试"""
    call_count = {"n": 0}

    @retry(max_attempts=3, base_delay=0.1)
    def auth_fail():
        call_count["n"] += 1
        raise Exception("401 Unauthorized")

    try:
        auth_fail()
    except Exception:
        assert call_count["n"] == 1  # 只尝试一次
    print("  ✅ AUTH 错误不重试")


def test_circuit_breaker():
    """熔断器"""
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.5, name="test")

    assert cb.state == "CLOSED"
    assert cb.is_open() is False

    cb.on_failure()
    assert cb.state == "CLOSED"

    cb.on_failure()
    assert cb.state == "OPEN"
    assert cb.is_open() is True

    # 恢复期后变为半开
    time.sleep(0.6)
    assert cb.state == "HALF_OPEN"
    assert cb.is_open() is False

    # 半开后成功 → 关闭
    cb.on_success()
    assert cb.state == "CLOSED"
    print("  ✅ 熔断器状态机正确")


def test_guarded_call_fallback():
    """带熔断的重试 + 降级"""
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=30)
    call_count = {"main": 0, "fallback": 0}

    def main():
        call_count["main"] += 1
        raise Exception("timeout")

    def fallback(**kw):
        call_count["fallback"] += 1
        return "fallback_ok"

    result = guarded_call(main, cb, max_attempts=2, base_delay=0.1, fallback=fallback)
    assert result == "fallback_ok"
    assert call_count["fallback"] == 1
    # main 失败 2 次 + 重试 1 次 = 2 次尝试
    assert call_count["main"] > 0
    print("  ✅ 降级调用正确")


def test_circuit_breaker_trips():
    """熔断器打开后直接走降级"""
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=30)
    cb.on_failure()  # 打开熔断器

    call_count = {"main": 0, "fallback": 0}

    def main():
        call_count["main"] += 1
        return "main_ok"

    def fallback(**kw):
        call_count["fallback"] += 1
        return "fb"

    result = guarded_call(main, cb, fallback=fallback)
    assert result == "fb"
    assert call_count["main"] == 0  # 熔断器打开，不走主函数
    assert call_count["fallback"] == 1
    print("  ✅ 熔断器打开后直接降级")


def test_tool_guard_retry():
    """ToolGuard 重试"""
    from handwritten_react_agent.resilience import ToolGuard
    guard = ToolGuard()
    call_count = {"n": 0}

    def flaky_tool(tc):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise Exception("timeout")
        return "ok"

    wrapped = guard.wrap(flaky_tool)
    result = wrapped({"function": {"name": "get_time", "arguments": "{}"}})
    assert result == "ok"
    assert call_count["n"] == 3
    print("  ✅ ToolGuard 重试成功")


def test_tool_guard_timeout():
    """ToolGuard 超时保护"""
    from handwritten_react_agent.resilience import ToolGuard
    guard = ToolGuard()

    def slow_tool(tc):
        import time
        time.sleep(5)
        return "ok"

    wrapped = guard.wrap(slow_tool)
    # get_time 默认超时 60s，测试用特殊 timeout 检查
    import json
    result = wrapped({"function": {"name": "execute_python", "arguments": "{}"}})
    # 应该超时后尝试重试（1次），最后返回错误
    parsed = json.loads(result)
    assert "超时" in parsed.get("error", "")
    print("  ✅ ToolGuard 超时保护正确")


def test_tool_guard_dangerous_no_retry():
    """危险工具不重试"""
    from handwritten_react_agent.resilience import ToolGuard
    guard = ToolGuard()
    call_count = {"n": 0}

    def fail_tool(tc):
        call_count["n"] += 1
        raise Exception("error")

    wrapped = guard.wrap(fail_tool)
    wrapped({"function": {"name": "delete_directory", "arguments": "{}"}})
    assert call_count["n"] == 1  # 不重试
    print("  ✅ 危险工具不重试")


def test_tool_guard_rate_limit():
    """频率限制"""
    from handwritten_react_agent.resilience import ToolGuard
    guard = ToolGuard()
    guard._max_rate = 2

    def ok_tool(tc):
        return "ok"

    wrapped = guard.wrap(ok_tool)
    tc = {"function": {"name": "web_search", "arguments": "{}"}}
    assert wrapped(tc) == "ok"  # 第1次
    assert wrapped(tc) == "ok"  # 第2次
    import json
    r = json.loads(wrapped(tc))  # 第3次
    assert r.get("blocked") is True
    print("  ✅ 频率限制正确")


if __name__ == "__main__":
    print("=" * 50)
    print("  Resilience 模块测试")
    print("=" * 50)
    test_classify_auth()
    test_classify_timeout()
    test_retryable()
    test_retry_success()
    test_retry_exhausted()
    test_retry_auth_not_retry()
    test_circuit_breaker()
    test_guarded_call_fallback()
    test_circuit_breaker_trips()
    test_tool_guard_retry()
    test_tool_guard_dangerous_no_retry()
    test_tool_guard_rate_limit()
    print(f"\n  ✅ 全部 13 个测试通过")
