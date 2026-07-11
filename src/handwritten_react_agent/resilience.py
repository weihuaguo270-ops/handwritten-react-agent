"""resilience — Agent 运行时容错基础设施

提供重试、熔断、错误分类、降级策略。
不依赖任何外部框架，纯 Python 实现。
"""
from __future__ import annotations
import json
import time
import random
import functools
from typing import Any, Callable, Optional


# ══════════════════════════════════════════════
#  错误分类
# ══════════════════════════════════════════════

class ErrorCategory:
    """错误分类"""
    AUTH = "auth"              # API key / 权限问题（不可重试）
    RATE_LIMIT = "rate_limit"  # 频率限制（等待后可重试）
    TIMEOUT = "timeout"        # 超时（可重试）
    API_ERROR = "api_error"    # API 临时故障（可重试）
    TOOL_ERROR = "tool_error"  # 工具执行报错（可重试）
    VALIDATION = "validation"  # 参数校验失败（不可重试）
    NETWORK = "network"        # 网络错误（可重试）
    UNKNOWN = "unknown"        # 未知


def classify_error(error: Exception | str) -> str:
    """将异常分类"""
    msg = str(error).lower()

    if any(k in msg for k in ("401", "unauthorized", "auth", "api key")):
        return ErrorCategory.AUTH
    if any(k in msg for k in ("429", "rate limit", "too many")):
        return ErrorCategory.RATE_LIMIT
    if any(k in msg for k in ("timeout", "timed out")):
        return ErrorCategory.TIMEOUT
    if any(k in msg for k in ("connection", "dns", "resolve")):
        return ErrorCategory.NETWORK
    if any(k in msg for k in ("参数解析", "validation", "invalid")):
        return ErrorCategory.VALIDATION
    if any(k in msg for k in ("500", "502", "503", "service")):
        return ErrorCategory.API_ERROR
    return ErrorCategory.UNKNOWN


def is_retryable(category: str) -> bool:
    """该分类的错误是否可以重试"""
    return category in (
        ErrorCategory.RATE_LIMIT,
        ErrorCategory.TIMEOUT,
        ErrorCategory.API_ERROR,
        ErrorCategory.NETWORK,
        ErrorCategory.TOOL_ERROR,
        ErrorCategory.UNKNOWN,
    )


# ══════════════════════════════════════════════
#  指数退避重试
# ══════════════════════════════════════════════

def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_only: bool = True,
    on_retry: Optional[Callable] = None,
):
    """可重试的装饰器（指数退避 + 随机抖动）

    参数:
        max_attempts:  最大尝试次数（默认 3）
        base_delay:    初始延迟秒数（默认 1s）
        max_delay:     最大延迟秒数（默认 30s）
        retryable_only: 只重试可恢复的错误（默认 True）
        on_retry:      每次重试前的回调

    用法:
        @retry(max_attempts=3, base_delay=1.0)
        def call_llm(messages):
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    category = classify_error(e)

                    # 不可重试的错误直接抛出
                    if retryable_only and not is_retryable(category):
                        raise

                    # 最后一次尝试不再等待
                    if attempt == max_attempts:
                        raise

                    # 计算等待时间（指数退避 + 随机抖动）
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    jitter = random.uniform(0, delay * 0.3)
                    wait = delay + jitter

                    if on_retry:
                        on_retry(attempt, max_attempts, wait, category, str(e))

                    time.sleep(wait)

            raise last_error  # 不会执行到，但满足类型检查
        return wrapper
    return decorator


# ══════════════════════════════════════════════
#  熔断器
# ══════════════════════════════════════════════

class CircuitBreaker:
    """熔断器 — 连续失败后暂停调用

    状态机:
        CLOSED → OPEN （连续失败达到阈值）
        OPEN → HALF_OPEN （超时后允许一次试探）
        HALF_OPEN → CLOSED （试探成功）
        HALF_OPEN → OPEN （试探失败，重置计时）

    用法:
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
        for i in range(10):
            if breaker.is_open():
                print("熔断器已打开，跳过")
                continue
            try:
                risky_call()
                breaker.on_success()
            except Exception:
                breaker.on_failure()
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        name: str = "",
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.name = name
        self._failure_count = 0
        self._state = "CLOSED"  # CLOSED / OPEN / HALF_OPEN
        self._last_failure_time = 0.0
        self._stats = {"tripped": 0, "recovered": 0}

    @property
    def state(self) -> str:
        """当前状态"""
        if self._state == "OPEN":
            if time.time() - self._last_failure_time > self.recovery_timeout:
                self._state = "HALF_OPEN"
        return self._state

    def is_open(self) -> bool:
        """是否应阻止调用"""
        return self.state == "OPEN"

    def on_success(self):
        """调用成功"""
        if self._state == "HALF_OPEN":
            self._state = "CLOSED"
            self._failure_count = 0
            self._stats["recovered"] += 1
        self._failure_count = 0

    def on_failure(self):
        """调用失败"""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._failure_count >= self.failure_threshold:
            if self._state != "OPEN":
                self._state = "OPEN"
                self._stats["tripped"] += 1

    def reset(self):
        """手动重置熔断器"""
        self._state = "CLOSED"
        self._failure_count = 0

    @property
    def stats(self) -> dict:
        return {**self._stats, "state": self._state, "failures": self._failure_count}


# ══════════════════════════════════════════════
#  降级策略
# ══════════════════════════════════════════════

class FallbackStrategy:
    """降级策略注册表

    当主方案失败时，按优先级尝试备选方案。

    用法:
        strategy = FallbackStrategy()
        strategy.register("web_search", [
            ("web_search", {}),       # 默认
            ("web_search", {"engine": "bing"}),  # 备选
            ("fetch_page", {}),       # 兜底
        ])
        result = strategy.execute("web_search", query="...")
    """

    def __init__(self):
        self._chains: dict[str, list[tuple[str, dict]]] = {}

    def register(self, fallback_id: str, chains: list[tuple[str, dict]]):
        """注册降级链

        参数:
            fallback_id: 降级策略 ID
            chains: (方案名, 参数覆盖) 列表，按优先级排列
        """
        self._chains[fallback_id] = chains

    def execute(self, fallback_id: str, **kwargs) -> Any:
        """按优先级执行降级链

        第一个成功的方案返回，全部失败则抛出最后一个异常
        """
        chain = self._chains.get(fallback_id)
        if not chain:
            raise ValueError(f"未注册降级策略: {fallback_id}")

        last_error = None
        for plan_name, overrides in chain:
            try:
                params = {**kwargs, **overrides}
                # 查找并调用对应的函数
                func = globals().get(f"_{fallback_id}_{plan_name}")
                if func:
                    return func(**params)
            except Exception as e:
                last_error = e
                continue

        raise last_error or RuntimeError(f"降级链全部失败: {fallback_id}")


# ══════════════════════════════════════════════
#  便捷函数：带熔断的重试
# ══════════════════════════════════════════════

def guarded_call(
    func: Callable,
    breaker: CircuitBreaker,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    fallback: Optional[Callable] = None,
    **kwargs,
) -> Any:
    """带熔断保护 + 重试 + 降级的统一调用

    参数:
        func:     主调用函数
        breaker:  熔断器实例
        max_attempts: 最大重试次数
        base_delay:    重试间隔
        fallback:      降级函数（主函数失败时调用）
        kwargs:        传给 func 和 fallback 的参数

    返回:
        func 或 fallback 的返回值

    如果熔断器已打开 → 直接走 fallback
    如果 func 重试耗尽 → 走 fallback
    如果 fallback 也失败 → 抛出原始异常
    """
    # 熔断器已打开
    if breaker.is_open():
        if fallback:
            return fallback(**kwargs)
        raise RuntimeError(f"熔断器已打开: {breaker.name}")

    # 主调用（带重试）
    decorator = retry(max_attempts=max_attempts, base_delay=base_delay)
    try:
        result = decorator(func)(**kwargs)
        breaker.on_success()
        return result
    except Exception as e:
        breaker.on_failure()
        if fallback:
            return fallback(**kwargs)
        raise


# ══════════════════════════════════════════════
#  ToolGuard — 工具调用统一保护层
# ══════════════════════════════════════════════

class ToolGuard:
    """工具调用保护层 — 按安全级别分级保护

    安全边界规则：
      DENY 工具    → 不重试、不熔断（走 HITL，不由 resilience 干预）
      CONFIRM 工具 → 最多重试 1 次、无熔断（写/执行操作不能自动重试）
      SAFE 工具    → 重试 3 次、有熔断（只读操作可以安全重试）
      未知工具     → 默认 SAFE 级别

    用法：
        guard = ToolGuard()
        guard.wrap(original_execute_tool_call)
        # 替换 react_loop 中的 execute_tool_call
    """

    # 工具安全级别（简化版，与 permissions.py 保持一致）
    _DANGEROUS = {"delete_directory", "format_disk", "shutdown", "restart",
                  "modify_system_config", "install_package", "uninstall_package"}
    _WRITE = {"write_file", "patch_file", "execute_python", "execute_command",
              "send_email", "delete_file", "create_file",
              "save_memory", "update_memory", "mcp_call_tool"}

    # 工具超时（秒）
    _TOOL_TIMEOUTS = {
        "web_search": 30,
        "fetch_page": 30,
        "execute_python": 30,
        "calculator": 10,
        "default": 60,
    }

    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._rate_tracker: dict[str, list[float]] = {}
        self._max_rate = 10

    def _get_breaker(self, tool_name: str) -> CircuitBreaker:
        if tool_name not in self._breakers:
            if tool_name in self._DANGEROUS:
                # 危险工具：阈值极高，永不熔断（走 HITL）
                self._breakers[tool_name] = CircuitBreaker(
                    failure_threshold=999, recovery_timeout=999, name=tool_name)
            else:
                self._breakers[tool_name] = CircuitBreaker(
                    failure_threshold=3, recovery_timeout=30, name=tool_name)
        return self._breakers[tool_name]

    def _check_rate_limit(self, tool_name: str) -> bool:
        now = time.time()
        if tool_name not in self._rate_tracker:
            self._rate_tracker[tool_name] = []
        self._rate_tracker[tool_name] = [
            t for t in self._rate_tracker[tool_name] if now - t < 60
        ]
        if len(self._rate_tracker[tool_name]) >= self._max_rate:
            return False
        self._rate_tracker[tool_name].append(now)
        return True

    def _max_retries(self, tool_name: str) -> int:
        if tool_name in self._DANGEROUS:
            return 0
        if tool_name in self._WRITE:
            return 1
        return 3

    def wrap(self, original_fn: Callable) -> Callable:
        """包装 execute_tool_call"""
        def wrapped(tool_call: dict) -> str:
            if not isinstance(tool_call, dict) or "function" not in tool_call:
                return '{"error": "畸形工具调用", "blocked": true}'
            name = tool_call["function"].get("name", "unknown")
            breaker = self._get_breaker(name)

            # 频率限制
            if not self._check_rate_limit(name):
                return f'{{"error": "工具 {name} 调用频繁", "blocked": true}}'

            # 熔断检查
            if breaker.is_open():
                return f'{{"error": "工具 {name} 暂不可用", "blocked": true}}'

            max_retries = self._max_retries(name)
            timeout = self._TOOL_TIMEOUTS.get(name, 60)
            last_error = ""

            for attempt in range(max_retries + 1):
                try:
                    result = _call_with_timeout(original_fn, tool_call, timeout)
                    breaker.on_success()
                    return result
                except TimeoutError:
                    last_error = f"超时 ({timeout}s)"
                    breaker.on_failure()
                    if attempt < max_retries:
                        time.sleep(0.5)
                except Exception as e:
                    last_error = str(e)[:100]
                    cat = classify_error(e)
                    breaker.on_failure()
                    if not is_retryable(cat) or attempt >= max_retries:
                        return json.dumps({"error": last_error, "blocked": False})
                    time.sleep(0.5)

            return json.dumps({"error": last_error, "blocked": False,
                               "retry_exhausted": True})
        return wrapped


def _call_with_timeout(func, arg, timeout):
    """带超时的函数调用"""
    import threading
    result = [None]
    error = [None]
    done = threading.Event()

    def runner():
        try:
            result[0] = func(arg)
        except Exception as e:
            error[0] = e
        finally:
            done.set()

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    if not done.wait(timeout=timeout):
        raise TimeoutError(f"超时 ({timeout}s)")
    if error[0]:
        raise error[0]
    return result[0]
