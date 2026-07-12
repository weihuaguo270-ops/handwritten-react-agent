"""
容错与重试 — 为 LangGraph Agent 提供 LLM 调用重试和错误分类

从手写版 src/handwritten_react_agent/resilience.py 迁移核心逻辑，适配 LangChain 调用方式。
"""
from __future__ import annotations
import time
import functools
from typing import Any, Callable, Optional


class ErrorCategory:
    AUTH = "auth"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    API_ERROR = "api_error"
    NETWORK = "network"
    VALIDATION = "validation"
    TOOL_ERROR = "tool_error"
    UNKNOWN = "unknown"


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
    if any(k in msg for k in ("500", "502", "503", "service")):
        return ErrorCategory.API_ERROR
    if any(k in msg for k in ("command not found", "not found", "no such")):
        return ErrorCategory.TOOL_ERROR
    return ErrorCategory.UNKNOWN


def retry_call(
    func: Callable,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_only: bool = True,
    on_retry: Optional[Callable] = None,
) -> Callable:
    """可重试的函数包装器（指数退避 + 随机抖动）

    用法：
        safe_llm_call = retry_call(llm.invoke, max_attempts=3)
        result = safe_llm_call(messages)
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                category = classify_error(e)

                if retryable_only and category in (
                    ErrorCategory.AUTH, ErrorCategory.VALIDATION,
                    ErrorCategory.TOOL_ERROR,
                ):
                    raise

                if attempt == max_attempts:
                    raise

                delay = min(base_delay * (2 ** (attempt - 1)) + 0.1 * hash(str(e)) % 0.3, max_delay)

                category_label = {
                    ErrorCategory.RATE_LIMIT: "频率限制",
                    ErrorCategory.TIMEOUT: "超时",
                    ErrorCategory.API_ERROR: "API故障",
                    ErrorCategory.NETWORK: "网络错误",
                    ErrorCategory.UNKNOWN: "未知错误",
                }.get(category, "错误")

                msg = f"[重试 {attempt}/{max_attempts}] {category_label}: {str(e)[:60]} → 等待 {delay:.1f}s"
                if on_retry:
                    on_retry(msg)
                else:
                    print(msg)

                time.sleep(delay)

        raise last_error  # type: ignore

    return wrapper


def with_retry(llm_callable):
    """快捷方式：给 LLM 调用对象添加重试能力

    用法：
        llm = with_retry(get_llm())
        result = llm.invoke(messages)  # 自动重试
    """
    original_invoke = llm_callable.invoke

    def safe_invoke(*args, **kwargs):
        wrapped = retry_call(original_invoke, max_attempts=3, base_delay=1.0)
        return wrapped(*args, **kwargs)

    llm_callable.invoke = safe_invoke
    return llm_callable
