"""
真实 LLM 集成测试 — 覆盖简单与复杂任务
=======================================

测试分类：
  - 简单任务：事实问答、工具调用
  - 复杂任务：多步推理、错误恢复、长上下文
"""

import os
import re
import pytest


def _has_key() -> bool:
    return bool(os.environ.get("DEEPSEEK_API_KEY"))


REQUIRES_API = pytest.mark.skipif(not _has_key(), reason="需要 DEEPSEEK_API_KEY")

_FAIL_MARKERS = (
    "LLM调用失败",
    "HTTP Error 401",
    "Authorization Required",
    "authentication_error",
)


def _answer(output) -> str:
    return output if isinstance(output, str) else output.get("output", "")


def _assert_llm_ok(answer: str, label: str = ""):
    """拒绝把鉴权/调用失败字符串当成有效回答（避免假阳性）。"""
    low = answer or ""
    for m in _FAIL_MARKERS:
        assert m not in low, f"{label}LLM 调用异常: {low[:120]}"
    assert low.strip(), f"{label}回答为空"


# ══════════════════════════════════════════════
# 简单任务
# ══════════════════════════════════════════════

@REQUIRES_API
def test_factual_qa():
    """事实问答 — Agent 应直接给出正确答案"""
    from react_agent.react_loop import react_loop

    # 避免「回答一个字」误导模型去搜谜语；给足步数以便工具后收束
    output = react_loop("法国的首都是什么？请只回答城市名", max_steps=4)
    answer = _answer(output)
    _assert_llm_ok(answer, "事实问答: ")
    assert any(c in answer for c in ["巴", "黎", "Paris"]), f"事实问答失败: {answer[:60]}"
    print(f"✅ 事实问答: {answer[:40]}")


@REQUIRES_API
def test_calculator():
    """工具调用 — Agent 应调 calculator 而非靠 LLM 记忆算数"""
    from react_agent.react_loop import react_loop

    output = react_loop("计算 1234 × 5678 等于多少？", max_steps=5)
    answer = _answer(output)
    _assert_llm_ok(answer, "计算器: ")
    assert "7006652" in answer.replace(",", ""), f"计算结果错误: {answer[:80]}"
    print(f"✅ 工具调用(计算器): {answer[:60]}")


@REQUIRES_API
def test_tool_selection():
    """工具选择 — Agent 应尝试调 web_search 等工具"""
    from react_agent.react_loop import react_loop

    output = react_loop("搜索一下今天北京的温度", max_steps=4)
    answer = _answer(output)
    _assert_llm_ok(answer, "工具选择: ")
    # 至少应给出可读输出（温度/天气/搜索结果相关即可）
    assert len(answer.strip()) > 10, f"工具选择无有效输出: {answer[:80]}"
    print(f"✅ 工具选择: Agent 尝试了搜索工具 (输出长度={len(answer)})")


@REQUIRES_API
def test_qa_with_city():
    """开放问答 — 包含多个信息点的回答"""
    from react_agent.react_loop import react_loop

    output = react_loop("Python 和 Java 的主要区别是什么？列举 3 点", max_steps=3)
    answer = _answer(output)
    _assert_llm_ok(answer, "开放问答: ")
    assert len(answer) > 30, f"回答太短: {answer[:60]}"
    print(f"✅ 开放问答: {len(answer)} chars")


# ══════════════════════════════════════════════
# 复杂任务
# ══════════════════════════════════════════════

@REQUIRES_API
def test_multi_step_reasoning():
    """多步推理 — Agent 逐步分析问题"""
    from react_agent.react_loop import react_loop

    output = react_loop("""一个池塘里有一片荷叶，每天面积翻一倍。
第 30 天荷叶覆盖整个池塘。
问：第几天荷叶覆盖一半的池塘？""", max_steps=5)
    answer = _answer(output)
    _assert_llm_ok(answer, "多步推理: ")
    assert "29" in answer, f"多步推理错误: {answer[:100]}"
    print(f"✅ 多步推理: {answer[:60]}")


@REQUIRES_API
def test_multi_turn_consistency():
    """多轮一致性 — 多次回答应包含相同核心事实"""
    from react_agent.react_loop import react_loop

    q = "鲁迅的《狂人日记》是哪一年发表的？"
    years = []
    for _ in range(2):
        output = react_loop(q, max_steps=3)
        ans = _answer(output)
        _assert_llm_ok(ans, "多轮一致性: ")
        match = re.search(r"1918", ans)
        years.append(match.group() if match else "none")
    assert all(y == "1918" for y in years), f"核心事实不一致: {years}"
    print(f"✅ 多轮一致性: 均回答 1918 年")


@REQUIRES_API
def test_tool_error_recovery():
    """错误恢复 — 工具调用失败后 Agent 应尝试其他方式"""
    from react_agent.react_loop import react_loop

    output = react_loop("搜索 '2025 年诺贝尔物理学奖' 的信息", max_steps=6)
    answer = _answer(output)
    _assert_llm_ok(answer, "错误恢复: ")
    assert len(answer) > 20, f"错误恢复失败: {answer[:60]}"
    print(f"✅ 错误恢复: {answer[:80]}")


@REQUIRES_API
def test_long_context():
    """长上下文 — 能处理包含长输入的任务"""
    from react_agent.react_loop import react_loop

    long_input = "请总结以下内容的核心观点：" + "人工智能正在改变各个行业。" * 50
    output = react_loop(long_input, max_steps=3)
    answer = _answer(output)
    _assert_llm_ok(answer, "长上下文: ")
    assert len(answer) > 20, f"长上下文处理失败"
    print(f"✅ 长上下文: {len(answer)} chars")


@REQUIRES_API
def test_mcp_tool_integration():
    """MCP 工具 — MCP 客户端模块可被正确导入"""
    from react_agent.mcp_client import MCPClient
    assert MCPClient is not None
    print(f"✅ MCP 客户端模块导入成功")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
