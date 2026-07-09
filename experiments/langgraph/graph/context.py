"""
上下文管理 — 消息长度检查 + 多策略截断

策略：
  truncate（默认）: 从最早的非 system 消息开始删，保留最近 3 条
  drop:           只删除已执行完毕的 tool_call + tool_result 对
  summarize:      将最早的多轮对话压缩成一段摘要（需一次 LLM 调用）
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                  # graph/

from langchain_core.messages import SystemMessage, HumanMessage
from llm import get_llm

MAX_TOKENS = 32000
KEEP_RECENT = 3


def _estimate_tokens(msg) -> int:
    """估算一条消息的 token 数（中文字符 / 1.5 + 英文 / 4）"""
    text = msg.content if hasattr(msg, "content") else str(msg.get("content", ""))
    if not text:
        return 0
    chinese = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other = len(text) - chinese
    return int(chinese / 1.5 + other / 4) + 1


def _total_tokens(messages) -> int:
    return sum(_estimate_tokens(m) for m in messages)


def _truncate(msgs):
    """策略 A: 从最早的非 system 消息开始删"""
    deleted = 0
    while len(msgs) > KEEP_RECENT + 1:
        removed = False
        for i in range(1, len(msgs) - KEEP_RECENT):
            role = msgs[i].type if hasattr(msgs[i], "type") else msgs[i].get("role", "")
            if role != "system":
                msgs.pop(i)
                deleted += 1
                removed = True
                break
        if not removed:
            break
        if _total_tokens(msgs) <= MAX_TOKENS and len(msgs) <= 2000:
            break
    return msgs, deleted


def _drop_tool_calls(msgs):
    """策略 B: 只删除已执行完毕的 tool_call + tool_result 对"""
    removed_pairs = 0
    i = 0
    while i < len(msgs) - 2:
        msg = msgs[i]
        if (hasattr(msg, "type") and msg.type == "ai"
                and hasattr(msg, "tool_calls") and msg.tool_calls):
            tool_call_ids = {tc["id"] for tc in msg.tool_calls}
            j = i + 1
            found = False
            while j < len(msgs):
                m2 = msgs[j]
                if (hasattr(m2, "type") and m2.type == "tool"
                        and hasattr(m2, "tool_call_id")
                        and m2.tool_call_id in tool_call_ids):
                    j += 1
                    found = True
                else:
                    break
            if found:
                del msgs[i:j]
                removed_pairs += 1
                continue
        i += 1
        if _total_tokens(msgs) <= MAX_TOKENS and len(msgs) <= 2000:
            break
    return msgs, removed_pairs


def _summarize(msgs):
    """策略 C: 用 LLM 将最早的多轮对话压缩成一段摘要"""
    system_idx = 0
    for i, m in enumerate(msgs):
        role = m.type if hasattr(m, "type") else m.get("role", "")
        if role == "system":
            system_idx = i
            break
    if len(msgs) <= system_idx + 5:
        return _truncate(msgs)

    summarize_end = len(msgs) - KEEP_RECENT
    if summarize_end <= system_idx + 1:
        return _truncate(msgs)

    to_summarize = []
    for i in range(system_idx + 1, summarize_end):
        m = msgs[i]
        role = m.type if hasattr(m, "type") else m.get("role", "")
        content = m.content if hasattr(m, "content") else str(m.get("content", ""))
        if role == "user":
            to_summarize.append(f"用户: {content}")
        elif role == "assistant":
            to_summarize.append(f"助手: {content[:200]}")
        elif role == "tool":
            to_summarize.append(f"工具返回: {str(content)[:100]}")

    if not to_summarize:
        return _truncate(msgs)

    history = "\n".join(to_summarize)
    prompt = f"请将以下对话历史压缩成一段简短摘要，保留所有关键事实和用户意图。不要遗漏信息。\n\n对话历史:\n{history}\n\n摘要:"
    try:
        summary_llm = get_llm()
        reply = summary_llm.invoke([HumanMessage(content=prompt)]).content or ""
        if reply.strip():
            summary_msg = SystemMessage(content=f"[对话摘要] {reply.strip()}")
            msgs = [msgs[system_idx]] + [summary_msg] + msgs[summarize_end:]
            return msgs, 1
    except Exception:
        pass
    return _truncate(msgs)


def manage(messages):
    """
    统一入口：检查消息长度，超限时自动选择策略处理。

    参数:
        messages: LangChain Message 对象列表

    返回:
        (处理后的消息列表, 操作描述字符串)
    """
    total = _total_tokens(messages)
    if total <= MAX_TOKENS and len(messages) <= 2000:
        return messages, ""

    # 统计 tool_call 对数，决定策略
    tool_pair_count = 0
    for m in messages:
        if (hasattr(m, "type") and m.type == "ai"
                and hasattr(m, "tool_calls") and m.tool_calls):
            tool_pair_count += len(m.tool_calls)

    if tool_pair_count >= 5:
        result, deleted = _drop_tool_calls(list(messages))
        action = f"丢弃了 {deleted} 对 tool 调用"
    elif total > MAX_TOKENS * 1.5 and len(messages) > 20:
        result, deleted = _summarize(list(messages))
        action = f"摘要压缩（{len(result)} 条消息）"
    else:
        result, deleted = _truncate(list(messages))
        action = f"截断了 {deleted} 条消息"

    return result, action
