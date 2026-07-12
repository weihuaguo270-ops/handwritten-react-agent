"""
上下文窗口管理（Context Engineering）

防止 ReAct Loop 因对话历史过长而超出 LLM 的上下文限制。
在每步之后自动检查 token 用量，超限时按策略处理。

三种策略:
  - truncate:  从最早的非 system 消息开始删（默认，0 额外 LLM 调用）
  - drop:      删除已使用完的 tool_call/tool_result 对（0 额外 LLM 调用）
  - summarize: 将最早的多轮对话压缩成一段摘要（需 1 次 LLM 调用）

用法:
    from react_agent.context import CONTEXT
    # 在每步之后
    messages = CONTEXT.manage(messages, llm_call=my_llm)
"""

from enum import Enum
from typing import Optional, Callable
import re


# ============================================================
# 1. 策略枚举
# ============================================================
class ContextStrategy(Enum):
    TRUNCATE = "truncate"       # 删最早的非 system 消息
    DROP = "drop"               # 删已用完的 tool 调用对
    SUMMARIZE = "summarize"     # 压缩早期对话为摘要
    AUTO = "auto"               # LLM 根据对话内容自行决定


# ============================================================
# 2. Token 估算（无需 tokenizer 依赖）
# ============================================================

def estimate_tokens(text: str) -> int:
    """估算一段文本的 token 数

    中英文混合的粗略估算：
    - 英文/数字/符号: ~1 token / 4 chars
    - 中文/日文/韩文: ~1 token / 1.5 chars
    - 综合折中: len / 2
    """
    if not text:
        return 0

    # 统计中文字符数
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]', text))
    other_chars = len(text) - chinese_chars

    return int(chinese_chars / 1.5 + other_chars / 4) + 1


def estimate_messages_tokens(messages: list) -> int:
    """估算整个 messages 列表的 token 数"""
    total = 0
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""
        total += estimate_tokens(role) + estimate_tokens(content)

        # tool_calls 也有 token 开销
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                fn = tc.get("function", {})
                total += estimate_tokens(fn.get("name", ""))
                total += estimate_tokens(fn.get("arguments", ""))
    return total


# ============================================================
# 3. ContextManager 核心类
# ============================================================

class ContextManager:
    """上下文窗口管理器

    用法:
        ctx = ContextManager(max_tokens=4096)
        messages = ctx.manage(messages)
    """

    def __init__(self, max_tokens: int = 4096, strategy: ContextStrategy = ContextStrategy.TRUNCATE,
                 reserve_tokens: int = 1024, warn_ratio: float = 0.85):
        self.max_tokens = max_tokens          # 最大 token 数
        self.strategy = strategy               # 当前策略
        self.reserve_tokens = reserve_tokens   # 给后续回复预留的 token
        self.warn_ratio = warn_ratio           # 触发警告的比例（0.85 = 85%）
        self.last_action = ""                  # 上次执行的操作描述

    def set_strategy(self, strategy_name: str) -> str:
        """运行时切换策略"""
        try:
            self.strategy = ContextStrategy(strategy_name)
            return f"上下文策略已切换为 {strategy_name}"
        except ValueError:
            return f"未知策略: {strategy_name}，可选: truncate, drop, summarize"

    def check(self, messages: list) -> dict:
        """检查当前 token 用量

        返回:
            {"total": int, "limit": int, "usage_ratio": float, "is_ok": bool}
        """
        total = estimate_messages_tokens(messages)
        limit = self.max_tokens - self.reserve_tokens
        ratio = total / self.max_tokens
        return {
            "total": total,
            "limit": limit,
            "usage_ratio": round(ratio, 3),
            "is_ok": total <= limit,
        }

    def manage(self, messages: list,
               llm_call: Optional[Callable] = None) -> list:
        """统一入口：检查并管理上下文窗口

        在每步之后调用。如果未超限则原样返回。
        如果超限则按策略处理。

        参数:
            messages: 当前对话消息列表
            llm_call: 用于 summarize 策略的 LLM 调用函数

        返回:
            处理后的 messages 列表
        """
        check = self.check(messages)
        self.last_action = ""

        if check["is_ok"]:
            if check["usage_ratio"] >= self.warn_ratio:
                self.last_action = f"⚠️ 上下文使用 {check['usage_ratio']:.0%}，接近限制"
            return messages

        # 超限，按策略处理（用循环代替递归防止栈溢出）
        max_iterations = 10
        for _ in range(max_iterations):
            if self.check(messages)["is_ok"]:
                break

            if self.strategy == ContextStrategy.TRUNCATE:
                messages = self._truncate(messages)
            elif self.strategy == ContextStrategy.DROP:
                messages = self._drop_tool_calls(messages)
            elif self.strategy == ContextStrategy.SUMMARIZE:
                if llm_call is not None:
                    messages = self._summarize(messages, llm_call)
                else:
                    self.last_action = "summarize 需要 llm_call，回退到 truncate"
                    self.strategy = ContextStrategy.TRUNCATE
                    messages = self._truncate(messages)
            elif self.strategy == ContextStrategy.AUTO:
                if llm_call is not None:
                    messages = self._auto_choose(messages, llm_call)
                else:
                    self.last_action = "auto 需要 llm_call，回退到 truncate"
                    self.strategy = ContextStrategy.TRUNCATE
                    messages = self._truncate(messages)

        return messages

    # ----------------------------------------------------------
    # 策略 A: 截断
    # ----------------------------------------------------------
    def _truncate(self, messages: list) -> list:
        """从最早的非 system 消息开始删，直到 token 用量降到限制内"""
        deleted = 0
        # 保护 system prompt（第一条）和最新的几条消息
        keep_recent = 3  # 至少保留最近 3 条消息

        while len(messages) > keep_recent + 1:  # +1 是 system
            # 找到第一条非 system 消息
            for i in range(1, len(messages) - keep_recent):
                if messages[i]["role"] != "system":
                    removed = messages.pop(i)
                    deleted += 1
                    break
            else:
                break  # 没有可删的了

            # 检查是否已降到限制内
            new_check = self.check(messages)
            if new_check["is_ok"]:
                break

        self.last_action = f"截断了 {deleted} 条消息（当前 {estimate_messages_tokens(messages)} tokens 以内）"
        return messages

    # ----------------------------------------------------------
    # 策略 B: 丢弃 tool 调用对
    # ----------------------------------------------------------
    def _drop_tool_calls(self, messages: list) -> list:
        """删除已经完成（有 tool_result 回复）的 tool_call + tool_result 对

        tool_call 消息的格式:
            {"role": "assistant", "tool_calls": [...]}
        tool_result 消息的格式:
            {"role": "tool", "tool_call_id": "..."}
        """
        removed_pairs = 0

        i = 0
        while i < len(messages) - 2:  # 至少留 2 条
            msg = messages[i]

            # 找到 assistant 发的 tool_calls
            if msg["role"] == "assistant" and msg.get("tool_calls"):
                tool_call_ids = {tc["id"] for tc in msg["tool_calls"]}

                # 找到后续对应的 tool_result 消息
                j = i + 1
                found_results = False
                while j < len(messages):
                    if (messages[j]["role"] == "tool"
                            and messages[j].get("tool_call_id") in tool_call_ids):
                        found_results = True
                        j += 1
                    else:
                        break

                if found_results:
                    # 删除从 i 到 j-1 的所有消息（tool_call + 它的 results）
                    del messages[i:j]
                    removed_pairs += 1
                    continue  # 不推进 i，因为删掉了当前位置

            i += 1

            # 检查是否已降到限制内
            if self.check(messages)["is_ok"]:
                break

        self.last_action = f"丢弃了 {removed_pairs} 对 tool 调用（当前 {estimate_messages_tokens(messages)} tokens 以内）"
        return messages

    # ----------------------------------------------------------
    # 策略 C: 摘要
    # ----------------------------------------------------------
    def _summarize(self, messages: list, llm_call: Callable) -> list:
        """把最早的多轮对话压缩成一段摘要

        保留 system prompt 和最近 2 轮对话，把之前的全部压缩。
        """
        # 找到 system prompt
        system_idx = 0
        for i, msg in enumerate(messages):
            if msg["role"] == "system":
                system_idx = i
                break

        # 如果只有 system + 最近 2 轮，不值得摘要
        if len(messages) <= system_idx + 5:
            self.last_action = "对话太短，不适合摘要"
            return self._truncate(messages, check)

        # 要压缩的部分：system 之后到最近 2 轮之前
        summarize_end = len(messages) - 4  # 保留最近 2 轮（user + assistant）
        if summarize_end <= system_idx + 1:
            self.last_action = "对话太短，不适合摘要"
            return self._truncate(messages, check)

        # 提取要压缩的文本
        to_summarize = []
        for i in range(system_idx + 1, summarize_end):
            msg = messages[i]
            role = msg["role"]
            content = msg.get("content", "") or ""
            if role == "user":
                to_summarize.append(f"用户: {content}")
            elif role == "assistant":
                to_summarize.append(f"助手: {content}")
            elif role == "tool":
                to_summarize.append(f"工具返回: {str(content)[:100]}")

        history_text = "\n".join(to_summarize)
        if not history_text.strip():
            self.last_action = "无可摘要的内容"
            return self._truncate(messages, check)

        # 调 LLM 生成摘要
        prompt = f"""请将以下对话历史压缩成一段简短摘要，保留所有关键事实和用户意图。
不要遗漏信息，但尽量简洁。

对话历史:
{history_text}

摘要:"""
        try:
            reply = llm_call(prompt)
            summary = reply.strip() if reply else ""
            if not summary:
                self.last_action = "摘要生成失败，回退到截断"
                return self._truncate(messages, check)

            # 用一条摘要消息替换被压缩的部分
            summary_msg = {"role": "system", "content": f"[对话摘要] {summary}"}
            messages = [messages[system_idx]] + [summary_msg] + messages[summarize_end:]

            new_total = estimate_messages_tokens(messages)
            self.last_action = f"摘要压缩（{new_total} tokens）"
            return messages

        except Exception as e:
            self.last_action = f"摘要失败: {e}，回退到截断"
            return self._truncate(messages)


    # ============================================================
    # 策略 D: Auto — LLM 自行决定
    # ============================================================

    _AUTO_PROMPT = """当前对话上下文已接近 token 上限。

使用情况: {usage}/{limit} tokens（{ratio}%）
消息数: {msg_count} 条
系统消息: 1 条
工具调用: {tool_count} 对

请从以下三种策略中选择一种来处理：

1. truncate — 从最早的非 system 消息开始删，保留最近几轮
   - 优点：0 额外 LLM 调用，速度快
   - 缺点：可能丢失早期关键信息

2. drop — 只删除已经执行完毕的 tool_call + tool_result 对
   - 优点：不影响对话主体
   - 缺点：只对工具调用频繁的场景有效

3. summarize — 把早期对话发给 LLM 压缩成一段摘要
   - 优点：保留关键信息
   - 缺点：多一次 LLM 调用，有少量信息损失

请简要说明你的选择理由，然后在一行内输出 "选择: X"（X 为 truncate/drop/summarize）。"""

    def _auto_choose(self, messages: list, llm_call: Callable) -> list:
        """让 LLM 根据当前对话内容自行决定最优策略"""
        check = self.check(messages)
        total = check["total"]
        limit = max(check["limit"], 1)

        # 统计工具调用对数
        tool_count = 0
        for i, msg in enumerate(messages):
            if msg["role"] == "assistant" and msg.get("tool_calls"):
                for j in range(i + 1, min(i + 10, len(messages))):
                    if messages[j]["role"] == "tool":
                        tool_count += 1
                        break

        prompt = self._AUTO_PROMPT.format(
            usage=total,
            limit=limit,
            ratio=round(total / limit * 100),
            msg_count=len(messages),
            tool_count=tool_count,
        )

        try:
            reply = llm_call(prompt).strip()
            reply_lower = reply.lower()
            chosen = "truncate"

            # 解析 LLM 的选择
            for pattern in ["选择: truncate", "选择: drop", "选择: summarize"]:
                if pattern in reply_lower:
                    chosen = pattern.split("选择: ")[1]
                    break
            else:
                # 兜底：找第一个匹配的策略名
                for kw in ["truncate", "drop", "summarize"]:
                    if kw in reply_lower:
                        chosen = kw
                        break

            self.last_action = f"LLM 选择了 {chosen} 策略"
            if chosen == "drop":
                return self._drop_tool_calls(messages)
            elif chosen == "summarize":
                return self._summarize(messages, llm_call)
            return self._truncate(messages)

        except Exception as e:
            self.last_action = f"auto 选择失败 ({e})，回退到 truncate"
            return self._truncate(messages)


# ============================================================
# 4. 全局实例
CONTEXT = ContextManager(max_tokens=48000, reserve_tokens=2000, strategy=ContextStrategy.AUTO)


# ============================================================
# 5. 工具定义 + 工具函数
# ============================================================

CONTEXT_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "switch_context_strategy",
        "description": "切换上下文窗口管理策略: auto(LLM自动选择)/truncate(截断最早消息)/drop(丢弃已用完的tool调用)/summarize(压缩为摘要)",
        "parameters": {
            "type": "object",
            "properties": {
                "strategy": {
                    "type": "string",
                    "enum": ["truncate", "drop", "summarize"],
                    "description": "策略名称"
                }
            },
            "required": ["strategy"],
        },
    },
}


def tool_switch_context_strategy(strategy: str) -> str:
    """运行时切换上下文管理策略"""
    return CONTEXT.set_strategy(strategy)
