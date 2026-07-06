"""
LangGraph 单 Agent — 纯 LangChain 生态版本

状态类型：AgentState（自定义 TypedDict）
  - messages: 对话历史（自动追加）
  - search_count: 搜索次数
  - user_query: 原始用户问题（供记忆提取节点使用）

节点：
  call_model → (条件边) → tools → call_model（循环）
      │
      └── 无 tool_calls → extract_memory → END
"""

import json
import sys
import os
from typing import Literal, TypedDict, Annotated, List
import operator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                  # graph/

from langgraph.graph import StateGraph, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage

from llm import get_llm
from tools import get_tools
from prompts import inject_cot
from memory import MEMORY


# ============================================================
# 状态定义
# ============================================================

class AgentState(TypedDict):
    """LangGraph 单 Agent 状态"""
    messages: Annotated[List, operator.add]   # 对话历史（自动追加）
    search_count: int                          # 搜索次数
    user_query: str                            # 原始用户问题，供 extract_memory 节点使用


# ============================================================
# LangGraph 构建
# ============================================================

def build_agent():

    tools = get_tools()
    llm = get_llm().bind_tools(tools)
    tool_map = {t.name: t for t in tools}
    memory_llm = get_llm()  # 记忆提取用的 LLM（不绑定工具）

    def call_model(state: AgentState):
        """
        调 LLM 节点。

        读取当前对话历史（state["messages"]），调用绑定了工具的 ChatOpenAI，
        将 LLM 回复追加到 messages 中。

        参数:
            state: 当前 Agent 状态

        返回:
            {"messages": [AIMessage]} — messages 通过 operator.add reducer 自动追加
        """
        llm_response = llm.invoke(state["messages"])
        return {"messages": [llm_response]}

    def tools_node(state: AgentState):
        """
        执行工具调用节点。

        解析上一步 AIMessage 中的 tool_calls，查找对应的 @tool 函数并执行。
        支持搜索次数限制（search_count ≥ 4 时跳过 web_search）。

        参数:
            state: 当前 Agent 状态（从 messages[-1] 读取 tool_calls）

        返回:
            {"messages": [ToolMessage, ...], "search_count": int}
        """
        last_msg = state["messages"][-1]
        results = []
        search_count = state.get("search_count", 0)

        for tool_call in last_msg.tool_calls:
            tool_name, tool_args, tool_call_id = tool_call["name"], tool_call.get("args", {}), tool_call["id"]

            if tool_name in ("web_search",) and search_count >= 4:
                content = "搜索已达上限，请基于已有信息回答"
            else:
                if tool_name in tool_map:
                    try:
                        content = str(tool_map[tool_name].invoke(tool_args))
                        if tool_name in ("web_search",):
                            search_count += 1
                    except Exception as e:
                        content = json.dumps({"error": f"执行错误: {e}"})
                elif tool_name == "rag_query":
                    from rag import rag_query
                    try:
                        content = rag_query.invoke(tool_args)
                    except Exception as e:
                        content = json.dumps({"error": f"RAG错误: {e}"})
                else:
                    content = json.dumps({"error": f"未知工具: {tool_name}"})

            results.append(ToolMessage(content=content, tool_call_id=tool_call_id))

        return {"messages": results, "search_count": search_count}

    def should_continue(state: AgentState) -> Literal["tools", "extract_memory"]:
        """
        条件边：判断下一步执行方向。

        - 有 tool_calls → 返回 "tools"，继续工具循环
        - 无 tool_calls → 返回 "extract_memory"，执行记忆提取后结束

        参数:
            state: 当前 Agent 状态

        返回:
            "tools" 或 "extract_memory"
        """
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return "extract_memory"

    def extract_memory_node(state: AgentState):
        """
        记忆提取节点。

        Agent 循环结束后，用 LLM 从对话中提取事实性信息，
        通过 add_or_update 存入记忆系统（语义去重 + 冲突替换）。

        作为 LangGraph 节点运行，不依赖外部函数。全部逻辑在节点内完成。

        参数:
            state: 最终 Agent 状态（含完整对话历史和 user_query）

        返回:
            {} — 不修改状态，只产生副作用（写入 memory.json）
        """
        user_query = state.get("user_query", "")
        answer = ""
        for m in reversed(state["messages"]):
            if hasattr(m, "content") and m.content:
                answer = m.content.strip()
                break

        if len(answer) < 20 or any(w in user_query for w in ["忘记", "删除"]):
            return {}

        prompt = (
            "从以下对话中提取**具体的事实性信息**。\n"
            "规则：\n"
            "- 提取具体信息，如姓名、职业、爱好、背景、联系方式等\n"
            "- 每个事实单独一行\n"
            "- 忽略闲聊、问候、临时问题\n"
            "- 如果用户明确告知个人信息，务必提取\n"
            "- 没有任何具体事实就输出空行\n\n"
            f"用户: {user_query}\n\n"
            f"助手: {answer}\n\n"
            "事实:"
        )
        try:
            raw = memory_llm.invoke([HumanMessage(content=prompt)]).content or ""
        except Exception:
            return {}

        if not raw.strip():
            return {}

        facts = [f.strip() for f in raw.split("\n") if f.strip()]
        saved = 0
        for fact in facts:
            if len(fact) > 5 and not any(
                w in fact for w in ["LLM失败", "错误", "抱歉", "没有提供", "没有任何", "个人信息"]
            ):
                action, detail = MEMORY.add_or_update(fact)
                if action == "added":
                    saved += 1
                elif action == "updated":
                    print(f"[记忆] 更新: \"{detail}\" → \"{fact}\"")
                    saved += 1
        if saved > 0:
            print(f"[记忆] 自动记忆: 保存了 {saved} 条新信息")
        return {}

    builder = StateGraph(AgentState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", tools_node)
    builder.add_node("extract_memory", extract_memory_node)

    builder.set_entry_point("call_model")
    builder.add_conditional_edges(
        "call_model", should_continue,
        {"tools": "tools", "extract_memory": "extract_memory"},
    )
    builder.add_edge("tools", "call_model")
    builder.add_edge("extract_memory", END)

    return builder.compile(checkpointer=MemorySaver())


# ============================================================
# 入口函数
# ============================================================

def run(query: str, max_steps: int = 10, thread_id: str = "default") -> str:
    """
    运行单 Agent。

    1. 构建 LangGraph 应用
    2. 注入 system prompt（含 CoT 推理引导）
    3. 执行图：call_model → tools（循环）→ extract_memory → END
    4. 从结果中提取最终答案

    参数:
        query: 用户问题
        max_steps: 最大迭代步数（默认 10）
        thread_id: LangGraph checkpoint 线程 ID

    返回:
        最终答案字符串
    """
    app = build_agent()
    system_prompt = inject_cot(query)

    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": max_steps + 3}
    result = app.invoke({
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query),
        ],
        "search_count": 0,
        "user_query": query,
    }, config)

    return _extract_answer(result)


def _extract_answer(result: dict) -> str:
    """
    从 LangGraph 执行结果中提取最终答案。

    倒序遍历消息列表：
    1. 优先匹配 FINAL ANSWER: 标记后面的文本
    2. 无标记则返回最后一条非空消息的 content

    参数:
        result: LangGraph invoke 返回的状态字典（含 messages 列表）

    返回:
        提取的答案字符串，未找到时返回空字符串
    """
    import re
    for m in reversed(result["messages"]):
        if hasattr(m, "content") and m.content:
            fa = re.search(r'FINAL ANSWER:\s*(.*)', m.content, re.IGNORECASE | re.DOTALL)
            if fa:
                return fa.group(1).strip()
            return m.content.strip()
    return ""
