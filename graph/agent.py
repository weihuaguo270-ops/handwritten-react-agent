"""
LangGraph 单 Agent — 纯 LangChain 生态版本

状态类型：AgentState（自定义 TypedDict）
  - messages: 对话历史（自动追加）
  - search_count: 搜索次数
  - user_query: 原始用户问题

节点：
  call_model → (条件边) → tools → call_model（循环）
      │
      └── 无 tool_calls → context_manage → extract_memory → END

Harness 集成：
  通过 config["configurable"]["harness"] 传入 Harness 实例。
  call_model 节点自动记录 thought。
  tools_node 节点自动记录每个 action 的 name/args/observation/duration。
"""

import json
import sys
import os
from typing import Literal, TypedDict, Annotated, List, Any, Optional
import operator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                  # graph/

from langgraph.graph import StateGraph, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage

from llm import get_llm
from tools import get_tools
from prompts import build_system_prompt
from memory import MEMORY
from context import manage as manage_context

# Harness 集成
from harness import Harness


# ============================================================
# 状态定义
# ============================================================

class AgentState(TypedDict):
    """LangGraph 单 Agent 状态"""
    messages: Annotated[List, operator.add]   # 对话历史（自动追加）
    search_count: int                          # 搜索次数
    user_query: str                            # 原始用户问题


# ============================================================
# LangGraph 构建
# ============================================================

def build_agent(mcp_clients=None, harness: "Harness | None" = None):
    """构建 LangGraph Agent

    参数:
        mcp_clients: MCP 客户端列表（可选）
        harness: Harness 实例（可选）。传入后自动记录轨迹。
    """

    tools = get_tools()
    local_names = {t.name for t in tools}
    # 合并 MCP 工具（跳过与本地区名的工具）
    if mcp_clients:
        for client in mcp_clients:
            for name, fn in client.to_langchain_tools():
                if name in local_names:
                    print(f"  [MCP] 跳过 {name}（已被本地工具覆盖）")
                    continue
                from langchain_core.tools import tool
                wrapped = tool(fn)
                tools.append(wrapped)
    llm = get_llm().bind_tools(tools)
    tool_map = {t.name: t for t in tools}
    memory_llm = get_llm()  # 记忆提取用的 LLM（不绑定工具）
    step_counter = [0]  # 闭包可变引用——用列表绕过 int 不可变

    def call_model(state: AgentState, config: dict | None = None):
        """
        调 LLM 节点。

        读取当前对话历史（state["messages"]），调用绑定了工具的 ChatOpenAI，
        将 LLM 回复追加到 messages 中。

        参数:
            state: 当前 Agent 状态
            config: LangGraph 运行时配置，可选。含 {"configurable": {"harness": Harness}}
        """
        step_counter[0] += 1
        current_step = step_counter[0]

        llm_response = llm.invoke(state["messages"])

        # ── Harness 记录 thought ──
        h = _get_harness(config)
        if h:
            thought_content = llm_response.content or ""
            tokens = 0
            if hasattr(llm_response, "usage_metadata") and llm_response.usage_metadata:
                tokens = llm_response.usage_metadata.get("total_tokens", 0)
            h.record_thought(step=current_step, thought=thought_content, tokens=tokens)

        return {"messages": [llm_response]}

    def tools_node(state: AgentState, config: dict | None = None):
        """
        执行工具调用节点。

        解析上一步 AIMessage 中的 tool_calls，查找对应的 @tool 函数并执行。
        支持搜索次数限制（search_count ≥ 4 时跳过 web_search）。

        参数:
            state: 当前 Agent 状态（从 messages[-1] 读取 tool_calls）
            config: LangGraph 运行时配置，可选。含 {"configurable": {"harness": Harness}}
        """
        last_msg = state["messages"][-1]
        results = []
        search_count = state.get("search_count", 0)
        current_step = step_counter[0]

        # ── Harness 实例 ──
        h = _get_harness(config)

        for tool_call in last_msg.tool_calls:
            tool_name, tool_args, tool_call_id = tool_call["name"], tool_call.get("args", {}), tool_call["id"]
            import time as _time
            action_start = _time.time()

            if tool_name in ("web_search",) and search_count >= 4:
                content = "搜索已达上限，请基于已有信息回答"
                action_duration = _time.time() - action_start
            else:
                if tool_name in tool_map:
                    try:
                        # ── 沙箱检查：是否在子进程执行 ──
                        if h and h.is_sandboxed(tool_name):
                            sandbox_call = {
                                "function": {
                                    "name": tool_name,
                                    "arguments": json.dumps(tool_args),
                                }
                            }
                            content = h.run_sandboxed(sandbox_call)
                        else:
                            content = str(tool_map[tool_name].invoke(tool_args))

                        action_duration = _time.time() - action_start
                        if tool_name in ("web_search",):
                            search_count += 1
                    except Exception as e:
                        content = json.dumps({"error": f"执行错误: {e}"})
                        action_duration = _time.time() - action_start
                elif tool_name == "rag_query":
                    from rag import rag_query
                    try:
                        content = rag_query.invoke(tool_args)
                        action_duration = _time.time() - action_start
                    except Exception as e:
                        content = json.dumps({"error": f"RAG错误: {e}"})
                        action_duration = _time.time() - action_start
                else:
                    content = json.dumps({"error": f"未知工具: {tool_name}"})
                    action_duration = _time.time() - action_start

            results.append(ToolMessage(content=content, tool_call_id=tool_call_id))

            # ── Harness 记录 action ──
            if h:
                h.record_action(
                    step=current_step,
                    action_name=tool_name,
                    action_args=json.dumps(tool_args, ensure_ascii=False),
                    observation=content,
                    duration_seconds=round(action_duration, 3),
                    tokens=int(len(content) / 4),  # 粗略估计 tokens
                )

        return {"messages": results, "search_count": search_count}

    def should_continue(state: AgentState) -> Literal["tools", "context_manage"]:
        """
        条件边：判断下一步执行方向。

        - 有 tool_calls → 返回 "tools"，继续工具循环
        - 无 tool_calls → 返回 "context_manage"，执行上下文检查后再到记忆提取
        """
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return "context_manage"

    def context_manage_node(state: AgentState):
        """
        上下文管理节点。

        在 Agent 循环结束时检查消息总长度。超限时按策略处理：
        truncate / drop / summarize（具体逻辑在 graph/context.py 中）。
        """
        search_count = state.get("search_count", 0)
        managed_messages, action = manage_context(state["messages"])
        if action:
            print(f"[上下文] {action}")
        return {"messages": managed_messages, "search_count": search_count}

    def extract_memory_node(state: AgentState):
        """
        记忆提取节点。

        Agent 循环结束后，用 LLM 从对话中提取事实性信息，
        通过 add_or_update 存入记忆系统（语义去重 + 冲突替换）。
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
    builder.add_node("context_manage", context_manage_node)
    builder.add_node("extract_memory", extract_memory_node)

    builder.set_entry_point("call_model")
    builder.add_conditional_edges(
        "call_model", should_continue,
        {"tools": "tools", "context_manage": "context_manage"},
    )
    builder.add_edge("tools", "call_model")
    builder.add_edge("context_manage", "extract_memory")
    builder.add_edge("extract_memory", END)

    return builder.compile(checkpointer=MemorySaver())


# ============================================================
# 辅助函数
# ============================================================

def _get_harness(config: dict | None) -> Harness | None:
    """从 LangGraph config 中提取 Harness 实例

    config 结构由 LangGraph 传入：
        {"configurable": {"harness": Harness(), ...}}
    """
    if not config:
        return None
    return config.get("configurable", {}).get("harness", None)


# ============================================================
# 入口函数
# ============================================================

def run(query: str, max_steps: int = 10, thread_id: str = "default",
        mcp_clients: list = None, harness: Harness | None = None) -> str:
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
        mcp_clients: MCP 客户端列表（可选）
        harness: Harness 实例（可选）。传入后自动记录轨迹。

    返回:
        最终答案字符串
    """
    app = build_agent(mcp_clients=mcp_clients, harness=harness)
    system_prompt = build_system_prompt(query)

    config = {
        "configurable": {
            "thread_id": thread_id,
            "harness": harness,  # 传入 Harness 供节点函数使用
        },
        "recursion_limit": max_steps + 3,
    }
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
    """
    import re
    for m in reversed(result["messages"]):
        if hasattr(m, "content") and m.content:
            fa = re.search(r'FINAL ANSWER:\s*(.*)', m.content, re.IGNORECASE | re.DOTALL)
            if fa:
                return fa.group(1).strip()
            return m.content.strip()
    return ""
