"""
LangGraph 多 Agent 编排 — 纯 LangChain 生态版本

supervisor 分解任务 → worker 并行执行 → join 合并结果
"""

from typing import TypedDict, List, Annotated
import operator
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                  # graph/

from langgraph.graph import StateGraph, END, MessagesState
from langchain_core.messages import SystemMessage, HumanMessage

from llm import get_llm
from agent import build_agent


class OrchestratorState(TypedDict):
    query: str
    tasks: List[str]
    results: Annotated[List[str], operator.add]
    final_answer: str


def supervisor(state: OrchestratorState):
    """用 LLM 分解任务"""
    llm = get_llm()
    prompt = f"""将以下用户请求分解为多个可独立执行的子任务。
每个子任务一行，用数字编号。
不要添加额外说明。

请求: {state['query']}"""
    response = llm.invoke([HumanMessage(content=prompt)])
    tasks = [t.strip() for t in response.content.strip().split("\n") if t.strip()]
    # 去掉编号前缀
    clean = []
    for t in tasks:
        for prefix in ["1.", "2.", "3.", "4.", "5.", "- ", "* "]:
            if t.startswith(prefix):
                t = t[len(prefix):]
                break
        clean.append(t.strip())
    return {"tasks": clean}


def worker_node(state: OrchestratorState):
    """执行所有子任务"""
    results = []
    worker_app = build_agent()
    for task in state.get("tasks", []):
        result = worker_app.invoke({
            "messages": [
                SystemMessage(content="你是一个专注于子任务的 AI 助手。完成后输出 FINAL ANSWER。"),
                HumanMessage(content=task),
            ]
        }, {"recursion_limit": 12})
        answer = ""
        for m in reversed(result["messages"]):
            if hasattr(m, "content") and m.content:
                answer = m.content.strip()
                break
        results.append(f"[任务] {task}\n{answer}")
    return {"results": results}


def join(state: OrchestratorState):
    results = state.get("results", [])
    if not results:
        return {"final_answer": "没有可汇总的结果"}
    if len(results) == 1:
        return {"final_answer": results[0]}
    parts = [f"-- 结果{i} --\n{r}" for i, r in enumerate(results, 1)]
    return {"final_answer": "\n\n".join(parts)}


def build_orchestrator():
    builder = StateGraph(OrchestratorState)
    builder.add_node("supervisor", supervisor)
    builder.add_node("worker", worker_node)
    builder.add_node("join", join)
    builder.set_entry_point("supervisor")
    builder.add_conditional_edges(
        "supervisor",
        lambda s: "worker" if s.get("tasks") else "join",
    )
    builder.add_edge("worker", "join")
    builder.set_finish_point("join")
    return builder.compile()


def run(query: str) -> str:
    app = build_orchestrator()
    result = app.invoke({"query": query, "tasks": [], "results": [], "final_answer": ""})
    return result.get("final_answer", "")
