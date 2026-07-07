"""
工具定义 — 替代手写 tools/ 目录 + TOOL_REGISTRY

所有工具通过 @tool 装饰器定义，LangGraph 自动解析。
"""

import json
import os
import ast
import operator
import datetime
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                  # graph/

from langchain_core.tools import tool


@tool
def get_current_time() -> str:
    """获取当前系统时间"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculator(expression: str) -> str:
    """
    计算数学表达式。支持 + - * / 和括号。
    
    参数:
        expression: 数学表达式，如 '2 + 3 * 4'
    """
    allowed_ops = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.USub: operator.neg, ast.Pow: operator.pow,
    }

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        elif isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            return allowed_ops[type(node.op)](_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.UnaryOp):
            return allowed_ops[type(node.op)](_eval(node.operand))
        raise ValueError("不支持的表达式")

    try:
        return str(_eval(ast.parse(expression, mode="eval")))
    except Exception as e:
        return f"计算错误: {e}"


@tool
def web_search(query: str) -> str:
    """搜索互联网获取实时信息。当用户需要实时数据、新闻、或你不知道的信息时使用。
    
    参数:
        query: 搜索关键词
    """
    try:
        import urllib.request
        import urllib.parse
        url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json"
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            abstract = data.get("AbstractText", "")
            if abstract:
                return abstract
            results = data.get("RelatedTopics", [])
            if results:
                parts = []
                for r in results[:3]:
                    if isinstance(r, dict) and "Text" in r:
                        parts.append(r["Text"])
                return "\n".join(parts) if parts else f"未找到 '{query}' 相关信息"
            return f"未找到 '{query}' 相关信息"
    except Exception as e:
        return f"搜索失败: {e}"


# 工具函数清单
def get_tools():
    """返回所有可用工具的列表"""
    from rag import rag_query as _rag
    from rag import web_rag as _web_rag
    return [get_current_time, calculator, web_search, _rag, _web_rag]


# ============================================================
# 工具分类（供 Worker 隔离使用）
# ============================================================

TOOL_PROFILES = {
    "time": {"get_current_time"},
    "calc": {"calculator"},
    "web": {"web_search"},
    "summary": {"rag_query"},
    "web_rag": {"web_rag"},
}


def classify_task(task_description: str) -> set:
    """根据子任务描述判断需要的工具类型"""
    desc = task_description.lower()
    needed = set()

    if any(w in desc for w in ["时间", "时区", "当前时间", "现在几点"]):
        needed.add("time")
    if any(w in desc for w in ["计算", "数学", "等于", "+", "-", "*", "/"]):
        needed.add("calc")
    if any(w in desc for w in ["搜索", "网页", "新闻", "查询", "查找"]):
        needed.add("web")
    if any(w in desc for w in ["总结", "摘要", "概括", "归纳"]):
        needed.add("summary")

    return needed if needed else {"web", "calc"}  # 默认


def filter_tools(task_description: str) -> list:
    """根据子任务描述，返回该任务允许使用的工具子集"""
    tags = classify_task(task_description)
    allowed_names = set()
    for tag in tags:
        allowed_names |= TOOL_PROFILES.get(tag, set())

    all_tools = get_tools()
    return [t for t in all_tools if t.name in allowed_names]
