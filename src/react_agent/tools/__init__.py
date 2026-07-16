"""
tools/ — 工具模块统一入口

默认只注册 Core 工具。实验工具（RAG / ToT / Dashboard）需：
  REACT_AGENT_EXPERIMENTAL_TOOLS=1
"""

from __future__ import annotations

import os

from .get_time import get_time as _tool_get_time
from .get_time import TOOL_DEFINITION as _DEF_GET_TIME
from .calculator import calculator as _tool_calculator
from .calculator import TOOL_DEFINITION as _DEF_CALCULATOR
from .web_search import web_search as _tool_web_search
from .web_search import TOOL_DEFINITION as _DEF_WEB_SEARCH
from .fetch_page import fetch_page as _tool_fetch_page
from .fetch_page import TOOL_DEFINITION as _DEF_FETCH_PAGE
from .summarize import summarize as _tool_summarize
from .summarize import TOOL_DEFINITION as _DEF_SUMMARIZE
from .execute_python import execute_python as _tool_execute_python
from .execute_python import TOOL_DEFINITION as _DEF_EXECUTE_PYTHON

from react_agent.cot import tool_switch_cot_strategy, COT_TOOL_DEFINITION
from react_agent.prompts import tool_switch_role, ROLE_TOOL_DEFINITION
from react_agent.context import tool_switch_context_strategy, CONTEXT_TOOL_DEFINITION
from react_agent.harness import tool_toggle_sandbox, SANDBOX_TOOL_DEFINITION
from react_agent.harness.recorder import clear_trajectories

# ===== TOOL_REGISTRY：name → 函数（Core）=====
TOOL_REGISTRY = {
    "get_time": _tool_get_time,
    "calculator": _tool_calculator,
    "web_search": _tool_web_search,
    "fetch_page": _tool_fetch_page,
    "summarize": _tool_summarize,
    "switch_cot_strategy": tool_switch_cot_strategy,
    "switch_role": tool_switch_role,
    "switch_context_strategy": tool_switch_context_strategy,
    "toggle_sandbox": tool_toggle_sandbox,
    "clear_trajectories": clear_trajectories,
    "execute_python": _tool_execute_python,
}

# ===== TOOL_DEFINITIONS：发给 LLM 的工具描述（Core）=====
TOOL_DEFINITIONS = [
    _DEF_WEB_SEARCH,
    _DEF_CALCULATOR,
    _DEF_FETCH_PAGE,
    _DEF_SUMMARIZE,
    _DEF_GET_TIME,
    COT_TOOL_DEFINITION,
    ROLE_TOOL_DEFINITION,
    CONTEXT_TOOL_DEFINITION,
    SANDBOX_TOOL_DEFINITION,
    _DEF_EXECUTE_PYTHON,
    {
        "type": "function",
        "function": {
            "name": "clear_trajectories",
            "description": "删除历史轨迹文件，用于清理 Agent 的对话记录。支持按天数保留（如只保留最近7天）或全部删除",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "保留最近几天的文件（0=全部删除，7=保留近7天）",
                    }
                },
                "required": ["days"],
            },
        },
    },
]


def _enable_experimental_tools() -> None:
    """按需挂载 RAG / ToT / Dashboard（默认关闭，避免叙事与默认面混杂）。"""
    if os.environ.get("REACT_AGENT_EXPERIMENTAL_TOOLS", "").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return
    from .dashboard import start_dashboard as _tool_start_dashboard
    from .dashboard import TOOL_DEFINITION as _DEF_DASHBOARD
    from react_agent.rag import rag_query, RAG_TOOL_DEFINITION
    from react_agent.tot import tool_tot_reasoning, TOT_TOOL_DEFINITION

    TOOL_REGISTRY["rag_query"] = rag_query
    TOOL_REGISTRY["tot_reasoning"] = tool_tot_reasoning
    TOOL_REGISTRY["start_dashboard"] = _tool_start_dashboard
    for defn in (RAG_TOOL_DEFINITION, TOT_TOOL_DEFINITION, _DEF_DASHBOARD):
        if defn not in TOOL_DEFINITIONS:
            TOOL_DEFINITIONS.append(defn)


_enable_experimental_tools()
