"""
tools/ — 工具模块统一入口

每个工具一个独立文件，遵循统一接口：
  - 一个函数实现工具逻辑
  - 一个 TOOL_DEFINITION 字典（OpenAI Function Calling 格式）

新增工具只需在 tools/ 下加文件，__init__.py 会自动发现。
"""

from tools.get_time import get_time as _tool_get_time
from tools.get_time import TOOL_DEFINITION as _DEF_GET_TIME
from tools.calculator import calculator as _tool_calculator
from tools.calculator import TOOL_DEFINITION as _DEF_CALCULATOR
from tools.web_search import web_search as _tool_web_search
from tools.web_search import TOOL_DEFINITION as _DEF_WEB_SEARCH
from tools.fetch_page import fetch_page as _tool_fetch_page
from tools.fetch_page import TOOL_DEFINITION as _DEF_FETCH_PAGE
from tools.summarize import summarize as _tool_summarize
from tools.summarize import TOOL_DEFINITION as _DEF_SUMMARIZE
from tools.dashboard import start_dashboard as _tool_start_dashboard
from tools.dashboard import TOOL_DEFINITION as _DEF_DASHBOARD

# 来自其他模块的工具（保持原有模块独立）
from rag import rag_query, RAG_TOOL_DEFINITION
from cot import tool_switch_cot_strategy, COT_TOOL_DEFINITION
from tot import tool_tot_reasoning, TOT_TOOL_DEFINITION
from prompts import tool_switch_role, ROLE_TOOL_DEFINITION
from context import tool_switch_context_strategy, CONTEXT_TOOL_DEFINITION
from harness import tool_toggle_sandbox, SANDBOX_TOOL_DEFINITION
from harness.recorder import clear_trajectories

# ===== TOOL_REGISTRY：name → 函数 =====
TOOL_REGISTRY = {
    "get_time": _tool_get_time,
    "calculator": _tool_calculator,
    "web_search": _tool_web_search,
    "fetch_page": _tool_fetch_page,
    "summarize": _tool_summarize,
    "rag_query": rag_query,
    "switch_cot_strategy": tool_switch_cot_strategy,
    "tot_reasoning": tool_tot_reasoning,
    "switch_role": tool_switch_role,
    "switch_context_strategy": tool_switch_context_strategy,
    "toggle_sandbox": tool_toggle_sandbox,
    "start_dashboard": _tool_start_dashboard,
    "clear_trajectories": clear_trajectories,
}

# ===== TOOL_DEFINITIONS：发给 LLM 的工具描述 =====
TOOL_DEFINITIONS = [
    _DEF_WEB_SEARCH,
    _DEF_CALCULATOR,
    _DEF_FETCH_PAGE,
    _DEF_SUMMARIZE,
    _DEF_GET_TIME,
    RAG_TOOL_DEFINITION,
    COT_TOOL_DEFINITION,
    TOT_TOOL_DEFINITION,
    ROLE_TOOL_DEFINITION,
    CONTEXT_TOOL_DEFINITION,
    SANDBOX_TOOL_DEFINITION,
    _DEF_DASHBOARD,
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
                        "description": "保留最近几天的文件（0=全部删除，7=保留近7天）"
                    }
                },
                "required": ["days"],
            },
        },
    },
]
