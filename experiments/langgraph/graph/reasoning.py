"""
推理增强策略 — Chain of Thought / 结构化推理

从手写版 src/handwritten_react_agent/cot.py 迁移，适配 LangGraph 的 prompt 构建方式。
"""
from enum import Enum
from typing import Optional


class Strategy(Enum):
    """推理策略"""
    ZERO_SHOT = "zero_shot"          # "Let's think step by step"
    FEW_SHOT_MATH = "few_shot_math"  # 数学问题示例
    FEW_SHOT_REASON = "few_shot_reason"  # 逻辑推理示例
    STRUCTURED = "structured"        # 固定框架分析
    DEFAULT = "default"              # 自动选择


# ── 策略特征关键词 ──

_MATH_KEYWORDS = ["计算", "等于", "方程", "比.*大", "速度", "距离",
                  "面积", "体积", "概率", "统计", "多少", "数字"]
_REASON_KEYWORDS = ["为什么", "原因", "推理", "逻辑", "论证", "判断",
                     "如果.*那么", "假设", "结论", "反驳"]
_STRUCTURED_KEYWORDS = ["分析", "对比", "评估", "优缺点", "影响",
                        "方案", "建议", "决策", "计划"]


def auto_select_strategy(query: str) -> Strategy:
    """根据问题自动选择推理策略"""
    q = query.lower()
    if any(k in q for k in _STRUCTURED_KEYWORDS):
        return Strategy.STRUCTURED
    if any(k in q for k in _MATH_KEYWORDS):
        return Strategy.FEW_SHOT_MATH
    if any(k in q for k in _REASON_KEYWORDS):
        return Strategy.FEW_SHOT_REASON
    return Strategy.ZERO_SHOT


def choose_strategy(query: str, strategy: Optional[str] = None) -> Strategy:
    """选择推理策略（用户指定或自动）"""
    if strategy:
        try:
            return Strategy(strategy)
        except ValueError:
            pass
    return auto_select_strategy(query)


# ── Prompt 模板 ──

def build_cot_prompt(query: str, strategy: Strategy) -> str:
    """根据策略生成 CoT 引导 prompt"""
    templates = {
        Strategy.ZERO_SHOT: (
            "请一步一步地推理，在给出最终答案前展示你的思考过程。\n"
            "格式：\n"
            "思考：...\n"
            "步骤 1：...\n"
            "步骤 2：...\n"
            "...\n"
            "答案：..."
        ),
        Strategy.FEW_SHOT_MATH: (
            "请逐步计算，参考以下示例：\n\n"
            "示例 1：\n"
            "问：小明有 5 个苹果，妈妈又给了他 3 个，他一共有几个？\n"
            "解：原来有 5 个，妈妈给了 3 个，所以 5 + 3 = 8\n"
            "答案：8 个\n\n"
            "示例 2：\n"
            "问：一个长方形长 6 米，宽 4 米，面积是多少？\n"
            "解：长方形面积 = 长 × 宽 = 6 × 4 = 24\n"
            "答案：24 平方米\n\n"
            f"问：{query}\n"
            "解："
        ),
        Strategy.FEW_SHOT_REASON: (
            "请按逻辑推理，参考以下示例：\n\n"
            "示例 1：\n"
            "问：所有的猫都喜欢吃鱼。Tom 是一只猫。Tom 喜欢吃鱼吗？\n"
            "推理：前提 1：所有的猫都喜欢吃鱼。前提 2：Tom 是一只猫。\n"
            "结论：Tom 喜欢吃鱼。\n"
            "答案：是的\n\n"
            "示例 2：\n"
            "问：今天是晴天。如果晴天我就去公园。我会去公园吗？\n"
            "推理：前提 1：今天是晴天。前提 2：如果晴天我就去公园。\n"
            "结论：我会去公园。\n"
            "答案：会\n\n"
            f"问：{query}\n"
            "推理："
        ),
        Strategy.STRUCTURED: (
            "请按以下框架分析问题：\n\n"
            "1. 问题定义：明确要解决的问题是什么\n"
            "2. 现状分析：列出已知条件和约束\n"
            "3. 多角度评估：从至少 2 个角度分析\n"
            "4. 权衡与优先级：各选项的优缺点\n"
            "5. 结论与建议：给出明确的结论\n\n"
            f"问题：{query}\n\n"
            "分析："
        ),
    }
    return templates.get(strategy, templates[Strategy.ZERO_SHOT])


def apply_cot(system_prompt: str, query: str, strategy: Optional[str] = None) -> str:
    """将 CoT 策略注入 system prompt"""
    strategy_enum = choose_strategy(query, strategy)
    cot_prompt = build_cot_prompt(query, strategy_enum)
    return f"{system_prompt}\n\n## 推理指引\n{cot_prompt}"
