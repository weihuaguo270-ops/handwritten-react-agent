"""
思维链（Chain-of-Thought）策略模块

为 ReAct Loop 提供多种推理引导策略，根据问题类型自动选择最优 prompt。

支持的策略:
  - ZERO_SHOT:     "Let's think step by step"（最简单，通用）
  - FEW_SHOT_MATH: 数学/计算问题（给 2 个带步骤的示例）
  - FEW_SHOT_REASONING: 逻辑推理问题（给 2 个逻辑推理示例）
  - STRUCTURED:    复杂问题强制按固定框架思考

每次查询只影响 system prompt 的追加内容，不改变主循环逻辑。
"""

from enum import Enum
from typing import Optional


# ============================================================
# 1. 策略枚举
# ============================================================
class CoTStrategy(Enum):
    """支持的 CoT 策略类型"""
    ZERO_SHOT = "zero_shot"            # 最简单：直接要求逐步思考
    FEW_SHOT_MATH = "few_shot_math"    # 数学：给数学推理示例
    FEW_SHOT_REASONING = "few_shot_reasoning"  # 逻辑推理：给逻辑推理示例
    STRUCTURED = "structured"          # 结构化：强制按框架思考


# ============================================================
# 2. 各策略对应的 prompt 模板
# ============================================================

_ZERO_SHOT_PROMPT = """在回答之前，请一步一步思考（think step by step）。
把每一步推理过程写在 THOUGHT 块中，最后给出 FINAL ANSWER。"""

_FEW_SHOT_MATH_EXAMPLES = """这里有两个"先一步步推理再回答"的例子——注意它们是怎么把问题拆成小步骤的：

[示例1]
用户: 一个苹果 5 元，一个西瓜比苹果贵 18 元，买 3 个西瓜多少钱？
助手:
THOUGHT: 先算西瓜单价。西瓜比苹果贵 18 元，苹果 5 元，所以西瓜 = 5 + 18 = 23 元。
THOUGHT: 再算 3 个西瓜总价。23 × 3 = 69 元。
FINAL ANSWER: 69 元

[示例2]
用户: 小明有 24 颗糖，给了小红 1/3，又给了小刚剩下的一半，小明还剩多少颗？
助手:
THOUGHT: 第一步：24 颗的 1/3 = 24 ÷ 3 = 8 颗给了小红。
THOUGHT: 第二步：剩下 24 - 8 = 16 颗。
THOUGHT: 第三步：剩下的一半给小刚 = 16 ÷ 2 = 8 颗。
THOUGHT: 第四步：小明还剩 16 - 8 = 8 颗。
FINAL ANSWER: 8 颗"""

_FEW_SHOT_REASONING_PROMPT = """回答前请先逐步分析推理，把每一步写在 THOUGHT 中。

参考下面的推理方式——先列出已知条件，再一步步推结论：

[示例1]
用户: 所有猫都怕水。Tom 是一只猫。Tom 怕水吗？
助手:
THOUGHT: 条件1：所有猫都怕水。（全称命题）
THOUGHT: 条件2：Tom 是一只猫。（个体属于集合）
THOUGHT: 根据条件1和2，由全称命题推出个体结论：Tom 怕水。
FINAL ANSWER: 是的，Tom 怕水。

[示例2]
用户: 如果今天下雨，我就不去公园。今天没下雨。我会去公园吗？
助手:
THOUGHT: 条件1：下雨 → 不去公园。（P→Q）
THOUGHT: 条件2：没下雨。（¬P）
THOUGHT: 注意：¬P 不能推出 ¬Q（否定前件不能否定后件）。所以不能推出"去"或"不去"。
THOUGHT: 条件不足，无法确定。
FINAL ANSWER: 无法确定。没下雨只是"不去公园"的条件不成立，但我可能因为其他原因不去。"""

_STRUCTURED_PROMPT = """对于这个问题，请按以下框架逐步思考：

[分析] 问题在问什么？有哪些已知条件？
[拆解] 需要哪几步才能得到答案？
[步骤] 一步步执行
[验证] 答案合理吗？有没有遗漏？

每一步用 THOUGHT 标签，最终答案用 FINAL ANSWER 标签。"""


# ============================================================
# 3. 策略选择器（基于关键词自动匹配）
# ============================================================

# 数学类关键词
_MATH_KEYWORDS = [
    "计算", "多少", "等于", "总共", "平均", "比...大", "比...小", "比...贵", "比...便宜",
    "+", "-", "×", "÷", "加", "减", "乘", "除",
    "percent", "percent", "count", "sum", "total", "average",
    "数字", "数量", "价格", "成本", "距离", "速度", "时间",
    "倍", "比例", "概率",
]

# 逻辑推理类关键词
_REASON_KEYWORDS = [
    "如果", "那么", "否则", "推理", "逻辑", "假设",
    "原因", "结果", "可能", "必然", "所有", "有些",
    "if", "then", "else", "because", "therefore",
    "推断", "结论", "前提", "条件",
    "就",           # 如果…就…（条件句标志）
    "要么", "或者", "且",
    "真假", "对错", "是否",
    "所以", "因此", "都",       # 三段论标志："所有…都…，…所以…"
]

# 研究/搜索类关键词（不需要复杂推理，用最简单的 CoT）
_RESEARCH_KEYWORDS = [
    "搜索", "查询", "查找", "最新", "今天", "新闻",
    "天气", "股价", "汇率", "search", "find", "latest",
    "current", "today", "news", "weather",
]


def _classify_query(query: str) -> CoTStrategy:
    """根据查询内容自动判断最合适的 CoT 策略"""
    q = query.lower()

    # 排除"对比""比较"中的"比"误触（它们表示"compare"不是数学比较）
    _math_q = q.replace("对比", "XX").replace("比较", "XX")

    math_score = sum(1 for kw in _MATH_KEYWORDS if kw.lower() in _math_q)
    reason_score = sum(1 for kw in _REASON_KEYWORDS if kw.lower() in q)
    research_score = sum(1 for kw in _RESEARCH_KEYWORDS if kw.lower() in q)

    # 研究类用最简单的 zero-shot（重点在搜索不在推理）
    if research_score >= 2 and research_score > math_score and research_score > reason_score:
        return CoTStrategy.ZERO_SHOT

    # 数学类用 few-shot math
    if math_score >= 2 or (math_score >= 1 and reason_score == 0):
        return CoTStrategy.FEW_SHOT_MATH

    # 逻辑推理类
    if reason_score >= 2:
        return CoTStrategy.FEW_SHOT_REASONING

    # 复杂问题（字多、有步骤性）用结构化
    if len(query) > 40 and (query.count("，") >= 3 or query.count(",") >= 3):
        return CoTStrategy.STRUCTURED

    # 默认用最简单的 zero-shot
    return CoTStrategy.ZERO_SHOT


# ============================================================
# 4. 策略 → prompt 文本的映射
# ============================================================

_STRATEGY_PROMPTS = {
    CoTStrategy.ZERO_SHOT: _ZERO_SHOT_PROMPT,
    CoTStrategy.FEW_SHOT_MATH: _FEW_SHOT_MATH_EXAMPLES,
    CoTStrategy.FEW_SHOT_REASONING: _FEW_SHOT_REASONING_PROMPT,
    CoTStrategy.STRUCTURED: _STRUCTURED_PROMPT,
}


# ============================================================
# 5. 推理提取器
# ============================================================

def extract_reasoning(text: str) -> tuple:
    """从 LLM 输出中提取推理过程和最终答案

    返回:
        (reasoning_parts: list[str], final_answer: str | None)
        - reasoning_parts: 所有的 THOUGHT 块内容列表
        - final_answer: FINAL ANSWER 后面的内容，如果没有则返回 None
    """
    import re

    # 提取所有 THOUGHT 块
    thought_pattern = r"THOUGHT:\s*(.*?)(?=THOUGHT:|FINAL ANSWER:|$)"
    thoughts = re.findall(thought_pattern, text, re.DOTALL)
    thoughts = [t.strip() for t in thoughts if t.strip()]

    # 提取 FINAL ANSWER
    final = None
    fa_pattern = r"FINAL ANSWER:\s*(.*)"
    fa_match = re.search(fa_pattern, text, re.DOTALL)
    if fa_match:
        final = fa_match.group(1).strip()

    return thoughts, final


# ============================================================
# 6. CoT 核心类
# ============================================================

class CoT:
    """思维链策略管理

    用法:
        cot = CoT()
        # 自动选择策略并注入到 system prompt
        system_prompt = cot.inject(base_system_prompt, user_query=query)
        
        # 或者手动指定策略
        system_prompt = cot.inject(base_system_prompt, strategy=CoTStrategy.FEW_SHOT_MATH)
        
        # 从 LLM 回复中提取推理过程
        thoughts, final = cot.parse(response_text)
    """

    def __init__(self, default_strategy: CoTStrategy = CoTStrategy.ZERO_SHOT):
        self._default = default_strategy

    def select(self, query: str) -> CoTStrategy:
        """根据查询内容自动选择最合适的策略"""
        return _classify_query(query)

    def get_prompt(self, strategy: Optional[CoTStrategy] = None, query: Optional[str] = None) -> str:
        """获取指定策略的 prompt 文本

        如果 strategy 为 None 且 query 不为 None，则自动选择。
        如果两者都提供，以 strategy 为准。
        """
        if strategy is None:
            if query is not None:
                strategy = self.select(query)
            else:
                strategy = self._default
        return _STRATEGY_PROMPTS.get(strategy, _ZERO_SHOT_PROMPT)

    def inject(self, base_system_prompt: str,
               query: Optional[str] = None,
               strategy: Optional[CoTStrategy] = None) -> str:
        """将 CoT 策略注入到 system prompt 末尾

        参数:
            base_system_prompt: 原始的 system prompt
            query: 用户查询（用于自动选择策略）
            strategy: 手动指定策略（优先级高于自动选择）

        返回:
            拼接了 CoT 指令的完整 system prompt
        """
        cot_text = self.get_prompt(strategy=strategy, query=query)
        return base_system_prompt.rstrip() + "\n\n" + cot_text

    @staticmethod
    def parse(response_text: str) -> tuple:
        """从 LLM 回复中提取推理过程和最终答案

        返回:
            (thoughts: list[str], final_answer: str | None)
        """
        return extract_reasoning(response_text)


# ============================================================
# 7. 全局实例 + 便捷工具函数
# ============================================================

# 预创建的全局实例——项目其他模块 import COT 即可用
COT = CoT()


def build_cot_prompt(base_system_prompt: str, query: str) -> str:
    """单步调用入口：传入原始 system prompt 和用户问题，返回带 CoT 的完整 prompt

    用法:
        from react_agent.cot import build_cot_prompt
        system_prompt = build_cot_prompt(base, user_query)
    """
    return COT.inject(base_system_prompt, query=query)


# ============================================================
# 8. 工具定义（供 react_loop.py 注册为工具）
# ============================================================

COT_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "switch_cot_strategy",
        "description": "切换 CoT（思维链）推理策略，影响 LLM 后续的推理方式。可选: zero_shot(通用), few_shot_math(数学), few_shot_reasoning(逻辑推理), structured(复杂问题结构化)",
        "parameters": {
            "type": "object",
            "properties": {
                "strategy": {
                    "type": "string",
                    "enum": ["zero_shot", "few_shot_math", "few_shot_reasoning", "structured"],
                    "description": "目标策略名称"
                }
            },
            "required": ["strategy"],
        },
    },
}


def tool_switch_cot_strategy(strategy: str) -> str:
    """运行时切换 CoT 策略——可通过 LLM 工具调用来动态改变推理方式"""
    try:
        s = CoTStrategy(strategy)
        COT._default = s
        return f"已切换为 {s.value} 策略"
    except ValueError:
        return f"未知策略: {strategy}，可选: zero_shot, few_shot_math, few_shot_reasoning, structured"
