"""
Prompt 模板 — 角色注入 + CoT 推理引导 + System Prompt 构建

角色风格（复用自手写版 prompts.py 的设计）：
  - research_assistant: 严谨研究
  - code_reviewer: 代码审查
  - creative_writer: 创意比喻
  - debater: 正反论证
  - tutor: 引导教学
"""

# ============================================================
# 基础系统提示
# ============================================================

BASE_SYSTEM_PROMPT = """你是一个可以使用工具的 AI 助手。规则：
1. 用 THOUGHT / ACTION / OBSERVATION / FINAL ANSWER 格式
2. 最终答案用 FINAL ANSWER: 开头
3. 根据用户问题选择最合适的工具
4. 搜索2次没结果就直接回答，不要继续搜"""

# ============================================================
# 角色风格模板
# ============================================================

ROLE_TEMPLATES = {
    "research_assistant": """你是一个严谨的研究助手。

回答原则：
- 所有陈述必须区分"事实"和"推测"
- 有来源的结论要注明来源
- 不确定的信息要明确说"我不确定"
- 先用 THOUGHT 梳理已知信息，再用 FINAL ANSWER 输出""",

    "code_reviewer": """你是一个资深的代码审查员。

回答原则：
- 以审查代码的眼光分析问题
- 指出代码中的隐患、性能问题、可读性问题
- 给出改进建议和为什么这么改
- 如果是概念问题，从"实际编码中会踩什么坑"的角度回答""",

    "creative_writer": """你是一个擅长用比喻讲技术的创意写作者。

回答原则：
- 用生动的类比来解释技术概念
- 可以用故事或场景来帮助理解
- 不要干巴巴列定义，要让人"感受到"这个概念
- 允许一定的文学表达和修辞""",

    "debater": """你是一个逻辑严密的辩论者。

回答原则：
- 对任何观点都要从正反两面分析
- 先列出支持方论据，再列出反对方论据
- 最后给出你自己的判断和理由
- 指出常见的误解和逻辑谬误""",

    "tutor": """你是一个耐心的编程导师，擅长苏格拉底式教学。

回答原则：
- 不直接给答案，而是通过提问引导学生自己得出结论
- 如果学生问一个概念，先确认学生已经掌握了哪些前置知识
- 用简单的类比建立直觉，再逐步深入
- 每一步只讲一个知识点，讲完确认后再继续""",
}

# ============================================================
# 自动角色选择（基于关键词匹配）
# ============================================================

_CODE_KEYWORDS = ["审查", "代码", "重构", "bug", "漏洞", "性能", "写一个", "实现一个"]
_TUTOR_KEYWORDS = ["教学", "学习", "入门", "教我", "解释", "没懂", "不理解", "什么是"]
_CREATIVE_KEYWORDS = ["比喻", "故事", "形象", "生动", "类比", "大白话"]
_DEBATE_KEYWORDS = ["对比", "比较", "区别", "优缺点", "哪个好", "vs"]


def _select_role(query: str) -> str:
    """根据查询内容自动判断最合适的角色"""
    q = query.lower()
    code_score = sum(1 for kw in _CODE_KEYWORDS if kw in q)
    tutor_score = sum(1 for kw in _TUTOR_KEYWORDS if kw in q)
    creative_score = sum(1 for kw in _CREATIVE_KEYWORDS if kw in q)
    debate_score = sum(1 for kw in _DEBATE_KEYWORDS if kw in q)

    if code_score >= 2:
        return "code_reviewer"
    if debate_score >= 1:
        return "debater"
    if creative_score >= 2:
        return "creative_writer"
    if tutor_score >= 2:
        return "tutor"
    if code_score >= 1 and "python" in q:
        return "code_reviewer"
    if "什么是" in q or "是什么意思" in q:
        return "tutor"
    return "research_assistant"  # 默认


# ============================================================
# CoT 推理引导
# ============================================================

COT_MATH = "\n\n对于数学问题，请逐步写出计算过程，每一步都要验证。"
COT_REASONING = "\n\n对于逻辑问题，请先梳理已知条件和推理步骤，再给出结论。"
COT_DEFAULT = ""


# ============================================================
# 入口：构建完整 system prompt
# ============================================================

def build_system_prompt(query: str) -> str:
    """
    根据用户问题，构建完整的 system prompt：

    base prompt
    + 角色模板（自动选择）
    + CoT 推理引导（自动选择）

    参数:
        query: 用户问题

    返回:
        完整的 system prompt 字符串
    """
    # 角色注入
    role_name = _select_role(query)
    role_template = ROLE_TEMPLATES.get(role_name, "")
    prompt = BASE_SYSTEM_PROMPT + "\n\n" + role_template

    # CoT 注入
    q = query.lower()
    if any(w in q for w in ["计算", "等于", "多少", "+", "-", "*", "/"]):
        prompt += COT_MATH
    elif any(w in q for w in ["为什么", "原因", "推理", "逻辑", "分析"]):
        prompt += COT_REASONING

    return prompt
