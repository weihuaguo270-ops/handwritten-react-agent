"""
角色 Prompt 模板库 — 5 种不同角色的 System Prompt

支持自动角色选择（基于关键词匹配）和运行时切换。

用法:
    from prompts import ROLE_MANAGER, Role
    # 自动选择
    role = ROLE_MANAGER.select("帮我审查一下这段代码")
    # 手动指定
    system_prompt = ROLE_MANAGER.get_prompt("tutor", question="xxx")
    # 注入到 base prompt
    full_prompt = ROLE_MANAGER.inject(base_prompt, query="xxx")
"""

from enum import Enum
from typing import Optional


# ============================================================
# 1. 角色枚举
# ============================================================
class Role(Enum):
    RESEARCH_ASSISTANT = "research_assistant"
    CODE_REVIEWER = "code_reviewer"
    CREATIVE_WRITER = "creative_writer"
    DEBATER = "debater"
    TUTOR = "tutor"


# ============================================================
# 2. 角色 prompt 模板
# ============================================================
ROLE_PROMPTS = {
    Role.RESEARCH_ASSISTANT: """你是一个严谨的研究助手。

回答原则：
- 所有陈述必须区分"事实"和"推测"
- 有来源的结论要注明来源（如"根据 Python 官方文档"）
- 不确定的信息要明确说"我不确定"
- 先用 THOUGHT 梳理已知信息，再用 FINAL ANSWER 输出

{question}""",

    Role.CODE_REVIEWER: """你是一个资深的代码审查员。

回答原则：
- 以审查代码的眼光分析问题
- 指出代码中的隐患、性能问题、可读性问题
- 给出改进建议和为什么这么改
- 如果是概念问题，从"实际编码中会踩什么坑"的角度回答
- 先用 THOUGHT 梳理审查要点，再用 FINAL ANSWER 输出详细审查意见

{question}""",

    Role.CREATIVE_WRITER: """你是一个擅长用比喻讲技术的创意写作者。

回答原则：
- 用生动的类比来解释技术概念
- 可以用故事或场景来帮助理解
- 不要干巴巴列定义，要让人"感受到"这个概念
- 允许一定的文学表达和修辞
- 先用 THOUGTH 构思怎么讲这个故事，再用 FINAL ANSWER 输出

{question}""",

    Role.DEBATER: """你是一个逻辑严密的辩论者。

回答原则：
- 对任何观点都要从正反两面分析
- 先列出支持方论据，再列出反对方论据
- 最后给出你自己的判断和理由
- 指出常见的误解和逻辑谬误
- 先用 THOUGHT 梳理正反论点，再用 FINAL ANSWER 输出

{question}""",

    Role.TUTOR: """你是一个耐心的编程导师，擅长苏格拉底式教学。

回答原则：
- 不直接给答案，而是通过提问引导学生自己得出结论
- 如果学生问一个概念，先确认学生已经掌握了哪些前置知识
- 用简单的类比建立直觉，再逐步深入
- 可能会反问学生来确认理解
- 每一步只讲一个知识点，讲完确认后再继续
- 先用 THOUGHT 规划教学步骤，再用 FINAL ANSWER 输出

{question}""",
}


# ============================================================
# 3. 自动角色选择器
# ============================================================

# 代码类关键词 → code_reviewer
_CODE_KEYWORDS = [
    "审查", "review", "review", "代码", "代码质量", "重构", "优化",
    "bug", "漏洞", "安全问题", "性能", "复杂度", "设计模式",
    "写一个", "实现一个", "代码示例",
]

# 教学类关键词 → tutor
_TUTOR_KEYWORDS = [
    "教学", "学习", "入门", "教我", "解释", "没懂", "不理解",
    "初学者", "新手", "零基础", "教程", "什么是",
]

# 创意类关键词 → creative_writer
_CREATIVE_KEYWORDS = [
    "比喻", "故事", "形象", "生动", "通俗", "大白话",
    "类比", "举例说明", "帮我写一篇", "文案", "帖子",
]

# 分析比较类 → debater
_DEBATE_KEYWORDS = [
    "对比", "比较", "区别", "优缺点", "哪个好", "选择",
    "vs", "versus", "优劣", "利与弊",
]

# 研究类 → research_assistant（默认类别之外的兜底选择）


def _classify_query(query: str) -> Role:
    """根据查询内容自动判断最合适的角色"""
    q = query.lower()

    # 代码类优先（"写一个函数"这种强信号）
    code_score = sum(1 for kw in _CODE_KEYWORDS if kw.lower() in q)
    tutor_score = sum(1 for kw in _TUTOR_KEYWORDS if kw.lower() in q)
    creative_score = sum(1 for kw in _CREATIVE_KEYWORDS if kw.lower() in q)
    debate_score = sum(1 for kw in _DEBATE_KEYWORDS if kw.lower() in q)

    # 取舍逻辑：信号最强的优先
    if code_score >= 2:
        return Role.CODE_REVIEWER
    if debate_score >= 1:
        return Role.DEBATER
    if creative_score >= 2:
        return Role.CREATIVE_WRITER
    if tutor_score >= 2:
        return Role.TUTOR

    # 弱信号 + 兜底
    if code_score >= 1 and "python" in q:
        return Role.CODE_REVIEWER
    if "什么是" in q or "是什么意思" in q:
        return Role.TUTOR
    if tutor_score >= 1 and creative_score >= 1:
        return Role.CREATIVE_WRITER

    # 默认：research_assistant（最通用严谨）
    return Role.RESEARCH_ASSISTANT


# ============================================================
# 4. RoleManager 核心类
# ============================================================

class RoleManager:
    """角色管理

    用法:
        rm = RoleManager()
        role = rm.select("帮我审查这段代码")
        prompt = rm.get_prompt(role, question="xxx")
        full = rm.inject(base_prompt, query="xxx")
    """

    def __init__(self, default_role: Role = Role.RESEARCH_ASSISTANT):
        self._default = default_role
        self.current_role = default_role

    def select(self, query: str) -> Role:
        """根据查询自动选择角色"""
        self.current_role = _classify_query(query)
        return self.current_role

    def set_role(self, role_name: str) -> str:
        """运行时切换角色（通过工具调用）"""
        try:
            self.current_role = Role(role_name)
            return f"角色已切换为 {role_name}"
        except ValueError:
            available = ", ".join(r.value for r in Role)
            return f"未知角色: {role_name}，可选: {available}"

    def get_prompt(self, role: Optional[Role] = None,
                   query: Optional[str] = None,
                   question: str = "") -> str:
        """获取指定角色的 prompt 文本

        如果 role 为 None 且 query 不为 None，自动选择
        """
        if role is None:
            if query is not None:
                role = self.select(query)
            else:
                role = self._default
        template = ROLE_PROMPTS.get(role)
        if template is None:
            template = ROLE_PROMPTS[Role.RESEARCH_ASSISTANT]
        return template.format(question=question)

    def inject(self, base_system_prompt: str,
               query: Optional[str] = None,
               role: Optional[Role] = None,
               question: str = "") -> str:
        """将角色 prompt 注入到 system prompt 末尾"""
        role_prompt = self.get_prompt(role=role, query=query, question=question)
        # 角色 prompt 已包含 {question} 占位，用空字符串填充
        role_text = role_prompt if question else role_prompt.replace("{question}", "")
        return base_system_prompt.rstrip() + "\n\n" + role_text

    def current_role_name(self) -> str:
        return self.current_role.value

    def list_roles(self) -> list[str]:
        return [r.value for r in Role]


# ============================================================
# 5. 全局实例
# ============================================================

ROLE_MANAGER = RoleManager()


# ============================================================
# 6. 工具定义 + 工具函数
# ============================================================

ROLE_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "switch_role",
        "description": "切换 AI 助手的角色风格，影响回答方式。"
                       "research_assistant(严谨研究)/code_reviewer(代码审查)/"
                       "creative_writer(创意比喻)/debater(正反论证)/tutor(引导教学)",
        "parameters": {
            "type": "object",
            "properties": {
                "role": {
                    "type": "string",
                    "enum": ["research_assistant", "code_reviewer",
                             "creative_writer", "debater", "tutor"],
                    "description": "目标角色名称"
                }
            },
            "required": ["role"],
        },
    },
}


def tool_switch_role(role: str) -> str:
    """运行时切换角色风格（可供 LLM 调用的工具）"""
    return ROLE_MANAGER.set_role(role)
