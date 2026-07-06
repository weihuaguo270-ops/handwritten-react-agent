"""
Prompt 模板 — 替代手写 prompts.py + cot.py

角色注入 + CoT 推理引导 + System Prompt 构建。
"""

# 基础系统提示
BASE_SYSTEM_PROMPT = """你是一个可以使用工具的 AI 助手。规则：
1. 用 THOUGHT / ACTION / OBSERVATION / FINAL ANSWER 格式
2. 最终答案用 FINAL ANSWER: 开头
3. 根据用户问题选择最合适的工具
4. 搜索2次没结果就直接回答，不要继续搜"""

# CoT 推理引导
COT_MATH = "\n\n对于数学问题，请逐步写出计算过程，每一步都要验证。"
COT_REASONING = "\n\n对于逻辑问题，请先梳理已知条件和推理步骤，再给出结论。"
COT_DEFAULT = ""


def inject_cot(query: str, base: str = BASE_SYSTEM_PROMPT) -> str:
    """根据问题类型注入 CoT 引导到 system prompt 末尾"""
    q = query.lower()
    if any(w in q for w in ["计算", "等于", "多少", "+", "-", "*", "/"]):
        return base + COT_MATH
    if any(w in q for w in ["为什么", "原因", "推理", "逻辑", "分析"]):
        return base + COT_REASONING
    return base + COT_DEFAULT
