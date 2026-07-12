"""classifier — 任务意图自动分类

根据用户输入自动判断任务类型：
  - functional_test:  测试工具/功能是否正常 → 直接返回用户
  - generative_task:  复杂生成式任务 → 进入 Eval Loop

分类策略：
  1. 正则模板匹配（快速，零 LLM 成本）
  2. 匹配不到 → 降级到 LLM 判断（慢但有兜底）

正则模板持续更新——每次分类错误时记录，后续补充模板。
"""

from __future__ import annotations
import re
from typing import Callable, Optional


# ──────────────────────────────────────────────
# 任务类型
# ──────────────────────────────────────────────


class TaskType:
    """任务类型常量"""
    FUNCTIONAL_TEST = "functional_test"   # 测试类：返回给用户
    GENERATIVE_TASK = "generative_task"   # 生成类：进入 Eval Loop


# ──────────────────────────────────────────────
# 正则模板
# ──────────────────────────────────────────────


# 功能测试类特征
_TEST_PATTERNS = [
    # 明确要求测试
    r"测试\w*(工具|功能|模块|搜索|计算|时间)",
    r"帮我\w*(跑|做|执行).*(测试|评估|评测|验证|检查|test|eval)",
    r"检查.*(能不能|是否正常|是否可用|有没有毛病)",
    # 直接问 Agent 能力
    r"你(会|能|可以).*吗",
    r"你(有|支持).*(什么|哪些).*(工具|功能)",
    r"展示.*(能力|工具|功能)",
    # 简单单次查询
    r"现在.*几[点分]",
    r"\d+\s*[+\-*/×÷]\s*\d+",  # 纯计算
    r"搜索.*(是什么|是谁|的定义|的定义是什么)",
    r"查(一下|一查)?\w*(天气|时间|新闻)",
]

# 生成式任务特征
_GENERATE_PATTERNS = [
    # 复杂生成
    r"帮[我我]\w*(写|生成|创建|设计|画|做|制作|分析|对比|总结|整理)",
    r"写一[份篇个段]\w*(报告|分析|总结|方案|计划|文章|代码|脚本|函数|邮件|算法)",
    r"(写|生成|创建).*(代码|函数|脚本|算法|程序)",
    r"生成一[份篇个段]",
    # 分析/对比/总结
    r"对.*进行.*(分析|对比|总结|评估|审查|优化)",
    r"(分析|对比|总结|概括|归纳).*\w+(报告|结果|数据|信息|代码|优缺点|区别|差异)",
    r".*(优缺点|优劣势|区别|对比|比较).*(分析|总结)",
    r"从.*(提取|生成|推导|计算).*并",
    r"把.*(整理|汇总|翻译|改写|转换)成",
    # 多步复合
    r"(首先|第一步|先).*(然后|接着|第二步).*(最后|第三步)",
    r"同时.*并.*还",
    r"既.*又.*还",
    r"分别.*和.*以及",
    # 搜索+处理复合
    r"搜索.*并.*(总结|分析|对比|整理|归纳|汇总)",
    r"查.*并.*(总结|分析|对比|整理|汇总)",
]


# ──────────────────────────────────────────────
# 分类器
# ──────────────────────────────────────────────


class IntentClassifier:
    """任务意图分类器

    用法：
        classifier = IntentClassifier()
        task_type = classifier.classify("帮我写一份关于AI的报告")
        # → TaskType.GENERATIVE_TASK
    """

    def __init__(
        self,
        llm_classifier: Optional[Callable[[str], str]] = None,
        debug: bool = False,
    ):
        """初始化分类器

        参数:
            llm_classifier: 可选，LLM 降级分类函数。
                            输入用户 query，返回 "functional_test" 或 "generative_task"
            debug:          是否输出调试信息
        """
        self.llm_classifier = llm_classifier
        self.debug = debug
        self._unknown_count = 0

    def classify(self, user_input: str) -> str:
        """判断用户输入的任务类型

        优先级：测试类 > 生成类 > LLM 降级 > 默认生成类

        参数:
            user_input: 用户输入的字符串

        返回:
            TaskType.FUNCTIONAL_TEST 或 TaskType.GENERATIVE_TASK
        """
        user_input = user_input.strip()

        # 1. 测试类正则匹配
        if self._match_any(_TEST_PATTERNS, user_input):
            self._log(f"正则匹配 → functional_test")
            return TaskType.FUNCTIONAL_TEST

        # 2. 生成类正则匹配
        if self._match_any(_GENERATE_PATTERNS, user_input):
            self._log(f"正则匹配 → generative_task")
            return TaskType.GENERATIVE_TASK

        # 3. LLM 降级判断
        if self.llm_classifier:
            try:
                result = self.llm_classifier(user_input)
                if result in (TaskType.FUNCTIONAL_TEST, TaskType.GENERATIVE_TASK):
                    self._log(f"LLM 降级 → {result}")
                    return result
            except Exception as e:
                self._log(f"LLM 降级异常: {e}")

        # 4. 兜底：按输入长度和复杂度判断
        self._unknown_count += 1
        if len(user_input) > 80 or user_input.count("\n") > 1:
            self._log(f"兜底（长文本）→ generative_task")
            return TaskType.GENERATIVE_TASK

        self._log(f"兜底（短文本）→ functional_test")
        return TaskType.FUNCTIONAL_TEST

    def _match_any(self, patterns: list[str], text: str) -> bool:
        """检查文本是否匹配任一正则"""
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _log(self, msg: str) -> None:
        """调试输出"""
        if self.debug:
            print(f"[IntentClassifier] {msg}")

    def report(self) -> dict:
        """返回分类统计"""
        return {
            "unknown_fallback_count": self._unknown_count,
            "test_patterns_count": len(_TEST_PATTERNS),
            "generate_patterns_count": len(_GENERATE_PATTERNS),
        }


# ──────────────────────────────────────────────
# LLM 降级分类 prompt
# ──────────────────────────────────────────────


GENERIC_CLASSIFIER_PROMPT = """判断以下用户输入属于哪一类：

- "functional_test": 用户在测试工具/功能是否正常，或询问 Agent 的能力边界
- "generative_task": 用户要求生成复杂内容（报告/分析/代码/方案），或需要多步推理

只输出一个词："functional_test" 或 "generative_task"，不要附加其他内容。

用户输入: {query}
分类:"""


def default_llm_classifier_fn(
    query: str,
    llm_call: Callable[[str], str],
) -> str:
    """默认的 LLM 降级分类函数

    参数:
        query:    用户输入
        llm_call: LLM 调用函数（输入 prompt，输出文本）

    返回:
        "functional_test" 或 "generative_task"
    """
    prompt = GENERIC_CLASSIFIER_PROMPT.format(query=query)
    result = llm_call(prompt).strip().lower()
    if "generative" in result or "generate" in result:
        return TaskType.GENERATIVE_TASK
    return TaskType.FUNCTIONAL_TEST
