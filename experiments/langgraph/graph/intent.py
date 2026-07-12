"""意图分类 — 判断用户输入是功能测试还是生成式任务"""
import re


class TaskType:
    FUNCTIONAL_TEST = "functional_test"
    GENERATIVE_TASK = "generative_task"


_TEST_PATTERNS = [
    r"测试\w*(工具|功能|模块|搜索|计算|时间)",
    r"检查.*(能不能|是否正常|是否可用)",
    r"你(会|能|可以).*吗",
    r"你(有|支持).*(什么|哪些).*(工具|功能)",
    r"现在.*几[点分]",
    r"\d+\s*[+\-*/×÷]\s*\d+",
    r"搜索.*(是什么|是谁|的定义)",
    r"查(一下|一查)?\w*(天气|时间|新闻)",
]

_GENERATE_PATTERNS = [
    r"帮[我我]\w*(写|生成|创建|设计|画|做|分析|对比|总结|整理)",
    r"写一[份篇个段]\w*(报告|分析|总结|方案|计划|文章|代码)",
    r"(分析|对比|总结|概括|归纳).*\w+(报告|结果|数据|代码|优缺点)",
    r".*(优缺点|优劣势|区别|对比|比较).*(分析|总结)",
    r"搜索.*并.*(总结|分析|对比|整理)",
    r"(首先|第一步).*(然后|接着).*(最后)",
]


class IntentClassifier:
    def classify(self, text: str) -> str:
        text = text.strip()
        if self._match(_TEST_PATTERNS, text):
            return TaskType.FUNCTIONAL_TEST
        if self._match(_GENERATE_PATTERNS, text):
            return TaskType.GENERATIVE_TASK
        # 兜底：长文本或含换行 → 生成式
        if len(text) > 80 or text.count("\n") > 1:
            return TaskType.GENERATIVE_TASK
        return TaskType.FUNCTIONAL_TEST

    @staticmethod
    def _match(patterns, text):
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)
