"""
任务规划器（Planner）— LLM 驱动的任务分解 + 依赖分析

让 LLM 自动将用户请求分解为子任务，并分析任务间的依赖关系。
通过拓扑排序确定执行顺序：无依赖的先执行（可并行），有依赖的等前置完成。

用法:
    from handwritten_react_agent.planner import PLANNER
    tasks = PLANNER.plan("查今天和明天的天气，对比温差")
    # → [Task(id='1', desc='搜索今天天气', depends_on=[]),
    #     Task(id='2', desc='搜索明天天气', depends_on=[]),
    #     Task(id='3', desc='对比温差', depends_on=['1', '2'])]
"""

from typing import Optional


# ============================================================
# 1. Task 数据类
# ============================================================
class Task:
    """一个可调度的任务单元

    属性:
        id:         任务编号（"1", "2", ...）
        description: 任务描述
        depends_on:  依赖的任务 ID 列表，这些任务必须完成才能执行本任务
        result:      执行结果（执行前为 None）
    """
    def __init__(self, id: str, description: str, depends_on: list[str] = None):
        self.id = id
        self.description = description
        self.depends_on = depends_on or []
        self.result = None

    def ready(self, completed_ids: set[str]) -> bool:
        """检查当前任务是否已经满足依赖（可以执行了）"""
        return all(dep in completed_ids for dep in self.depends_on)

    def __repr__(self):
        deps = f", 依赖: {self.depends_on}" if self.depends_on else ""
        return f"<Task #{self.id}: {self.description[:40]}{deps}>"


# ============================================================
# 2. Planner 核心类
# ============================================================

_PLAN_PROMPT = """你是一个专业的任务分解专家。将用户的请求拆成可执行的子任务，并分析任务之间的依赖关系。

要求：
- 每个子任务只做一件事
- 用 task_N: 描述 的格式
- 如果某个任务依赖其他任务先完成，在后面加 | depends_on: N, M
- 没有依赖的任务可以并行执行
- 不要解释，直接输出任务列表

例子1（搜索+对比）:
请求: 搜索今天和明天的天气，对比温差
输出:
task_1: 搜索今天北京天气
task_2: 搜索明天北京天气
task_3: 对比今天和明天的温差 | depends_on: 1, 2

例子2（计算）:
请求: 帮我查一下计算器的历史，然后算一下123*456
输出:
task_1: 搜索计算器的历史
task_2: 计算123乘以456

例子3（搜索+总结+写作）:
请求: 搜索最新的AI新闻，用中文总结，然后写一个Twitter帖子
输出:
task_1: 搜索最新AI新闻
task_2: 用中文总结搜索到的AI新闻 | depends_on: 1
task_3: 写一个关于AI新闻的Twitter帖子 | depends_on: 2

例子4（代码生成+计算）:
请求: 用Python生成10个随机数，并计算它们的平均值和标准差
输出:
task_1: 用Python生成10个随机数
task_2: 计算平均值和标准差 | depends_on: 1

例子5（代码+多步分析）:
请求: 同时生成一个3x3随机矩阵，并计算每行每列的和
输出:
task_1: 用Python生成3x3随机矩阵
task_2: 计算每行每列的和 | depends_on: 1

现在处理以下请求：
请求: {query}
输出:"""


class Planner:
    """任务规划器

    用法:
        planner = Planner()

        def my_llm(messages):
            return call_llm(messages)

        tasks = planner.plan("搜索今天天气并对比明天", llm_call=my_llm)
        order = planner.schedule(tasks)
        # order = [[task1, task2], [task3]]  # 同层可并行
    """

    # 模板匹配列表（优先于 LLM 分解，零成本、稳定）
    # 每项: (正则, 构建函数) — 正则匹配则直接用函数构建任务列表
    _TEMPLATES = [
        # 搜索+对比类: "搜索今天和明天的天气" "查A和B"
        (r"(?:搜索|查|查找|搜一下)(.+?)(?:和|与|跟|、)(.+)",
         lambda m: [
             Task("1", f"搜索{m.group(1).strip()}"),
             Task("2", f"搜索{m.group(2).strip()}"),
             Task("3", "对比分析", depends_on=["1", "2"]),
         ]),
        # 代码生成+计算类: "生成10个随机数，计算平均值" "用Python生成..."
        (r"(?:用Python|使用Python|通过Python)?(?:生成|创建|构造)(.+?)(?:，|,)(?:并)?(?:计算|统计|分析)(.+)",
         lambda m: [
             Task("1", f"用Python生成{m.group(1).strip()}"),
             Task("2", f"计算{m.group(2).strip()}", depends_on=["1"]),
         ]),
        # 计算类: "计算123*456" "算一下..."
        (r"(?:计算|算一下|算)(.+)",
         lambda m: [
             Task("1", f"计算{m.group(1).strip()}"),
         ]),
        # 搜索单个: "搜索XXX" "查XXX"
        (r"(?:搜索|查|查找|搜一下)(.+)",
         lambda m: [
             Task("1", f"搜索{m.group(1).strip()}"),
         ]),
    ]

    def __init__(self):
        self._last_query = ""

    def plan(self, query: str, llm_call: callable = None) -> list[Task]:
        """分解任务为子任务

        优先匹配模板（零 LLM 调用），未命中则走 LLM 自动分解。
        """
        self._last_query = query

        # 第1关：模板匹配（模型无关、零成本、稳定）
        import re
        for pattern, builder in self._TEMPLATES:
            m = re.search(pattern, query)
            if m:
                tasks = builder(m)
                if tasks:
                    return tasks

        # 第2关：兜底走 LLM 自动分解
        if llm_call is None:
            return []
        prompt = _PLAN_PROMPT.format(query=query)
        msg = llm_call([
            {"role": "system", "content": "你是一个专业的任务分解助手，严格按格式输出。"},
            {"role": "user", "content": prompt},
        ])
        content = (msg.get("content", "") or "").strip()
        return self._parse_tasks(content)

    @staticmethod
    def _parse_tasks(text: str) -> list[Task]:
        """解析 LLM 输出为 Task 列表

        解析格式:
            task_1: 描述文字
            task_2: 描述文字 | depends_on: 1
        """
        tasks = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue

            # 匹配 task_N: 开头
            if not line.startswith("task_"):
                continue

            # 去掉 task_N: 前缀
            # task_1: 描述 | depends_on: 2, 3
            rest = line.split(":", 1)[1].strip() if ":" in line else ""

            # 分离描述和依赖部分
            description = rest
            depends_on = []

            if "|" in rest:
                parts = rest.split("|", 1)
                description = parts[0].strip()
                dep_part = parts[1].strip().lower()
                if "depends_on" in dep_part or "depends on" in dep_part:
                    # 提取数字列表
                    dep_text = dep_part.replace("depends_on:", "").replace("depends on:", "").strip()
                    depends_on = [d.strip() for d in dep_text.split(",") if d.strip()]

            if description:
                tasks.append(Task(
                    id=str(len(tasks) + 1),
                    description=description,
                    depends_on=depends_on,
                ))

        return tasks

    @staticmethod
    def schedule(tasks: list[Task]) -> list[list[Task]]:
        """拓扑排序：确定任务的执行层级

        返回:
            [[level0_tasks], [level1_tasks], ...]
            同层可以并行执行，不同层必须按顺序
        """
        remaining = {t.id: t for t in tasks}
        levels = []

        while remaining:
            current_level = []
            for tid in list(remaining.keys()):
                t = remaining[tid]
                # 检查所有依赖是否已在之前的层级中完成
                if all(dep not in remaining for dep in t.depends_on):
                    current_level.append(t)

            if not current_level:
                # 死锁检测：还有任务但无法推进（可能有循环依赖）
                break

            for t in current_level:
                del remaining[t.id]

            # 按 ID 排序保持稳定顺序
            current_level.sort(key=lambda x: int(x.id) if x.id.isdigit() else 0)
            levels.append(current_level)

        return levels

    @staticmethod
    def describe_schedule(levels: list[list[Task]]) -> str:
        """把调度层级格式化成可读文本"""
        lines = [f"共 {sum(len(l) for l in levels)} 个任务，{len(levels)} 个层级："]
        for i, level in enumerate(levels):
            task_desc = ", ".join([f"#{t.id} {t.description[:30]}" for t in level])
            parallel = "（可并行）" if len(level) > 1 else ""
            lines.append(f"  第{i+1}层: {task_desc}{parallel}")
        return "\n".join(lines)


# ============================================================
# 3. 全局实例
# ============================================================

PLANNER = Planner()


# ============================================================
# 4. 工具定义（供 react_loop.py 注册）
# ============================================================

PLANNER_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "plan_tasks",
        "description": "将复杂请求拆成子任务并分析它们的依赖关系。"
                       "适用于需要多步骤、前后有依赖的复杂任务。"
                       "返回每个子任务及其依赖关系。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "要分解的请求"
                },
            },
            "required": ["query"],
        },
    },
}


# ============================================================
# 5. 工具函数（注入 llm_call 后用）
# ============================================================

_planner_llm_call = None


def set_planner_llm_call(func):
    """设置 Planner 使用的 LLM 调用函数"""
    global _planner_llm_call
    _planner_llm_call = func


def tool_plan_tasks(query: str) -> str:
    """在 ReAct Loop 中调用的任务分解工具"""
    global _planner_llm_call
    if _planner_llm_call is None:
        return "错误: Planner 的 LLM 调用函数未设置"

    planner = Planner()
    tasks = planner.plan(query, llm_call=_planner_llm_call)
    if not tasks:
        return "未能分解任务"

    levels = planner.schedule(tasks)
    output = [f"[Planner] 分解为 {len(tasks)} 个子任务："]
    for t in tasks:
        deps = f"（等待 {'、'.join(['#' + d for d in t.depends_on])}）" if t.depends_on else "（无依赖，可立即执行）"
        output.append(f"  #{t.id}: {t.description} {deps}")

    output.append("")
    output.append(planner.describe_schedule(levels))
    return "\n".join(output)
