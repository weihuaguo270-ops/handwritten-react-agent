
"""
Orchestrator — 独立的多 Agent 协作模块
支持 Worker 隔离：每个 Worker 有独立的工具集
"""
import json, urllib.request, sys, os
from concurrent.futures import ThreadPoolExecutor, as_completed
from handwritten_react_agent.planner import Planner, Task


TOOL_PROFILES = {
    "time": {"get_current_time", "convert_time"},
    "file": {"read_text_file", "write_file", "edit_file", "create_directory",
             "list_directory", "directory_tree", "move_file", "search_files",
             "get_file_info", "list_allowed_directories"},
    "web": {"web_search", "fetch_page"},
    "calc": {"calculator"},
    "summary": {"summarize"},
    "code": {"execute_python"},
}

PROFILE_HINTS = {
    "time": "查询时间、时区转换",
    "file": "文件读写、目录管理、文件信息",
    "web": "搜索互联网、读取网页",
    "calc": "数学计算",
    "summary": "文本摘要",
    "code": "Python 代码执行、数据分析、脚本编写",
}


def classify_tool_needs(task, call_llm):
    task_lower = task.lower()
    tags = set()
    if any(w in task_lower for w in ["时间", "时区", "当前时间", "现在几点", "纽约", "伦敦"]):
        tags.add("time")
    if any(w in task_lower for w in ["文件", "目录", "文件夹", "大小", "写", "读", "创建"]):
        tags.add("file")
    if any(w in task_lower for w in ["搜索", "网页", "新闻", "查询"]):
        tags.add("web")
    if any(w in task_lower for w in ["计算", "数学", "等于"]):
        tags.add("calc")
    if any(w in task_lower for w in ["总结", "摘要", "概括"]):
        tags.add("summary")
    if any(w in task_lower for w in ["python", "代码", "脚本", "编写", "数据分析", "生成", "统计", "计算"]):
        tags.add("code")
    return tags or {"web", "calc"}


def filter_tools(all_defs, needed_tags):
    allowed = set()
    for tag in needed_tags:
        allowed |= TOOL_PROFILES[tag]
    return [d for d in all_defs if d["function"]["name"] in allowed]


class Orchestrator:
    def __init__(self, call_llm_func, react_loop_func, tool_definitions=None):
        self.tasks = []
        self.results = []
        self.call_llm = call_llm_func
        self.react_loop = react_loop_func
        self.all_tools = tool_definitions or []
        self.shared_data = {}

    def plan(self, user_query):
        planner = Planner()
        tasks = planner.plan(user_query, self.call_llm)
        if not tasks:
            print("[Orchestrator] Planner 未返回任务，使用默认 fallback")
            return []
        self.tasks = tasks
        self._levels = planner.schedule(tasks)
        print(f"\n[Orchestrator] 分解为 {len(tasks)} 个子任务，{len(self._levels)} 个执行层级:")
        for t in tasks:
            deps = f"（依赖 #{','.join(t.depends_on)}）" if t.depends_on else "（无依赖）"
            print(f"  #{t.id}: {t.description} {deps}")
        print()
        print(planner.describe_schedule(self._levels))
        return tasks

    def run_worker(self, task, context=""):
        print(f"\n{'='*50}")
        print(f"[Worker] {task}")
        print(f"{'='*50}")
        if context:
            print(f"  [上下文] 收到 {context.count('---')} 个前置任务的结果")
            task_with_context = f"{task}\n\n参考信息：\n{context}"
        else:
            task_with_context = task
        needed = classify_tool_needs(task_with_context if context else task, self.call_llm)
        print(f"  需要工具类型: {', '.join(needed) if needed else '默认'}\n")
        old_tools = None
        if needed and self.all_tools:
            old_tools = self.all_tools[:]
            filtered = filter_tools(self.all_tools, needed)
            self.all_tools[:] = filtered
            print(f"  暴露 {len(filtered)}/{len(old_tools)} 个工具")

        result = self.react_loop(task_with_context)

        # 捕获本 Worker 的工具输出原始数据
        self._capture_worker_outputs(task, result)

    def _capture_worker_outputs(self, task, result):
        """从 react_loop 保存的轨迹步骤中提取工具调用输出"""
        try:
            from handwritten_react_agent.react_loop import last_trajectory_steps
            outputs = []
            for step in last_trajectory_steps:
                actions = step.get("actions", [])
                if "action" in step:
                    actions = [step["action"]] + actions
                for act in actions:
                    name = act.get("name", "")
                    obs = act.get("observation", "")
                    if name and obs and obs not in ("None", ""):
                        outputs.append(f"[{name}] {obs[:2000]}")
            self.shared_data[task.id] = {"answer": result or "", "tool_outputs": outputs}
        except Exception:
            self.shared_data[task.id] = {"answer": result or "", "tool_outputs": []}

    def _build_context(self, task: Task, completed_ids: set[str]) -> str:
        """为依赖任务构建上下文：前置任务的工具输出数据 + 结果摘要"""
        if not task.depends_on:
            return ""
        parts = ["以下是已完成任务的结果，供你参考："]
        for t in self.tasks:
            if t.id in task.depends_on:
                data = self.shared_data.get(t.id, {})
                parts.append(f"\n--- #{t.id}: {t.description} ---")
                # 优先传工具输出的原始数据（结构化数据）
                tool_outputs = data.get("tool_outputs", [])
                if tool_outputs:
                    parts.append("【实际输出数据】")
                    parts.extend(tool_outputs)
                # 再传 LLM 的总结
                if data.get("answer"):
                    parts.append(f"【任务总结】{data['answer']}")
        return "\n".join(parts)

    def synthesize(self):
        if len(self.results) == 1:
            final = self.results[0]
        elif not self.results:
            final = "没有可汇总的结果"
        else:
            parts = [f"-- 结果{i} --\n{r}" for i, r in enumerate(self.results, 1)]
            final = "\n\n".join(parts)
        print(f"\n{'='*50}")
        print("[汇总结果]")
        print(final)
        return final

    def execute(self, user_query, parallel=False):
        self.plan(user_query)
        self.results = []
        completed_ids = set()
        if not hasattr(self, '_levels') or not self._levels:
            print("[Orchestrator] 无任务可执行")
            return ""
        for level_idx, level in enumerate(self._levels):
            print(f"\n{'='*50}")
            print(f"[层级 {level_idx + 1}/{len(self._levels)}] {len(level)} 个任务")
            print(f"{'='*50}")
            if len(level) == 1:
                t = level[0]
                context = self._build_context(t, completed_ids)
                result = self.run_worker(t.description, context=context)
                t.result = result
                self.results.append(f"[#{t.id}] {t.description}\n{result}")
                completed_ids.add(t.id)
            else:
                self._execute_level_parallel(level, completed_ids)
        return self.synthesize()

    def _execute_level_parallel(self, level: list, completed_ids: set[str]):
        def run_one(task: Task) -> tuple:
            context = self._build_context(task, completed_ids)
            result = self.run_worker(task.description, context=context)
            return (task.id, result)
        with ThreadPoolExecutor(max_workers=len(level)) as ex:
            futures = {ex.submit(run_one, t): t for t in level}
            for f in as_completed(futures):
                t = futures[f]
                try:
                    tid, result = f.result()
                    t.result = result
                    self.results.append(f"[#{tid}] {t.description}\n{result}")
                    completed_ids.add(tid)
                    print(f"  [完成] #{tid}: {t.description[:50]}")
                except Exception as e:
                    print(f"  [失败] #{t.id}: {t.description[:50]}: {e}")

    def _execute_parallel(self):
        import copy
        worker_snapshots = []
        for task in self.tasks:
            needed = classify_tool_needs(task, self.call_llm)
            filtered = filter_tools(self.all_tools, needed) if needed else copy.deepcopy(self.all_tools)
            worker_snapshots.append((task, filtered))
            print(f"  [并行] {task} ({len(filtered)} 个工具)")
        def run_one(task, tools_snapshot):
            old = self.all_tools[:]
            self.all_tools[:] = tools_snapshot
            try:
                result = self.react_loop(task)
                return f"[任务] {task}\n{result}"
            finally:
                self.all_tools[:] = old
        with ThreadPoolExecutor(max_workers=len(self.tasks)) as ex:
            futures = {ex.submit(run_one, t, tl): t for t, tl in worker_snapshots}
            for f in as_completed(futures):
                task = futures[f]
                try:
                    r = f.result()
                    self.results.append(r)
                    print(f"  [完成] {task}")
                except Exception as e:
                    print(f"  [失败] {task}: {e}")
                    self.results.append(f"[任务] {task}\n失败: {e}")
