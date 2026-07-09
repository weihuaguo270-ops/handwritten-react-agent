
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
    task_lower = task.lower() if isinstance(task, str) else str(task).lower()
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

    def run_worker(self, task, context="", task_obj=None):
        print(f"\n{'='*50}")
        print(f"[Worker] {task}")
        if context:
            print(f"  [上下文] 收到 {context.count('---')} 个前置任务的结果")
            if "【前置数据】" in context:
                data_line = ""
                for line in context.split("\n"):
                    if line.startswith("data ="):
                        data_line = line.strip()
                        break
                if data_line:
                    task_with_context = (
                        f"对以下数据进行计算：{task}\n\n"
                        f"数据：\n{data_line}\n\n"
                        f"请用 Python 对这组数据执行计算，不要重新生成数据。"
                    )
                else:
                    task_with_context = f"{task}\n\n{context}"
            else:
                task_with_context = f"{task}\n\n{context}"
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
        if task_obj:
            self._capture_worker_outputs(task_obj, result)
        if old_tools:
            self.all_tools[:] = old_tools
        return result

    def _capture_worker_outputs(self, task, result):
        try:
            from handwritten_react_agent.react_loop import last_trajectory_steps
            outputs = []
            for step in last_trajectory_steps:
                obs = step.get("observation", "")
                act = step.get("action", {})
                name = act.get("name", "") if isinstance(act, dict) else ""
                if name and obs and obs not in ("None", ""):
                    outputs.append(f"[{name}] {obs[:2000]}")
                for multi_act in step.get("actions", []):
                    m_name = multi_act.get("name", "") if isinstance(multi_act, dict) else ""
                    m_obs = multi_act.get("observation", "")
                    if m_name and m_obs and m_obs not in ("None", ""):
                        outputs.append(f"[{m_name}] {m_obs[:2000]}")
            self.shared_data[task.id] = {"answer": result or "", "tool_outputs": outputs}
        except Exception:
            self.shared_data[task.id] = {"answer": result or "", "tool_outputs": []}

    def _build_context(self, task: Task, completed_ids: set[str]) -> str:
        if not task.depends_on:
            return ""
        parts = []
        for t in self.tasks:
            if t.id in task.depends_on:
                data = self.shared_data.get(t.id, {})
                tool_outputs = data.get("tool_outputs", [])
                numbers = None
                for out in tool_outputs:
                    import re
                    nums = re.findall(r'\[([\d.,\s]+)\]', out)
                    if nums:
                        numbers = nums[0]
                        break
                if numbers:
                    parts.append(
                        f"【前置数据】\n"
                        f"上一步生成的数据如下，请直接在代码中用这个变量：\n"
                        f"data = {numbers}\n"
                    )
                if tool_outputs:
                    parts.append(f"前置任务 #{t.id} 输出：")
                    parts.extend(tool_outputs[:2])
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
            if len(level) == 1:
                t = level[0]
                context = self._build_context(t, completed_ids)
                result = self.run_worker(t.description, context=context, task_obj=t)
                t.result = result
                self.results.append(f"[#{t.id}] {t.description}\n{result}")
                completed_ids.add(t.id)
            else:
                self._execute_level_parallel(level, completed_ids)
        return self.synthesize()

    def _execute_level_parallel(self, level: list, completed_ids: set[str]):
        def run_one(task: Task) -> tuple:
            context = self._build_context(task, completed_ids)
            result = self.run_worker(task.description, context=context, task_obj=task)
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
