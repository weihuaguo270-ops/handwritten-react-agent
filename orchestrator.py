
"""
Orchestrator — 独立的多 Agent 协作模块
支持 Worker 隔离：每个 Worker 有独立的工具集
"""
import json, urllib.request, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


TOOL_PROFILES = {
    "time": {"get_current_time", "convert_time"},
    "file": {"read_text_file", "write_file", "edit_file", "create_directory",
             "list_directory", "directory_tree", "move_file", "search_files",
             "get_file_info", "list_allowed_directories"},
    "web": {"web_search", "fetch_page"},
    "calc": {"calculator"},
    "summary": {"summarize"},
}

PROFILE_HINTS = {
    "time": "查询时间、时区转换",
    "file": "文件读写、目录管理、文件信息",
    "web": "搜索互联网、读取网页",
    "calc": "数学计算",
    "summary": "文本摘要",
}


def classify_tool_needs(task, call_llm):
    prompt = (
        f"判断以下任务需要使用哪种工具。只输出分类名，多个用逗号分隔。\n\n"
        f"可选分类:\n"
        + "\n".join(f"- {k}: {v}" for k, v in PROFILE_HINTS.items()) +
        f"\n\n任务: {task}\n分类:"
    )
    msg = call_llm([
        {"role": "system", "content": "你是一个工具分类助手，只输出分类名。"},
        {"role": "user", "content": prompt},
    ])
    text = (msg.get("content", "") or "").strip().lower()
    tags = set()
    for t in text.replace("\uff0c", ",").split(","):
        t = t.strip()
        if t in TOOL_PROFILES:
            tags.add(t)
    return tags or {"web"}


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

    def plan(self, user_query):
        prompt = (
            "将以下请求拆成独立子任务。要求：\n"
            "- 每个子任务一句话，只做一件事\n"
            "- 每行一个子任务，不要编号\n"
            "- 不要解释，直接输出任务\n\n"
            "例子：\n"
            "请求: 现在纽约几点？同时看看mcp_client.py大小\n"
            "输出:\n"
            "查询纽约的当前时间\n"
            "查看mcp_client.py的文件大小\n\n"
            f"请求: {user_query}\n"
            "输出:"
        )
        msg = self.call_llm([
            {"role": "system", "content": "你是一个任务分解助手。"},
            {"role": "user", "content": prompt},
        ])
        self.tasks = [t.strip() for t in (msg.get("content", "") or "").split("\n") if t.strip()]
        print(f"\n[Orchestrator] 拆分为 {len(self.tasks)} 个子任务:")
        for i, t in enumerate(self.tasks, 1):
            print(f"  {i}. {t}")
        return self.tasks

    def run_worker(self, task):
        print(f"\n{'='*50}")
        print(f"[Worker] {task}")
        print(f"{'='*50}")

        needed = classify_tool_needs(task, self.call_llm)
        print(f"  需要工具: {', '.join(needed) if needed else '默认'}")

        old_tools = None
        if needed and self.all_tools:
            old_tools = self.all_tools[:]
            filtered = filter_tools(self.all_tools, needed)
            self.all_tools[:] = filtered
            print(f"  暴露 {len(filtered)}/{len(old_tools)} 个工具")

        result = self.react_loop(task)

        if old_tools:
            self.all_tools[:] = old_tools
        return result

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

    def execute(self, user_query):
        self.plan(user_query)
        self.results = []
        for task in self.tasks:
            result = self.run_worker(task)
            self.results.append(f"[任务] {task}\n{result}")
        return self.synthesize()
