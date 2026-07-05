"""
Agent Eval — 自动化评测框架
用法：
    python eval.py                          # 使用默认 provider
    python eval.py --provider openai         # 切换到 OpenAI
    python eval.py --provider ollama         # 使用本地 Ollama
"""
import subprocess, sys, re, json, os

TEST_CASES = [
    # === 本地工具 / MCP 时间 ===
    # 注：uvx 启动后会替换 get_time 为 MCP 的 get_current_time
    {"question": "现在几点了？", "expected_tools": ["get_current_time", "get_time"],
     "must_contain": ["2026"]},
    {"question": "计算 (23 + 45) * 2 等于多少", "expected_tools": ["calculator"],
     "must_contain": ["136"]},
    {"question": "先告诉我时间，再计算 100 / 7",
     "expected_tools": ["get_current_time", "get_time", "calculator"],
     "must_contain": ["2026", "14."]},
    # === 搜索（AnySearch 可能较慢） ===
    {"question": "搜索一下2026年AI Agent的最新发展",
     "expected_tools": ["web_search"], "must_contain": ["AI", "Agent"],
     "max_steps": 8, "timeout": 120},
    {"question": "先搜索AI Agent的维基百科词条，打开第一条结果，然后总结内容（用中文）",
     "expected_tools": ["web_search", "fetch_page", "summarize"],
     "must_contain": ["Agent"], "timeout": 120},
    # === RAG（Agent 可能优先用 MCP 文件系统工具而非 rag_query） ===
    {"question": "这个项目里 react_loop.py 是做什么的？",
     "expected_tools": ["rag_query", "search_files", "read_text_file"],
     "must_contain": ["ReAct"]},
    {"question": "项目的 RAG 模块在哪个文件？",
     "expected_tools": [
         "rag_query",
         "search_files", "read_text_file", "list_directory",
         "directory_tree", "get_file_info", "list_allowed_directories"
     ],
     "must_contain": ["rag.py"]},
    # === MCP 时区转换 ===
    {"question": "现在东京时间是多少？",
     "expected_tools": ["get_current_time", "convert_time", "get_time", "web_search"],
     "must_contain": ["2026"], "max_steps": 6},
    # === 新增：角色自动选择 ===
    {"question": "帮我审查这段代码有什么问题：def add(a,b): return a+b",
     "must_contain": ["code_reviewer"],
     "must_contain_any": ["类型", "PEP", "类型注解", "隐患", "建议"],
     "max_steps": 4, "timeout": 60},
    {"question": "对比一下Python和JavaScript的优缺点",
     "must_contain": ["debater"],
     "must_contain_any": ["优点", "缺点", "优势", "劣势"],
     "max_steps": 4, "timeout": 60},
    # === 新增：多步工具调用 ===
    {"question": "计算 (123+456)*2 和 789/3 分别是多少",
     "expected_tools": ["calculator"],
     "must_contain": ["1158", "263"],
     "max_steps": 4, "timeout": 60},
    # === 新增：Orchestrator 多任务 ===
    {"question": "同时帮我查一下今天的时间，并计算 50*30",
     "tag": "orchestrator",
     "expected_tools": ["get_time", "get_current_time", "calculator"],
     "must_contain_any": ["[Orchestrator]", "层级", "子任务", "#1", "#2"],
     "max_steps": 8, "timeout": 90},
]

SCRIPT = r"D:\\agent_learning\\repo\\react_loop.py"

# 解析 --provider 参数
_EVAL_PROVIDER = None
if "--provider" in sys.argv:
    idx = sys.argv.index("--provider")
    if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("--"):
        _EVAL_PROVIDER = sys.argv[idx + 1]
        # 从 sys.argv 移除，避免传给子进程
        sys.argv = sys.argv[:idx] + sys.argv[idx + 2:]

# 构建子进程环境
_EVAL_ENV = {**os.environ}
if _EVAL_PROVIDER:
    _EVAL_ENV["LLM_PROVIDER"] = _EVAL_PROVIDER


def run_one(question, timeout=60):
    r = subprocess.run([sys.executable, SCRIPT, question],
                       capture_output=True, text=True, timeout=timeout,
                       env=_EVAL_ENV)
    return r.stdout


def parse(out):
    tools = re.findall(r'\[调工具\] (\w+)\(', out)
    steps = [int(s) for s in re.findall(r'Step (\d+)/', out)]
    return {
        "tools": tools,
        "steps": max(steps) if steps else 0,
        "has_answer": "最终答案" in out,
    }


results = []
print("=" * 50)
print("  Agent Eval 报告")
if _EVAL_PROVIDER:
    print(f"  Provider: {_EVAL_PROVIDER}")
else:
    print(f"  Provider: (默认，由 LLM_PROVIDER 或 llm_config.json 决定)")
print("=" * 50)

for i, case in enumerate(TEST_CASES, 1):
    q = case["question"]
    print(f"\n  [{i}/{len(TEST_CASES)}] {q[:50]}...")
    to = case.get("timeout", 60)
    try:
        out = run_one(q, timeout=to)
    except subprocess.TimeoutExpired:
        print(f"    ⏰ 超时 ({to}s)，跳过")
        results.append("0/3")
        continue

    info = parse(out)

    passed = 0
    total = 3

    # Tool check: 任意一个预期工具被调用即通过（OR）
    expected = case["expected_tools"]
    if any(t in info["tools"] for t in expected):
        passed += 1
        print(f"    ✅ 工具: {info['tools']}")
    else:
        print(f"    ❌ 工具: 预期 {expected}，实际 {info['tools']}")

    # Content check
    missing_k = [k for k in case.get("must_contain", []) if k not in out]
    if not missing_k:
        passed += 1
        print(f"    ✅ 内容: 含所有关键词")
    else:
        # 如果有 must_contain_any，检查是否命中至少一个
        any_k = case.get("must_contain_any", [])
        if any_k and any(k in out for k in any_k):
            passed += 1
            hit = [k for k in any_k if k in out]
            print(f"    ✅ 内容: 命中 {hit}")
        else:
            print(f"    ❌ 内容: 缺 {missing_k}" + (f"，预期至少含 {any_k}" if any_k else ""))

    # Steps check
    max_s = case.get("max_steps", 10)
    if info["steps"] <= max_s:
        passed += 1
        print(f"    ✅ 步数: {info['steps']}/{max_s}")
    else:
        print(f"    ❌ 步数: {info['steps']}/{max_s}")

    results.append(f"{passed}/{total}")

print(f"\n{'=' * 50}")
for i, r in enumerate(results):
    print(f"  测试{i+1}: {r}")
total_score = sum(int(r.split('/')[0]) for r in results)
max_score = len(results) * 3
print(f"  总分: {total_score}/{max_score}")
print(f"{'=' * 50}")
