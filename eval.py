"""
Agent Eval — 自动化评测框架
"""

import subprocess, sys, re, json

TEST_CASES = [
    {"question": "现在几点了？", "expected_tools": ["get_time"], "must_contain": ["2026"]},
    {"question": "计算 (23 + 45) * 2 等于多少", "expected_tools": ["calculator"], "must_contain": ["136"]},
    {"question": "先告诉我时间，再计算 100 / 7", "expected_tools": ["get_time", "calculator"], "must_contain": ["2026", "14."]},
    {"question": "搜索一下2026年AI Agent的最新发展", "expected_tools": ["web_search"], "must_contain": ["AI", "Agent"], "max_steps": 8},
    {"question": "先搜索AI Agent的维基百科词条，打开第一条结果，然后总结内容（用中文）",
     "expected_tools": ["web_search", "fetch_page"], "must_contain": ["Agent"]},
]

SCRIPT = r"D:\agent_learning\react_loop.py"

def run_one(question, timeout=60):
    r = subprocess.run([sys.executable, SCRIPT, question], capture_output=True, text=True, timeout=timeout)
    return r.stdout

def parse(out):
    tools = re.findall(r'\[调工具\] (\w+)\(', out)
    steps = [int(s) for s in re.findall(r'Step (\d+)/', out)]
    return {"tools": tools, "steps": max(steps) if steps else 0, "has_answer": "最终答案" in out}

results = []
print("=" * 50)
print("  Agent Eval 报告")
print("=" * 50)

for case in TEST_CASES:
    q = case["question"]
    print(f"\n  [{q[:30]}]")
    out = run_one(q)
    info = parse(out)

    passed = 0
    total = 3

    # Tool check
    for t in case["expected_tools"]:
        if t in info["tools"]:
            passed += 1
            print(f"    ✅ 工具: {t}")
            break
    else:
        print(f"    ❌ 工具: 未调用 {case['expected_tools']}")

    # Content check
    missing = [k for k in case["must_contain"] if k not in out]
    if not missing:
        passed += 1
        print(f"    ✅ 内容: 包含关键信息")
    else:
        print(f"    ❌ 内容: 缺 {missing}")

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
print(f"  总分: {sum(int(r.split('/')[0]) for r in results)}/{len(results)*3}")
print(f"{'=' * 50}")
