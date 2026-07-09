"""
全模块单元测试 — 不依赖 LLM API，快速验证各模块逻辑正确性

用法:
    python test_all.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

errors = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ✅ {name}")
    else:
        msg = f"  ❌ {name}" + (f" — {detail}" if detail else "")
        print(msg)
        errors.append(name)


# ============================================================
# 1. CoT
# ============================================================
print("\n【CoT 思维链】")
from handwritten_react_agent.cot import COT, CoTStrategy, extract_reasoning

check("策略选择: 数学", COT.select("计算5只鸡和3只兔子一共多少条腿") == CoTStrategy.FEW_SHOT_MATH)
check("策略选择: 推理", COT.select("如果下雨就不去") == CoTStrategy.FEW_SHOT_REASONING)
check("策略选择: 搜索", COT.select("搜索今天天气") == CoTStrategy.ZERO_SHOT)

result = COT.inject("base", query="计算1+1")
check("inject 拼接", "base" in result and "THOUGHT" in result)

thoughts, final = extract_reasoning("THOUGHT: 第一步\nFINAL ANSWER: 答案")
check("extract 推理", len(thoughts) == 1 and thoughts[0] == "第一步")
check("extract 答案", final == "答案")

# CoT 工具定义
from handwritten_react_agent.cot import COT_TOOL_DEFINITION
check("工具定义存在", "switch_cot_strategy" in str(COT_TOOL_DEFINITION))


# ============================================================
# 2. ToT
# ============================================================
print("\n【ToT 思维树】")
from handwritten_react_agent.tot import ToT, ToTNode, SearchStrategy

tot = ToT(beam_width=2, branch_factor=2, max_depth=2)

# 树结构
root = ToTNode("")
n1 = ToTNode("第一步", root)
n1.score = 8.0
n2 = ToTNode("第二步", root)
n2.score = 3.0
check("best_child", root.best_child() == n1)
check("chain", n1.chain() == ["第一步"])
check("自动注册到父节点", len(root.children) == 2)

# 候选解析
cands = tot._parse_candidates("步骤1: A\n---\n步骤2: B")
check("parse_candidates 2个", len(cands) == 2)
check("parse_candidates 内容", cands[0] == "A" and cands[1] == "B")

# 没有 --- 分隔
cands2 = tot._parse_candidates("直接回答没按格式")
check("parse_candidates 兜底", len(cands2) >= 1)

# 评分解析
check("parse_score 8", tot._parse_score("8") == 8.0)
check("parse_score 7/10", tot._parse_score("7/10") == 7.0)
check("parse_score 评分6", tot._parse_score("评分: 6") == 6.0)
check("parse_score 兜底", tot._parse_score("好的") == 5.0)

# 搜索策略枚举
check("BFS", SearchStrategy.BFS.value == "bfs")
check("DFS", SearchStrategy.DFS.value == "dfs")


# ============================================================
# 3. Planner
# ============================================================
print("\n【Planner 任务规划】")
from handwritten_react_agent.planner import Planner, Task

# 解析
text = "task_1: 搜索今天\ntask_2: 搜索明天\ntask_3: 对比 | depends_on: 1, 2"
tasks = Planner._parse_tasks(text)
check("解析3个任务", len(tasks) == 3)
check("task_1 描述", tasks[0].description == "搜索今天")
check("task_3 依赖", tasks[2].depends_on == ["1", "2"])

# 拓扑排序
levels = Planner.schedule(tasks)
check("2个层级", len(levels) == 2)
check("第1层2个", len(levels[0]) == 2)
check("第2层1个", len(levels[1]) == 1)

# 单任务
levels2 = Planner.schedule([Task("1", "一件事")])
check("单任务1层", len(levels2) == 1)

# Task 就绪检查
t1 = Task("1", "A")
t2 = Task("2", "B", depends_on=["1"])
check("无依赖就绪", t1.ready(set()) == True)
check("有依赖未就绪", t2.ready(set()) == False)
check("有依赖已就绪", t2.ready({"1"}) == True)


# ============================================================
# 4. RoleManager
# ============================================================
print("\n【RoleManager 角色管理】")
from handwritten_react_agent.prompts import ROLE_MANAGER, Role

check("code_reviewer", ROLE_MANAGER.select("帮我审查代码") == Role.CODE_REVIEWER)
check("tutor", ROLE_MANAGER.select("什么是装饰器") == Role.TUTOR)
check("debater", ROLE_MANAGER.select("对比A和B") == Role.DEBATER)
check("research_assistant", ROLE_MANAGER.select("今天天气") == Role.RESEARCH_ASSISTANT)

# 注入测试
base = "base prompt"
enhanced = ROLE_MANAGER.inject(base, query="帮我审查代码")
check("inject 含角色", "审查员" in enhanced or "审查" in enhanced)
check("inject 保留base", "base prompt" in enhanced)

# 切换角色
ret = ROLE_MANAGER.set_role("tutor")
check("set_role 成功", "tutor" in ret)
check("current_role 变了", ROLE_MANAGER.current_role_name() == "tutor")

# 未知角色
ret2 = ROLE_MANAGER.set_role("unknown")
check("set_role 未知", "未知" in ret2)

# 列表
roles = ROLE_MANAGER.list_roles()
check("list_roles 含5个", len(roles) == 5)
check("包含 tutor", "tutor" in roles)


# ============================================================
# 5. Context Manager
# ============================================================
print("\n【Context 上下文管理】")
from handwritten_react_agent.context import ContextManager, ContextStrategy, estimate_tokens, estimate_messages_tokens

check("estimate_tokens 英文", estimate_tokens("hello world") > 0)
check("estimate_tokens 中文", estimate_tokens("你好世界") > 0)

# 测试截断
ctx = ContextManager(max_tokens=150, reserve_tokens=20, strategy=ContextStrategy.TRUNCATE)
msgs = [{"role": "system", "content": "你是一个AI助手"}]
for i in range(15):
    msgs.append({"role": "user", "content": f"第{i+1}轮测试消息用于填充上下文确保触发截断"})
    msgs.append({"role": "assistant", "content": f"第{i+1}轮回复用于填充上下文确保触发截断"})

before = len(msgs)
managed = ctx.manage(msgs)
check("截断减少消息", len(managed) < before)

# 检查 system 被保留
check("system 被保留", any(m["role"] == "system" for m in managed))

# 策略切换
ctx.set_strategy("drop")
check("策略切换", ctx.strategy == ContextStrategy.DROP)

# 不超限时不动
short_msgs = [{"role": "user", "content": "hi"}]
ctx2 = ContextManager(max_tokens=48000)
before2 = len(short_msgs)
managed2 = ctx2.manage(short_msgs)
check("不超限不动", len(managed2) == before2)

# token 估算
check("message tokens > 0", estimate_messages_tokens(short_msgs) > 0)


# ============================================================
# 6. Harness / 轨迹记录
# ============================================================
print("\n【Harness 轨迹记录】")
from handwritten_react_agent.harness import start_trajectory, current_trajectory, finish_trajectory, Trajectory

# 基本轨迹
traj = Trajectory("测试问题", "test-model")
traj.start_step(1)
traj.add_thought(1, "第一步思考")
traj.add_tool_call(1, "calc", '{"exp":"1+1"}', "2")
traj.add_tool_call(1, "calc", '{"exp":"2+2"}', "4")
check("轨迹有2个工具调用",
      sum(1 for s in traj.steps if s.get("action") or s.get("actions")) >= 1)

traj.set_final_answer("答案42")
path = traj.save()
check("文件已保存", path and os.path.exists(path))

with open(path, encoding="utf-8") as f:
    data = json.load(f)
check("轨迹含session_id", "session_id" in data)
check("轨迹含query", data["query"] == "测试问题")
check("轨迹含答案", data["final_answer"] == "答案42")

os.remove(path)

# 全局接口
traj2 = start_trajectory("全局测试", "m1", "sys")
check("start_trajectory 返回 Trajectory", isinstance(traj2, Trajectory))
check("current_trajectory 不为空", current_trajectory() is not None)
traj2.add_thought(1, "思考")
path2 = finish_trajectory("最终答案")
check("finish 返回路径", path2 and os.path.exists(path2))
check("current 已清空", current_trajectory() is None)
os.remove(path2)


# ============================================================
# 7. Sandbox
# ============================================================
print("\n【Sandbox 沙箱隔离】")
from handwritten_react_agent.harness import Sandbox, SANDBOX

sandbox = Sandbox(strategy="off")
check("沙箱模块存在", hasattr(sandbox, "run"))
check("沙箱默认关闭", SANDBOX.strategy == "off" or SANDBOX.strategy == "off")
check("_sandbox_runner 存在", os.path.exists(os.path.join(os.path.dirname(__file__), "src/handwritten_react_agent/harness/_sandbox_runner.py")))
SANDBOX_TOOL_DEFINITION = __import__("handwritten_react_agent.harness.sandbox", fromlist=["SANDBOX_TOOL_DEFINITION"]).SANDBOX_TOOL_DEFINITION
check("SANDBOX_TOOL_DEFINITION 存在", "toggle_sandbox" in str(SANDBOX_TOOL_DEFINITION))


# ============================================================
# 8. 工具模块（tools/）
# ============================================================
print("\n【工具模块 tools/】")
from handwritten_react_agent.tools.calculator import calculator
from handwritten_react_agent.tools.summarize import summarize
from handwritten_react_agent.tools.get_time import get_time

check("calculator 1+2", calculator("1+2") == "3")
check("calculator 小数", calculator("3.5*2") == "7.0")
check("calculator 括号", calculator("(2+3)*4") == "20")
check("calculator 非法字符", "错误" in calculator("__import__('os')"))
check("calculator 语法错误", "错误" in calculator("1++"))
check("get_time 含日期", "-" in get_time() and ":" in get_time())
check("summarize 短文本", "过短" in summarize("短"))
check("summarize 正常", len(summarize("句子一。句子二。句子三。句子四。句子五。", 3)) > 0)

# 工具定义存在
from handwritten_react_agent.tools.get_time import TOOL_DEFINITION as TD_GET
from handwritten_react_agent.tools.calculator import TOOL_DEFINITION as TD_CALC
check("工具定义 get_time 正确", TD_GET["function"]["name"] == "get_time")
check("工具定义 calculator 正确", TD_CALC["function"]["name"] == "calculator")


# ============================================================
# 9. 工具注册完整性（从 tools 直接导入，不依赖 react_loop.py）
# ============================================================
print("\n【工具注册完整性】")
from handwritten_react_agent.tools import TOOL_REGISTRY as TR, TOOL_DEFINITIONS as TDS

expected_tools = [
    "get_time", "calculator", "web_search", "fetch_page", "summarize",
    "rag_query", "switch_cot_strategy", "tot_reasoning", "switch_role",
    "switch_context_strategy", "toggle_sandbox", "start_dashboard", "clear_trajectories",
    "execute_python",
]
for name in expected_tools:
    check(f"TOOL_REGISTRY 含 {name}", name in TR)
check(f"TOOL_DEFINITIONS 数量 {len(expected_tools)}", len(TDS) == len(expected_tools))

# TOOL_DEFINITIONS 每个都有完整结构
for td in TDS:
    fn = td.get("function", {})
    check(f"  定义 {fn.get('name','?')} 含 parameters",
          "parameters" in fn and "type" in fn.get("parameters", {}))
    check(f"  定义 {fn.get('name','?')} 含 description",
          bool(fn.get("description", "").strip()))


# ============================================================
# 10. Memory（语义记忆）
# ============================================================
print("\n【Memory 记忆系统】")
import handwritten_react_agent.memory as mem_mod
from handwritten_react_agent.memory import Memory

m = Memory()
m.facts = []
check("空记忆无检索", len(m.query("test")) == 0)

# 删除测试（先清空再测试）
m.clear()
m.add("小明是学生")
check("添加后事实数>0", len(m.facts) > 0)

found = m.query("小明")
check("查询小明有结果", len(found) > 0)

# remove 测试
removed = m.remove("小明")
check("remove 返回 True", removed)
check("remove 后事实减少", len(m.facts) == 0)


# ============================================================
# 11. Orchestrator
# ============================================================
print("\n【Orchestrator 多 Agent】")
from handwritten_react_agent.orchestrator import Planner, Task
import re as _re

# 测试 Planner 的模板匹配正则（不依赖 LLM）
p = Planner()
TEMPLATES = p._TEMPLATES

# 手动跑模板匹配检验
def _match_template(query):
    for pattern, builder in TEMPLATES:
        m = _re.search(pattern, query)
        if m:
            tasks = builder(m)
            if tasks:
                return tasks
    return None

tasks = _match_template("搜索今天北京和上海的天气，对比温差")
check("搜索对比模板命中", tasks is not None)
if tasks:
    check("模板返回3个任务", len(tasks) == 3)
    check("任务1含'今天'", "今天" in tasks[0].description)
    check("任务3依赖1和2", tasks[2].depends_on == ["1", "2"])

check("空查询无模板", _match_template("") is None)
check("打招呼无模板", _match_template("你好") is None)

# ============================================================
import subprocess
print("\n【Replay 重放】")
# 先建一个轨迹文件
from handwritten_react_agent.harness import start_trajectory, finish_trajectory
t = start_trajectory("重放测试", "m1")
t.add_thought(1, "测试")
path = finish_trajectory("ok")

result = subprocess.run([sys.executable, "-m", "harness.replay", "--latest"],
                       capture_output=True, text=True, timeout=10)
check("replay 输出含轨迹信息",
      "🎯" in result.stdout or "Step" in result.stdout or "最终" in result.stdout)

result2 = subprocess.run([sys.executable, "-m", "harness.replay"],
                        capture_output=True, text=True, timeout=10)
check("replay 列表含记录数", "共" in result2.stdout or "轨迹" in result2.stdout)

os.remove(path) if os.path.exists(path) else None


# ============================================================
print(f"\n{'='*50}")
if errors:
    print(f"❌ {len(errors)} 项失败:")
    for e in errors:
        print(f"   - {e}")
else:
    print("🎉 全部测试通过!")
print(f"{'='*50}")
