"""场景式测试 — Eval Loop 的迭代行为 + 复杂输入评分

测试目标：
  1. 在复杂输入下，evaluate 如何评判生成结果的质量
  2. 在输出结果前，进行了多少次循环 evaluate

覆盖场景：
  A. 一次性通过（迭代 1 次）
  B. 一次修正后通过（迭代 2 次）
  C. 三次修正才通过（迭代 3 次）
  D. 达到最大迭代仍未通过（迭代 3 次，最终失败）
  E. 检测到震荡自动停止（改进 < 阈值）
  F. 功能测试类任务（不走 loop）
  G. 复杂多步轨迹评分验证
  H. 错误传播根因定位验证
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

from core.trajectory_parser import parse_trajectory, StepsDAG
from core.process_reward import (
    ProcessRewardScorer, ProcessRewardReport,
    analyze_error_propagation, pack_revision_instructions,
)
from handwritten_react_agent.intent.classifier import IntentClassifier, TaskType
from loop.eval_loop import EvalLoopEngine, EvalLoopConfig


# ══════════════════════════════════════════════
#  辅助函数
# ══════════════════════════════════════════════


def _make_trajectory(query: str, steps: list[dict], final: str = "") -> dict:
    """构造标准轨迹"""
    return {
        "session_id": "test_scenario",
        "query": query,
        "steps": steps,
        "total_steps": len(steps),
        "final_answer": final or steps[-1].get("content", ""),
    }


def _simple_traj(query: str) -> dict:
    """简单的 3 步轨迹"""
    return _make_trajectory(query, [
        {"step_index": 0, "type": "thought", "content": f"分析{query}的步骤"},
        {"step_index": 1, "type": "action",
         "action": {"name": "web_search", "args": {"query": query}},
         "content": f"搜索{query}的结果"},
        {"step_index": 2, "type": "final", "content": f"关于{query}的完整报告"},
    ], final=f"关于{query}的完整报告")


def _complex_traj() -> dict:
    """复杂多步轨迹（6 步，含多种工具调用）"""
    return _make_trajectory(
        query="帮我分析一下2026年AI Agent行业，对比三个主流框架，并给出推荐",
        steps=[
            {"step_index": 0, "type": "thought",
             "content": "用户要求做行业分析+框架对比，需要先收集信息再分析"},
            {"step_index": 1, "type": "action",
             "action": {"name": "web_search", "args": {"query": "2026 AI Agent 行业报告"}},
             "content": "搜索行业报告"},
            {"step_index": 2, "type": "observation",
             "content": "2026年AI Agent市场规模78亿美元，预计2030年达520亿...",
             "observation": "2026年市场规模78亿美元，年增长率40%+"},
            {"step_index": 3, "type": "action",
             "action": {"name": "web_search", "args": {"query": "LangGraph vs CrewAI vs AutoGen 对比"}},
             "content": "搜索框架对比"},
            {"step_index": 4, "type": "observation",
             "content": "LangGraph适合复杂流程，CrewAI适合快速原型，AutoGen适合多Agent...",
             "observation": "对比结果：各有优势"},
            {"step_index": 5, "type": "final",
             "content": "## 2026 AI Agent 行业分析\n\n市场规模...\n\n## 框架对比\n\n推荐LangGraph..."},
        ],
        final="## 2026 AI Agent 行业分析\n\n市场规模...\n\n## 框架对比\n\n推荐LangGraph...",
    )


def _make_judge_fn(scores_by_step: dict[int, float]):
    """用分数表构造 mock judge

    scores_by_step: {step_index: score}
    如 {0: 4.5, 1: 4.0, 2: 2.5} → Step 2 低分
    """
    call_count = {"n": 0}

    def judge_fn(prompt: str) -> dict:
        call_count["n"] += 1
        # 从 prompt 中提取步骤索引
        for line in prompt.split("\n"):
            if "Step " in line and "/" in line:
                # "当前步骤（Step 2/5）"
                try:
                    parts = line.split("Step ")[1]
                    idx = int(parts.split("/")[0].split("）")[0])
                except (ValueError, IndexError):
                    idx = 0
                break
        else:
            idx = 0

        score = scores_by_step.get(idx, 4.0)
        return {
            "role_understanding": f"模拟评分 Step {idx}",
            "rubrics": [
                {"dimension": "quality", "criteria": "步骤质量",
                 "score": score, "reason": "模拟评分"},
            ],
            "step_score": score,
            "needs_revision": score < 3.5,
        }

    return judge_fn, call_count


def _make_agent_fn(trajectory: dict, quality_level: str = "good"):
    """用预设轨迹构造 mock agent

    quality_level:
      "good"    → 一次通过（所有步骤 >= 4.0）
      "poor"    → 第一次不行，改进后通过
      "bad"     → 一直不行
      "oscillate" → 分数震荡
    """
    iteration_count = {"n": 0}

    def agent_fn(query: str) -> dict:
        iteration_count["n"] += 1
        # 生成一个副本，避免互相影响
        import copy
        return {"output": trajectory.get("final_answer", ""), "trajectory": copy.deepcopy(trajectory)}

    return agent_fn, iteration_count


def _make_judge_for_scenario(scenario: str):
    """为不同场景构造对应的 judge

    返回 (judge_fn, call_count, description)
    """
    call_count = {"n": 0}

    if scenario == "good":
        # A: 所有步骤高分 → 一次性通过
        def fn(prompt):
            call_count["n"] += 1
            return {"role_understanding": "高质量", "rubrics": [
                {"dimension": "quality", "criteria": "质量", "score": 4.5, "reason": "好"},
            ], "step_score": 4.5, "needs_revision": False}
        return fn, call_count, "所有步骤高分，期望迭代 1 次"

    elif scenario == "fix_once":
        # B: 第一次 low score，第二次 high score
        def fn(prompt):
            call_count["n"] += 1
            # 第一次调用的所有步骤都低分，第二次之后高分
            if call_count["n"] <= 5:  # 第一次循环（3 步 + 首步可能多一次）
                score = 2.5
            else:
                score = 4.5
            return {"role_understanding": "修正后", "rubrics": [
                {"dimension": "quality", "criteria": "质量", "score": score, "reason": "模拟"},
            ], "step_score": score, "needs_revision": score < 3.5}
        return fn, call_count, "第一次低分，修正后高分，期望迭代 2 次"

    elif scenario == "fix_twice":
        # C: 前 8 次调用低分，之后高分
        def fn(prompt):
            call_count["n"] += 1
            score = 2.5 if call_count["n"] <= 8 else 4.5
            return {"role_understanding": "多轮修正", "rubrics": [
                {"dimension": "quality", "criteria": "质量", "score": score, "reason": "模拟"},
            ], "step_score": score, "needs_revision": score < 3.5}
        return fn, call_count, "前 8 次低分，之后高分，期望迭代 3 次后通过"

    elif scenario == "never_pass":
        # D: 一直低分
        def fn(prompt):
            call_count["n"] += 1
            return {"role_understanding": "始终低质量", "rubrics": [
                {"dimension": "quality", "criteria": "质量", "score": 2.0, "reason": "一直不行"},
            ], "step_score": 2.0, "needs_revision": True}
        return fn, call_count, "始终低分，期望迭代 3 次后失败"

    elif scenario == "oscillate":
        # E: 分数震荡（交替极接近的值，改进 < 0.1）
        def fn(prompt):
            call_count["n"] += 1
            # 交替 2.0 和 2.05，改进仅 0.05 < 0.1
            score = 2.0 if call_count["n"] % 2 == 1 else 2.05
            return {"role_understanding": "震荡", "rubrics": [
                {"dimension": "quality", "criteria": "质量", "score": score, "reason": "模拟"},
            ], "step_score": score, "needs_revision": True}
        return fn, call_count, "分数震荡（改进<0.1），期望约 2 次迭代后提前停止"

    else:
        raise ValueError(f"Unknown scenario: {scenario}")


# ══════════════════════════════════════════════
# 测试执行
# ══════════════════════════════════════════════


def run_scenario(
    name: str,
    query: str,
    trajectory: dict,
    scenario_type: str,
    max_iterations: int = 3,
) -> dict:
    """运行单个场景测试"""
    print(f"\n{'=' * 60}")
    print(f"  场景: {name}")
    print(f"  {'=' * 60}")

    # 意图分类检查
    classifier = IntentClassifier()
    task_type = classifier.classify(query)
    print(f"  [意图] {task_type}")

    # 构造 judge
    judge_fn, judge_count, desc = _make_judge_for_scenario(scenario_type)
    print(f"  [预期] {desc}")

    # 构造 agent
    agent_fn, agent_count = _make_agent_fn(trajectory)

    # 运行 Eval Loop
    config = EvalLoopConfig(
        max_iterations=max_iterations,
        min_improvement=0.1,
        verbose=False,
    )
    engine = EvalLoopEngine(
        agent_fn=agent_fn,
        judge_fn=judge_fn,
        config=config,
    )
    result = engine.execute(query)

    # 解析结果
    print(f"  [结果] 迭代: {result.iterations} 次")
    print(f"  [结果] 总分: {result.report.overall_score:.3f}")
    print(f"  [结果] 通过: {'✅' if result.passed else '❌'}")
    print(f"  [结果] Judge 调用: {judge_count['n']} 次")
    print(f"  [结果] Agent 调用: {agent_count['n']} 次")

    if result.healing_log:
        print(f"  [自愈过程]")
        for entry in result.healing_log:
            icon = "✅" if not entry["needs_revision"] else "🔄"
            print(f"    {icon} 迭代 {entry['iteration']}: "
                  f"得分 {entry['overall_score']:.3f}, "
                  f"失败 {entry['num_failed_steps']} 步")

    return {
        "name": name,
        "iterations": result.iterations,
        "passed": result.passed,
        "overall_score": result.report.overall_score,
        "judge_calls": judge_count["n"],
        "agent_calls": agent_count["n"],
        "task_type": task_type,
        "healing_log": result.healing_log,
    }


def test_scenario_g():
    """场景 G: 复杂多步轨迹评分验证"""
    print(f"\n{'=' * 60}")
    print(f"  场景 G: 复杂多步轨迹评分验证")
    print(f"  {'=' * 60}")

    traj = _complex_traj()
    dag = parse_trajectory(traj)

    print(f"  轨迹步骤数: {dag.num_steps}")
    for n in dag.nodes:
        tool = f" ({n.tool_name})" if n.tool_name else ""
        print(f"    Step {n.step_index} [{n.step_type}]{tool}")

    # 用不同分数的 judge 验证
    scores = {0: 4.0, 1: 4.5, 2: 4.0, 3: 3.0, 4: 4.0, 5: 2.5}
    judge_fn, count = _make_judge_fn(scores)
    scorer = ProcessRewardScorer(judge_fn=judge_fn)
    report = scorer.score_trajectory(dag)

    print(f"\n  [评分结果] 总分: {report.overall_score:.3f}")
    print(f"  [评分结果] 失败步骤: {report.num_failed_steps}/{report.num_scored}")

    for s in report.per_step:
        icon = "✅" if not s.needs_revision else "❌"
        print(f"    {icon} Step {s.step_index}: {s.step_score:.2f} "
              f"(预期: {scores.get(s.step_index, 0):.1f})")
        assert s.step_score == scores.get(s.step_index, 0), \
            f"Step {s.step_index} 分数不匹配: {s.step_score} != {scores.get(s.step_index, 0)}"

    print(f"  ✅ 复杂轨迹评分正确")


def test_scenario_h():
    """场景 H: 错误传播根因定位验证"""
    print(f"\n{'=' * 60}")
    print(f"  场景 H: 错误传播根因定位")
    print(f"  {'=' * 60}")

    traj = _complex_traj()
    dag = parse_trajectory(traj)

    # 模拟：Step 1（搜索行业报告）得分低 → Step 2（观察结果）受影响 → Step 5（最终答案）也受损
    scores = {0: 4.0, 1: 1.5, 2: 2.5, 3: 4.0, 4: 4.0, 5: 2.0}
    judge_fn, _ = _make_judge_fn(scores)
    scorer = ProcessRewardScorer(judge_fn=judge_fn)
    report = scorer.score_trajectory(dag)

    # 错误传播分析
    error_analysis = analyze_error_propagation(report, dag)
    print(f"  错误源头: {error_analysis['error_sources']}")
    print(f"  直接影响步骤: {error_analysis['impact_summary']['directly_affected']}")
    print(f"  级联影响步骤: {error_analysis['impact_summary']['cascading_affected']}")

    # 验证根因是 Step 1（搜索失败）
    assert 1 in error_analysis["error_sources"], "Step 1 应该是根因"
    print(f"  ✅ 根因定位正确: Step {error_analysis['error_sources']}")

    # 修正指令
    fix = pack_revision_instructions(report, dag)
    print(f"  修正指令长度: {len(fix)} 字符")
    assert "Step 1" in fix or "Step 5" in fix
    print(f"  ✅ 修正指令正确生成")


# ══════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Eval Engine 场景式测试")
    print("  =================================")
    print("  重点: 循环次数 + 复杂输入评分")
    print("=" * 60)

    simple_traj = _simple_traj("帮我对AI Agent进行安全性分析")
    complex_traj = _complex_traj()
    simple_query = "帮我对AI Agent进行安全性分析"
    complex_query = "帮我分析一下2026年AI Agent行业，对比三个主流框架，并给出推荐"

    # ── 场景 A-E：Eval Loop 迭代行为 ──
    scenarios = [
        ("A. 一次性通过（所有步骤高分）", simple_query, simple_traj, "good",       3),
        ("B. 一次修正后通过",           complex_query, complex_traj, "fix_once",   3),
        ("C. 三次修正才通过",           complex_query, complex_traj, "fix_twice",  3),
        ("D. 最大迭代仍未通过",         complex_query, complex_traj, "never_pass", 3),
        ("E. 震荡检测自动停止",         complex_query, complex_traj, "oscillate",  5),
    ]

    results = []
    for name, query, traj, scenario, max_iter in scenarios:
        r = run_scenario(name, query, traj, scenario, max_iterations=max_iter)
        results.append(r)

    # ── 场景 F：功能测试类不走 loop ──
    print(f"\n{'=' * 60}")
    print(f"  场景 F: 功能测试类（不走 loop）")
    print(f"  {'=' * 60}")
    agent_fn, _ = _make_agent_fn(simple_traj)
    judge_fn, jc, _ = _make_judge_for_scenario("good")
    config = EvalLoopConfig(max_iterations=3, verbose=False)
    engine = EvalLoopEngine(agent_fn=agent_fn, judge_fn=judge_fn, config=config)
    result = engine.execute("现在几点了？测试一下时间工具")
    print(f"  [意图] {result.task_type}")
    print(f"  [迭代] {result.iterations} 次")
    assert result.task_type == TaskType.FUNCTIONAL_TEST
    assert result.iterations == 1  # 不走 loop，一次
    print(f"  [结果] 通过: {'✅' if result.passed else '❌'}")
    print(f"  ✅ 功能测试类不走 loop，正确")

    # ── 场景 G-H：复杂输入 + 错误传播 ──
    test_scenario_g()
    test_scenario_h()

    # ── 汇总 ──
    print(f"\n{'=' * 60}")
    print(f"  场景测试汇总")
    print(f"  {'=' * 60}")
    print(f"  {'场景':<30} {'迭代':<6} {'通过':<6} {'总分':<8} {'Judge调用':<10}")
    print(f"  {'-'*60}")
    for r in results:
        print(f"  {r['name']:<30} {r['iterations']:<6} "
              f"{'✅' if r['passed'] else '❌':<6} "
              f"{r['overall_score']:<8.3f} {r['judge_calls']:<10}")
    print(f"  {'-'*60}")
    print(f"  E. 震荡: 期望提前停止（< 3 次迭代）")
    print(f"  D. 始终失败: 期望 3 次都用满")

    # 验证关键断言
    print(f"\n  ── 关键断言验证 ──")
    assert results[0]["iterations"] == 1, f"A 应迭代 1 次，实际 {results[0]['iterations']}"
    assert results[0]["passed"] is True, "A 应通过"
    print(f"  ✅ A: 高质量任务 1 次通过")

    assert results[1]["iterations"] == 2, f"B 应迭代 2 次，实际 {results[1]['iterations']}"
    assert results[1]["passed"] is True, "B 应通过"
    print(f"  ✅ B: 修正后第 2 次通过")

    assert results[3]["iterations"] == 3, f"D 应迭代 3 次，实际 {results[3]['iterations']}"
    assert results[3]["passed"] is False, "D 应失败"
    print(f"  ✅ D: 始终低质量，3 次用完仍未通过")

    assert results[4]["iterations"] < 5, f"E 应提前停止，实际 {results[4]['iterations']}"
    assert results[4]["passed"] is False, "E 应失败"
    print(f"  ✅ E: 震荡检测生效，{results[4]['iterations']} 次后提前停止（最大 5 次）")

    # ── 场景 I: HITL 用户拒绝修正 ──
    print(f"\n  ── 场景 I: HITL 权限拦截 ──")
    from handwritten_react_agent.safety.human_in_the_loop import HumanInTheLoop

    def _mock_ask_reject(msg, choices):
        return "3"  # 拒绝（在 CONFIRM 上下文中 3=拒绝）

    hitl = HumanInTheLoop(ask_fn=_mock_ask_reject)
    engine_hitl = EvalLoopEngine(
        agent_fn=_make_agent_fn(complex_traj)[0],
        judge_fn=_make_judge_for_scenario("fix_once")[0],
        config=EvalLoopConfig(max_iterations=3, verbose=False),
        hitl=hitl,
    )
    result_hitl = engine_hitl.execute(complex_query)
    print(f"  [迭代] {result_hitl.iterations} 次（期望 1 次，因为用户拒绝重试）")
    print(f"  [通过] {'✅' if result_hitl.passed else '❌'}")
    print(f"  [审计] HITL 记录条数: {len(hitl.audit_log)}")
    assert result_hitl.iterations == 1, f"HITL 应 1 次后停止，实际 {result_hitl.iterations}"
    assert result_hitl.passed is False, "HITL 拦截后应失败"
    assert len(hitl.audit_log) >= 1, "HITL 应有审计记录"
    print(f"  ✅ 用户拒绝修正后，循环在第 1 次迭代后停止")

    print(f"\n{'=' * 60}")
    print(f"  ✅ 全部场景测试完成")
    print(f"  {'=' * 60}")
