"""
端到端真实集成：调用 Agent → 捕获轨迹 → eval-engine 评分

运行方式（在项目根目录）：
    cd /d/agent_learning/repo
    python experiments/eval-engine/tests/run_real_e2e.py

会：
  1. 用 Agent 实际执行一次查询
  2. 将轨迹传给 eval-engine 逐步骤评分
  3. 输出完整评估报告
"""

import sys
import os
import json

# ── 路径设置 ──

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
EVAL_ENGINE_DIR = os.path.join(REPO_ROOT, "experiments", "eval-engine")
SRC_DIR = os.path.join(REPO_ROOT, "src")

sys.path.insert(0, SRC_DIR)
sys.path.insert(0, EVAL_ENGINE_DIR)

# ── 导入 Agent ──

from handwritten_react_agent.react_loop import react_loop
from handwritten_react_agent.harness.recorder import current_trajectory
from handwritten_react_agent.llm import LLM

# ── 导入 eval-engine ──

from core.trajectory_parser import parse_trajectory, dag_to_text
from core.process_reward import ProcessRewardScorer, analyze_error_propagation
from handwritten_react_agent.intent.classifier import IntentClassifier
from judge.executor import JudgeExecutor
from observability.report import format_report


def run_agent_and_eval(query: str) -> None:
    """运行 Agent 并执行 eval-engine 完整流程"""
    print("=" * 60)
    print(f"  Eval Engine 真实端到端测试")
    print(f"  查询: {query}")
    print("=" * 60)

    # ── 第 1 阶段：意图分类 ──
    print("\n[1] 意图分类...")
    classifier = IntentClassifier()
    task_type = classifier.classify(query)
    print(f"    结果: {task_type}")

    # ── 第 2 阶段：Agent 执行 ──
    print(f"\n[2] Agent 执行中...")
    print(f"    {'─' * 40}")

    # 使用项目的默认 LLM
    try:
        # Agent 输出会打印到 stdout，我们让它执行
        final_answer = react_loop(query, max_steps=8)
        print(f"    {'─' * 40}")
        print(f"    最终答案: {final_answer[:200]}")
    except Exception as e:
        print(f"    ❌ Agent 执行失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # ── 第 3 阶段：获取轨迹 ──
    print(f"\n[3] 获取轨迹...")
    traj_obj = current_trajectory()
    if traj_obj is None:
        print("    ⚠ 轨迹已结束（finish_trajectory 已清空），从最新文件读取")

        # 从 trajectories/ 目录读取最新文件
        traj_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "..", "..", "src", "handwritten_react_agent", "trajectories",
        )
        # 也试试 repo 根下的 trajectories/
        alt_traj_dir = os.path.join(REPO_ROOT, "trajectories")

        traj_data = None
        for d in [traj_dir, alt_traj_dir]:
            if os.path.exists(d):
                files = sorted(
                    [f for f in os.listdir(d) if f.endswith(".json")],
                    reverse=True,
                )
                if files:
                    latest = os.path.join(d, files[0])
                    with open(latest, encoding="utf-8") as f:
                        traj_data = json.load(f)
                    print(f"    从 {latest} 读取")
                    break
    else:
        print("    从内存对象读取轨迹")
        traj_data = traj_obj.to_dict()

    if not traj_data:
        print("    ❌ 无法获取轨迹数据")
        return

    print(f"    会话: {traj_data.get('session_id', 'N/A')}")
    print(f"    步骤数: {traj_data.get('total_steps', 0)}")
    print(f"    模型: {traj_data.get('model', 'N/A')}")

    # ── 第 4 阶段：轨迹解析 ──
    print(f"\n[4] 轨迹解析 → DAG...")
    try:
        dag = parse_trajectory(traj_data)
        print(f"    DAG: {dag.num_steps} 步")
        for node in dag.nodes:
            tool = f" ({node.tool_name})" if node.tool_name else ""
            print(f"      Step {node.step_index} [{node.step_type}]{tool}")
    except ValueError as e:
        print(f"    ❌ 轨迹解析失败: {e}")
        return

    # ── 第 5 阶段：Judge 评分 ──
    print(f"\n[5] Process Reward 评分...")

    # 创建 Judge（使用项目的 LLM 配置，但用更低温度）
    judge = JudgeExecutor(
        temperature=0.1,
        max_tokens=1024,
    )
    print(f"    Judge 配置: {judge}")

    scorer = ProcessRewardScorer(
        judge_fn=judge,
        min_step_score=3.5,
    )
    report = scorer.score_trajectory(dag, fast_mode=False)

    print(f"    总分: {report.overall_score:.3f}/5.0")
    print(f"    失败步骤: {report.num_failed_steps}/{report.num_scored}")
    print(f"    需修正: {report.needs_revision}")

    for step in report.per_step:
        icon = "✅" if not step.needs_revision else "❌"
        print(f"    {icon} Step {step.step_index} [{step.step_type}]: {step.step_score:.2f}")
        for r in step.rubrics:
            if r.needs_revision:
                print(f"        ❌ {r.dimension}: {r.reason[:80]}")

    # ── 第 6 阶段：错误传播分析 ──
    print(f"\n[6] 错误传播分析...")
    error_analysis = analyze_error_propagation(report, dag)
    if error_analysis.get("error_sources"):
        print(f"    错误源头: Step {error_analysis['error_sources']}")
        print(f"    影响: {error_analysis['impact_summary']}")
    else:
        print(f"    未检测到明显的错误传播链")

    # ── 第 7 阶段：报告输出 ──
    print(f"\n[7] 评估报告:")
    print()

    # 构建 report_to_json 所需的动态对象
    class _Result:
        pass

    result = _Result()
    result.query = query
    result.task_type = task_type
    result.final_output = dag.final_answer
    result.report = report
    result.iterations = 1
    result.healing_log = []
    result.error_analysis = error_analysis
    result.passed = not report.needs_revision

    text_report = format_report(result, detailed=True)
    print(text_report)

    # ── 结束 ──
    print(f"\nJudge 调用统计: {judge.stats}")


if __name__ == "__main__":
    # 默认测试查询
    import argparse

    parser = argparse.ArgumentParser(description="运行 Agent 并执行 Eval Engine 完整流程")
    parser.add_argument("query", nargs="?", default="现在几点了？",
                        help="要测试的查询（默认: '现在几点了？'）")
    args = parser.parse_args()

    run_agent_and_eval(args.query)
