"""端到端集成测试：模拟完整的 Eval Loop 全流程

测试路径：
  1. 模拟 Agent 执行 → 生成轨迹
  2. trajectory_parser 解析轨迹 → DAG
  3. intent/classifier 意图分类
  4. JudgeExecutor 评分（模拟）
  5. ProcessRewardScorer 逐步骤评分
  6. error_propagation 错误传播分析
  7. fix_packer 打包修正指令
  8. BaselineManager 保存/对比 baseline
  9. RegressionGate 回归门禁
  10. report 输出文本报告
"""

import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

from core.contract import VerifierContract
from core.trajectory_parser import parse_trajectory, dag_to_text, dag_summary
from core.dynamic_rubric import build_step_context, build_step_judge_prompt
from core.process_reward import (
    ProcessRewardScorer, analyze_error_propagation, pack_revision_instructions
)
from handwritten_react_agent.intent.classifier import IntentClassifier, TaskType
from judge.executor import JudgeExecutor, _extract_json
from gates.baseline import BaselineManager
from gates.regression_gate import RegressionGate
from observability.report import format_report, report_to_json


# ── 模拟轨迹 ──

MOCK_TRAJECTORY = {
    "session_id": "traj_e2e_001",
    "query": "搜索2026年AI Agent的发展趋势并总结",
    "steps": [
        {
            "step_index": 0, "type": "thought",
            "content": "用户想了解2026年AI Agent的发展趋势，需要搜索相关信息并总结。先搜索。",
        },
        {
            "step_index": 1, "type": "action",
            "action": {"name": "web_search", "args": {"query": "2026 AI Agent trends"}},
            "content": "web_search(query='2026 AI Agent trends')",
        },
        {
            "step_index": 2, "type": "observation",
            "content": "搜索结果显示了多智能体编排、MCP协议标准化、Agent评测等趋势",
            "observation": "2026年AI Agent主要趋势包括：\n1. 多Agent编排成为主流\n2. MCP协议标准化（Linux Foundation）\n3. Agent评测与可观测性需求增长",
        },
        {
            "step_index": 3, "type": "thought",
            "content": "已获取到有用的信息，可以做总结了。重点突出三个趋势。",
        },
        {
            "step_index": 4, "type": "final",
            "content": "2026年AI Agent的主要发展趋势包括：\n\n1. 多Agent编排成为主流\n   - 从单Agent转向专业Agent团队协作\n   - Gartner显示多Agent系统咨询量增长1445%\n\n2. MCP协议标准化\n   - MCP已由Linux Foundation管理\n   - 月均9700万SDK下载，5800+服务器\n\n3. Agent评测需求爆发\n   - 32%的企业认为质量是主要障碍\n   - LLM-as-Judge和Eval Pipeline成为必备设施",
        },
    ],
    "total_steps": 5,
    "total_tokens_estimated": 800,
    "final_answer": "2026年AI Agent的主要发展趋势包括：\n\n1. 多Agent编排成为主流\n   - 从单Agent转向专业Agent团队协作\n   - Gartner显示多Agent系统咨询量增长1445%\n\n2. MCP协议标准化\n   - MCP已由Linux Foundation管理\n   - 月均9700万SDK下载，5800+服务器\n\n3. Agent评测需求爆发\n   - 32%的企业认为质量是主要障碍\n   - LLM-as-Judge和Eval Pipeline成为必备设施",
}


# ── 模拟 Judge（按步骤类型返回不同分数） ──

_judge_call_count = {"count": 0}


def mock_judge_fn(prompt: str) -> dict:
    """模拟 Judge LLM 调用"""
    _judge_call_count["count"] += 1

    # 根据步骤类型返回不同分数
    if "类型: thought" in prompt:
        score = 4.0
    elif "类型: action" in prompt:
        if "search" in prompt or "web" in prompt:
            score = 4.5
        else:
            score = 3.5
    elif "类型: observation" in prompt:
        score = 4.0
    elif "类型: final" in prompt:
        score = 3.0  # 最终答案有提升空间
    else:
        score = 3.5

    return {
        "role_understanding": f"Agent 正在执行该步骤",
        "rubrics": [
            {"dimension": "quality", "criteria": "步骤质量",
             "score": score, "reason": "模拟评分"},
        ],
        "step_score": score,
        "needs_revision": score < 3.5,
    }


# ── 测试用例 ──


def test_e2e_full_flow():
    """端到端完整流程"""
    print("=" * 55)
    print("  端到端集成测试")
    print("=" * 55)

    # 1. 意图分类
    print("\n[1/10] IntentClassifier...")
    classifier = IntentClassifier()
    task_type = classifier.classify(MOCK_TRAJECTORY["query"])
    print(f"  输入: {MOCK_TRAJECTORY['query'][:50]}...")
    print(f"  分类: {task_type}")
    assert task_type == TaskType.GENERATIVE_TASK  # 生成式任务
    print("  ✅")

    # 2. 轨迹解析
    print("\n[2/10] TrajectoryParser...")
    dag = parse_trajectory(MOCK_TRAJECTORY)
    assert dag.num_steps == 5
    assert dag.get_node(1).tool_name == "web_search"
    print(f"  DAG: {dag.num_steps} 步, {len(dag.edges)} 条边")
    print(f"  工具: {[n.tool_name for n in dag.nodes if n.tool_name]}")
    print("  ✅")

    # 3. DAG 摘要
    print("\n[3/10] DAG Summary...")
    summary = dag_summary(dag)
    print(f"  摘要: {json.dumps(summary, ensure_ascii=False)}")
    assert summary["total_steps"] == 5
    print("  ✅")

    # 4. Process Reward 评分
    print("\n[4/10] ProcessRewardScorer...")
    scorer = ProcessRewardScorer(judge_fn=mock_judge_fn, min_step_score=3.5)
    report = scorer.score_trajectory(dag, fast_mode=False)
    print(f"  总分: {report.overall_score:.3f}")
    print(f"  失败步骤: {report.num_failed_steps}")
    print(f"  需修正: {report.needs_revision}")
    for s in report.per_step:
        icon = "✅" if not s.needs_revision else "❌"
        print(f"  {icon} Step {s.step_index} [{s.step_type}]: {s.step_score:.2f}")
    assert report.num_scored > 0
    print("  ✅")

    # 5. 错误传播分析
    print("\n[5/10] ErrorPropagation...")
    error_analysis = analyze_error_propagation(report, dag)
    print(f"  错误源头: {error_analysis['error_sources']}")
    print(f"  影响分析: {error_analysis['impact_summary']}")
    print("  ✅")

    # 6. 修正指令
    print("\n[6/10] FixPacker...")
    fix = pack_revision_instructions(report, dag)
    print(f"  修正指令长度: {len(fix)} 字符")
    print(f"  预览: {fix[:120]}...")
    print("  ✅")

    # 7. DAG 文本表示
    print("\n[7/10] DAG to Text...")
    text = dag_to_text(dag)
    print(f"  可读轨迹 ({len(text)} 字符):")
    for line in text.split("\n")[:6]:
        print(f"  {line}")
    print("  ✅")

    # 8. Baseline 管理
    print("\n[8/10] BaselineManager...")
    with tempfile.TemporaryDirectory() as tmpdir:
        bm = BaselineManager(baseline_dir=tmpdir)

        # 第一次：没有 baseline → 保存
        report_json = report_to_json(type("result", (), {
            "query": MOCK_TRAJECTORY["query"],
            "task_type": "generative_task",
            "passed": not report.needs_revision,
            "iterations": 1,
            "final_output": MOCK_TRAJECTORY["final_answer"],
            "report": report,
            "error_analysis": error_analysis,
            "healing_log": [],
        })())
        path = bm.save(report_json)
        print(f"  保存 baseline: {os.path.basename(path)}")

        # 对比（第一次应该没有 baseline 文件）
        # 因为刚保存的 baseline 是当前目录下的 baselines/ 子目录，而 tmpdir 是单独的
        # 实际上 bm.save 保存在 tmpdir 下，bm.compare 从 tmpdir 下读取
        bm2 = BaselineManager(baseline_dir=tmpdir)
        comparison = bm2.compare(report_json)
        print(f"  对比结果: baseline_found={comparison['baseline_found']}")
        if comparison["baseline_found"]:
            print(f"  Score diff: {comparison['score_diff']}")
        print("  ✅")

    # 9. Regression Gate
    print("\n[9/10] RegressionGate...")
    with tempfile.TemporaryDirectory() as tmpdir:
        bm = BaselineManager(baseline_dir=tmpdir)
        gate = RegressionGate(baseline_manager=bm)

        # 首次评估（无 baseline）
        report_json2 = report_to_json(type("result", (), {
            "query": MOCK_TRAJECTORY["query"],
            "task_type": "generative_task",
            "passed": True,
            "iterations": 1,
            "final_output": MOCK_TRAJECTORY["final_answer"],
            "report": report,
            "error_analysis": error_analysis,
            "healing_log": [],
        })())
        result = gate.evaluate(report_json2)
        print(f"  首次: {result['message']}")
        assert result["passed"]  # 首次应该通过

        # 第二次（对比 baseline）——添加 by_category
        report_json2["by_category"] = {
            "rag": {"overall_score": 4.0, "num_cases": 2},
            "search": {"overall_score": 4.2, "num_cases": 1},
        }
        result2 = gate.evaluate(report_json2)
        print(f"  第二次: {result2['message']}")
        print("  ✅")

    # 10. 报告输出
    print("\n[10/10] Report Output...")
    e2e_result = type("result", (), {
        "query": MOCK_TRAJECTORY["query"],
        "task_type": "generative_task",
        "final_output": MOCK_TRAJECTORY["final_answer"],
        "report": report,
        "iterations": 2,
        "healing_log": [
            {"iteration": 1, "overall_score": 3.2, "needs_revision": True, "num_failed_steps": 1, "error_sources": [], "final_output_preview": ""},
            {"iteration": 2, "overall_score": 4.1, "needs_revision": False, "num_failed_steps": 0, "error_sources": [], "final_output_preview": ""},
        ],
        "error_analysis": error_analysis,
        "passed": not report.needs_revision,
    })()

    text_report = format_report(e2e_result, detailed=True)
    print(f"  报告长度: {len(text_report)} 字符")
    for line in text_report.split("\n")[:15]:
        print(f"  {line}")
    print("  ...")

    json_report = report_to_json(e2e_result)
    print(f"  JSON keys: {list(json_report.keys())}")
    assert "timestamp" in json_report
    assert "overall_score" in json_report
    print("  ✅")

    # 汇总
    print("\n" + "=" * 55)
    print("  ✅ 全部 10 个阶段通过")
    print(f"  Judge 调用次数: {_judge_call_count['count']}")
    print("=" * 55)


if __name__ == "__main__":
    test_e2e_full_flow()
