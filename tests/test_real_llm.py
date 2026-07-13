"""
真实 LLM 集成测试 — 需配置 API Key
===================================

测试条件（满足任一即可）：
  1. DEEPSEEK_API_KEY 环境变量已设置
  2. llm_config.json 存在且包含有效 provider

无 API Key 时自动跳过（pytest 的 skip）。
"""

import os
import sys
import json

import pytest

# 检测是否有可用 API Key
def _has_api_key() -> bool:
    if os.environ.get("DEEPSEEK_API_KEY"):
        return True
    # 检查 llm_config.json
    for path in [".", "..", "../.."]:
        cfg = os.path.join(path, "llm_config.json")
        if os.path.exists(cfg):
            return True
    return False

REQUIRES_API = pytest.mark.skipif(
    not _has_api_key(),
    reason="需要 DEEPSEEK_API_KEY 或 llm_config.json"
)


@REQUIRES_API
def test_react_loop_with_real_llm():
    """真实 LLM 跑一次 ReAct 循环

    验证 Agent 能完整走完 thought → action → observation 循环。
    使用简单 query 减少 token 消耗。
    """
    from react_agent.react_loop import react_loop

    result = react_loop("法国的首都是什么？回答一个字即可", max_steps=3)
    output = result if isinstance(result, str) else result.get("output", "")
    assert len(output) > 0, "Agent 未产生输出"
    assert any(c in output for c in ["巴", "黎", "Paris", "paris"]), (
        f"输出不含预期答案: {output[:100]}"
    )
    print(f"✅ Agent 真实 LLM 测试通过: {output[:60]}")


@REQUIRES_API
def test_trajectory_recorded():
    """真实 LLM 执行后轨迹被正确录制"""
    from react_agent.react_loop import react_loop
    from react_agent.harness import current_trajectory

    _ = react_loop("中国的首都是什么？", max_steps=3)
    traj = current_trajectory()
    assert traj is not None, "无轨迹记录"
    if traj:
        assert len(traj.steps) > 0, "轨迹中没有步骤"
        print(f"✅ 轨迹录制成功: {len(traj.steps)} 步")


@REQUIRES_API
def test_eval_engine_with_real_judge():
    """llm-eval-engine 用真实 LLM 作为 Judge 评分

    需同时安装 llm-eval-engine。
    """
    pytest.importorskip("eval_engine")
    from eval_engine.loop.eval_loop import EvalLoopEngine, EvalLoopConfig
    from eval_engine.judge.executor import JudgeExecutor

    # 用真实 LLM 作为 Judge
    judge = JudgeExecutor()

    # 用 mock agent 生成轨迹
    def mock_agent(query):
        return {
            "output": "北京是中国的首都",
            "trajectory": {
                "steps": [
                    {"step": 0, "thought": "我知道答案", "observation": ""},
                ]
            }
        }

    engine = EvalLoopEngine(agent_fn=mock_agent, judge_fn=judge)
    result = engine.execute("中国的首都是什么？")
    assert result.passed is not None
    print(f"✅ 真实 Judge 评分: score={result.report.overall_score:.2f}")
