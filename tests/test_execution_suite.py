"""Execution suite：offline + agent(mock) 测试"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from react_agent.eval.execution_scorer import (
    load_execution_dataset,
    run_execution_suite,
    score_task,
)


def test_dataset_has_both_modes():
    tasks = load_execution_dataset()
    modes = {t.get("mode", "offline_tools") for t in tasks}
    assert "offline_tools" in modes
    assert "agent" in modes


def test_suite_offline_full_pass():
    report = run_execution_suite(modes=["offline_tools"])
    s = report["summary"]
    assert s["total"] >= 6
    assert s["passed"] == s["total"], report
    assert s["pass_rate"] == 100.0


def test_failing_expectation():
    task = {
        "id": "bad",
        "mode": "offline_tools",
        "tags": ["execution"],
        "steps": [
            {
                "tool": "calculator",
                "arguments": {"expression": "1 + 1"},
                "expect_equals": "999",
            }
        ],
    }
    r = score_task(task)
    assert r["passed"] is False


def test_agent_mode_with_mock_runner():
    def mock_runner(question, timeout=90, max_steps=None, provider=None):
        traj = {
            "final_answer": "FINAL ANSWER: 323",
            "steps": [
                {
                    "step": 1,
                    "action": {"name": "calculator", "arguments": "{}"},
                    "observation": "323",
                }
            ],
        }
        stdout = "[调工具] calculator({})\n>>> 最终答案: 323\nFINAL ANSWER: 323"
        return stdout, traj, 0, 0.1

    report = run_execution_suite(
        modes=["agent"],
        agent_runner=mock_runner,
        only_ids={"agent_calc_17x19"},
    )
    assert report["summary"]["total"] == 1
    assert report["summary"]["passed"] == 1
