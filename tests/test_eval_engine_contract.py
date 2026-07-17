"""跨仓 API 契约：react-agent ↔ llm-eval-engine ProcessRewardScorer。"""
from __future__ import annotations

import inspect
import sys

import pytest

pytest.importorskip("eval_engine")

from eval_engine.core.process_reward import ProcessRewardScorer  # noqa: E402

from react_agent.eval.scorer import (  # noqa: E402
    EVAL_API_VERSION,
    EVAL_ENGINE_API_CONTRACT,
    EvalIntegrationError,
    score_with_eval_engine,
)

_TRAJ = {
    "session_id": "contract_test",
    "query": "test query",
    "steps": [
        {
            "step": 1,
            "thought": "search",
            "action": {"name": "web_search", "arguments": "{}"},
            "observation": "ok",
        }
    ],
    "final_answer": "done",
}


def test_eval_api_version_pinned():
    assert EVAL_API_VERSION == "0.1"
    assert EVAL_ENGINE_API_CONTRACT == f"ProcessRewardScorer.extra_contracts@{EVAL_API_VERSION}"
    from eval_engine.core.process_reward import EVAL_API_VERSION as ee_ver

    assert ee_ver == EVAL_API_VERSION


def test_process_reward_scorer_accepts_extra_contracts_not_verifiers():
    sig = inspect.signature(ProcessRewardScorer.__init__)
    assert "extra_contracts" in sig.parameters
    assert "verifiers" not in sig.parameters


def test_score_with_eval_engine_success_shape():
    result = score_with_eval_engine(
        {"expected_tool": "web_search"},
        _TRAJ,
    )
    assert result is not None
    assert result.get("status") == "success"
    assert result.get("eval_engine") is True
    assert "error" not in result
    assert result["total"] > 0
    assert result["passed"] is True
    assert result.get("api_contract") == EVAL_ENGINE_API_CONTRACT


def test_legacy_verifiers_kwarg_raises_type_error():
    with pytest.raises(TypeError):
        ProcessRewardScorer(judge_fn=lambda _p: {}, verifiers=[])


def test_ci_verify_integration_script_exits_zero():
    """与 CI 同路径：tests/ci_verify_integration.py 必须 exit 0。"""
    import subprocess
    from pathlib import Path

    script = Path(__file__).resolve().parent / "ci_verify_integration.py"
    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "[PASS]" in proc.stdout
