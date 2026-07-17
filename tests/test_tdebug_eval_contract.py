"""跨仓契约：Harness fixture → schema → trace-debugger → eval-engine。"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("trace_debugger")
pytest.importorskip("eval_engine")

from react_agent.eval.scorer import (  # noqa: E402
    EVAL_ENGINE_API_CONTRACT,
    score_with_eval_engine,
)
from react_agent.harness.schema import (  # noqa: E402
    SCHEMA_VERSION,
    assert_valid,
    normalize_trajectory,
    validate_trajectory,
)
from trace_debugger.reader import parse as tdebug_parse  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples" / "fixtures" / "harness_closed_loop.json"


def test_fixture_schema_tdebug_eval_chain():
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert validate_trajectory(raw) == []
    assert_valid(raw)

    norm = normalize_trajectory(raw)
    assert norm["schema_version"] == SCHEMA_VERSION

    traj = tdebug_parse(norm)
    assert traj.steps, "tdebug must parse at least one step"
    assert all(s.index >= 1 for s in traj.steps), "Format B steps are 1-based"
    first = traj.steps[0]
    assert first.action_name, "first step should have a tool name"
    assert isinstance(first.action_args, str)

    result = score_with_eval_engine(
        {"expected_tool": first.action_name},
        norm,
    )
    assert result is not None
    assert result.get("status") == "success"
    assert result.get("api_contract") == EVAL_ENGINE_API_CONTRACT
    assert result.get("eval_engine") is True


def test_args_object_survives_normalize_into_tdebug_string():
    raw = {
        "session_id": "tdebug_contract",
        "query": "q",
        "final_answer": "a",
        "steps": [
            {
                "step": 1,
                "thought": "t",
                "action": {"name": "calculator", "args": {"expr": "1+1"}},
                "observation": "2",
            }
        ],
    }
    norm = normalize_trajectory(raw)
    action = norm["steps"][0]["action"]
    assert "arguments" in action
    assert isinstance(action["arguments"], str)
    assert "1+1" in action["arguments"]

    traj = tdebug_parse(norm)
    assert traj.steps[0].index == 1
    assert traj.steps[0].action_name == "calculator"
    assert "1+1" in traj.steps[0].action_args
