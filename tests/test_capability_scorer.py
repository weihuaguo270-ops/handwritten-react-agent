"""capability_scorer 无网络单测"""

from react_agent.eval.dataset import TestCase, filter_by_capability, load_dataset
from react_agent.eval.capability_scorer import (
    score_capability,
    extract_tools,
    normalize_answer,
)
from react_agent.eval.report import generate_report


def _case(**kw):
    return TestCase(kw)


def _traj(final="", tools=None, thoughts=None, observations=None):
    steps = []
    tools = tools or []
    thoughts = thoughts or [""] * max(1, len(tools) or 1)
    observations = observations or [""] * len(tools)
    if not tools:
        steps.append({"step": 1, "thought": thoughts[0] if thoughts else ""})
    else:
        for i, name in enumerate(tools):
            steps.append({
                "step": i + 1,
                "thought": thoughts[i] if i < len(thoughts) else "",
                "action": {"name": name, "arguments": "{}"},
                "observation": observations[i] if i < len(observations) else "",
            })
    return {
        "final_answer": final,
        "total_steps": len(steps),
        "steps": steps,
    }


def test_accuracy_pass():
    case = _case(
        id="a1", capability="accuracy",
        question="q", expected_answer="巴黎",
    )
    traj = _traj(final="巴黎")
    r = score_capability(case, "最终答案: 巴黎", traj)
    assert r["passed"] is True
    assert r["metrics"]["accuracy"] == 1.0


def test_accuracy_fail():
    case = _case(
        id="a2", capability="accuracy",
        question="q", expected_answer="巴黎",
    )
    traj = _traj(final="伦敦")
    r = score_capability(case, "最终答案: 伦敦", traj)
    assert r["passed"] is False


def test_tool_selection_precision_recall():
    case = _case(
        id="t1", capability="tool_selection",
        question="q", expected_tools=["calculator"],
    )
    traj = _traj(final="136", tools=["calculator"])
    r = score_capability(case, "[调工具] calculator({\"expression\":\"1\"})", traj)
    assert r["metrics"]["tool_recall"] == 1.0
    assert r["metrics"]["tool_precision"] == 1.0
    assert r["passed"] is True


def test_tool_selection_missing_tool():
    case = _case(
        id="t2", capability="tool_selection",
        question="q", expected_tools=["calculator"],
    )
    traj = _traj(final="136", tools=["web_search"])
    r = score_capability(case, "", traj)
    assert r["metrics"]["tool_recall"] == 0.0
    assert r["passed"] is False


def test_tool_time_alias_or():
    """get_time / get_current_time 视为别名，命中其一即可。"""
    case = _case(
        id="t_alias", capability="tool_selection",
        question="q",
        expected_tools=["get_time", "get_current_time"],
        expected_tool_groups=[["get_time", "get_current_time"]],
    )
    traj = _traj(final="ok", tools=["get_current_time"])
    r = score_capability(case, "", traj)
    assert r["metrics"]["tool_recall"] == 1.0
    assert r["passed"] is True


def test_tool_search_allows_fetch_page():
    """web_search 为主工具时，额外 fetch_page 不应直接判失败。"""
    case = _case(
        id="t_weather", capability="tool_selection",
        question="q",
        expected_tools=["web_search"],
        expected_tool_groups=[["web_search"]],
    )
    traj = _traj(
        final="ok",
        tools=["get_current_time", "web_search", "fetch_page", "fetch_page"],
    )
    r = score_capability(case, "", traj)
    assert r["metrics"]["tool_recall"] == 1.0
    assert r["passed"] is True


def test_tool_sequence():
    case = _case(
        id="t3", capability="tool_selection",
        question="q",
        expected_tools=["get_time", "calculator"],
        expected_tool_sequence=["get_time", "calculator"],
    )
    traj = _traj(final="ok", tools=["get_time", "calculator"])
    r = score_capability(case, "", traj)
    assert r["metrics"].get("sequence_ok") is True
    assert r["passed"] is True


def test_reasoning_checkpoints():
    case = _case(
        id="r1", capability="reasoning",
        question="q", expected_answer="29",
        reasoning_checkpoints=["翻", "30", "29"],
    )
    traj = _traj(
        final="第 29 天",
        thoughts=["每天翻一倍，第30天全满，所以一半是29"],
    )
    r = score_capability(case, "FINAL ANSWER: 29", traj)
    assert r["passed"] is True
    assert r["metrics"]["reasoning_pass"] == 1.0


def test_consistency_agree():
    case = _case(
        id="c1", capability="consistency",
        question="q", expected_answer="1918",
        consistency_runs=3,
    )
    runs = [
        {"stdout": "最终答案: 1918", "trajectory": _traj("1918")},
        {"stdout": "最终答案: 1918年", "trajectory": _traj("1918年")},
        {"stdout": "FINAL ANSWER: 1918", "trajectory": _traj("1918")},
    ]
    r = score_capability(case, runs[-1]["stdout"], runs[-1]["trajectory"], run_results=runs)
    assert r["metrics"]["consistency_rate"] >= 0.67
    assert r["passed"] is True


def test_consistency_disagree():
    case = _case(
        id="c2", capability="consistency",
        question="q", expected_answer="1918",
        consistency_runs=3,
    )
    runs = [
        {"stdout": "1918", "trajectory": _traj("1918")},
        {"stdout": "1920", "trajectory": _traj("1920")},
        {"stdout": "1917", "trajectory": _traj("1917")},
    ]
    r = score_capability(case, runs[-1]["stdout"], runs[-1]["trajectory"], run_results=runs)
    assert r["passed"] is False


def test_hallucination_forbid():
    case = _case(
        id="h1", capability="hallucination",
        question="q", expected_answer="1918",
        forbid_claims=["1919", "1920"],
    )
    traj = _traj(final="1920")
    r = score_capability(case, "最终答案: 1920", traj)
    assert r["metrics"]["hallucinated"] == 1.0
    assert r["passed"] is False


def test_hallucination_clean():
    case = _case(
        id="h2", capability="hallucination",
        question="q", expected_answer="1918",
        forbid_claims=["1919", "1920"],
    )
    traj = _traj(final="1918")
    r = score_capability(case, "最终答案: 1918", traj)
    assert r["metrics"]["hallucinated"] == 0.0
    assert r["passed"] is True


def test_hallucination_grounded():
    case = _case(
        id="h3", capability="hallucination",
        question="q", expected_answer="7006652",
        require_grounded=True,
        forbid_claims=["7006653", "0"],
    )
    traj = _traj(
        final="7006652",
        tools=["calculator"],
        observations=["7006652"],
    )
    r = score_capability(case, "最终答案: 7006652", traj)
    assert r["passed"] is True
    assert r["metrics"]["hallucinated"] == 0.0


def test_hallucination_digit_boundary():
    """短数字禁止主张不得子串命中更大数字。"""
    case = _case(
        id="h4", capability="hallucination",
        question="q", expected_answer="7006652",
        forbid_claims=["0"],
    )
    traj = _traj(final="7006652")
    r = score_capability(case, "最终答案: 7006652", traj)
    assert r["passed"] is True

    case_fail = _case(
        id="h5", capability="hallucination",
        question="q", expected_answer="1",
        forbid_claims=["0"],
    )
    traj_fail = _traj(final="结果是 0")
    r_fail = score_capability(case_fail, "最终答案: 0", traj_fail)
    assert r_fail["passed"] is False


def test_functional_fallback_no_capability():
    case = _case(
        id="f1", question="q",
        expected_tools=["calculator"],
        must_contain=["136"],
        max_steps=5,
    )
    traj = _traj(final="结果是 136", tools=["calculator"])
    stdout = ">>> 最终答案: 136\n[调工具] calculator("
    r = score_capability(case, stdout, traj)
    assert r["capability"] is None
    assert "functional" in r
    assert r["max_score"] == 4


def test_filter_by_capability_and_dataset_loads():
    cases = load_dataset("capability")
    assert len(cases) >= 15
    acc = filter_by_capability(cases, "accuracy")
    assert all(c.capability == "accuracy" for c in acc)
    assert len(acc) >= 3
    all_caps = filter_by_capability(cases, "all")
    assert len(all_caps) == len(cases)


def test_report_by_capability():
    cases = [
        _case(id="a", capability="accuracy", question="q", expected_answer="巴黎"),
        _case(id="t", capability="tool_selection", question="q", expected_tools=["calculator"]),
    ]
    raw = [
        {
            "stdout": "最终答案: 巴黎",
            "trajectory": _traj("巴黎"),
            "exit_code": 0,
            "duration_seconds": 1.0,
            "timed_out": False,
        },
        {
            "stdout": "[调工具] calculator(\n最终答案: 1",
            "trajectory": _traj("1", tools=["calculator"]),
            "exit_code": 0,
            "duration_seconds": 2.0,
            "timed_out": False,
        },
    ]
    report = generate_report(raw, cases, provider="test")
    assert "by_capability" in report
    assert "accuracy" in report["by_capability"]
    assert "tool_selection" in report["by_capability"]
    assert "accuracy_rate" in report["summary"]
    assert "tool_selection_f1" in report["summary"]


def test_extract_tools_order():
    traj = _traj(tools=["get_time", "calculator", "calculator"])
    assert extract_tools("", traj) == ["get_time", "calculator", "calculator"]


def test_normalize_answer():
    assert normalize_answer("最终答案: 1,918") == "1918"
    assert "paris" in normalize_answer("Paris")
