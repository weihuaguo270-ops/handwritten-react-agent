"""Tests for eval report markdown publishing."""
from react_agent.eval.report import report_to_markdown


def test_report_to_markdown_minimal():
    report = {
        "report_id": "eval_test",
        "timestamp": "2026-07-13T00:00:00",
        "provider": "deepseek",
        "summary": {
            "total": 2,
            "passed": 1,
            "pass_rate": 0.5,
            "avg_duration": 1.0,
            "avg_steps": 1,
            "accuracy_rate": 1.0,
        },
        "by_capability": {
            "accuracy": {"total": 1, "passed": 1, "pass_rate": 1.0, "avg_metrics": {}},
            "reasoning": {"total": 1, "passed": 0, "pass_rate": 0.0, "avg_metrics": {}},
        },
        "results": [
            {
                "case_id": "a1",
                "capability": "accuracy",
                "duration_seconds": 1.2,
                "score": {"passed": True},
            },
            {
                "case_id": "r1",
                "capability": "reasoning",
                "duration_seconds": 2.0,
                "score": {"passed": False, "details": {"x": {"passed": False, "reason": "miss"}}},
            },
        ],
        "failures": [
            {
                "case_id": "r1",
                "score": {"details": {"x": {"passed": False, "reason": "miss"}}},
            }
        ],
    }
    md = report_to_markdown(report, dataset_name="capability_dataset.json")
    assert "eval_test" in md
    assert "1/2" in md
    assert "accuracy" in md
    assert "r1" in md
    assert "如何复现" in md
    print("✅ report_to_markdown OK")


if __name__ == "__main__":
    test_report_to_markdown_minimal()
