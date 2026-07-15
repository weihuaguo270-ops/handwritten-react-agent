"""Reliability harness smoke test"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

os.environ["REACT_AGENT_SKIP_RAG"] = "1"

from examples.run_reliability_harness import run_harness  # noqa: E402


def test_reliability_harness_all_pass():
    report = run_harness()
    assert report["summary"]["total"] == 4
    assert report["summary"]["passed"] == 4, report
