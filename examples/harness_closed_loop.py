"""One-click closed loop: Agent Harness → Trace Debugger → Eval Engine.

Modes:
  --fixture   Offline golden trajectory (default; CI-safe, no API key)
  --mock-agent  Generate a trajectory via Harness recorder (mock steps)
  --live PATH   Analyze an existing trajectory JSON file

Requires (CI installs these):
  - react-agent (this repo)
  - trace-debugger  (optional at runtime; skip analysis if missing)
  - llm-eval-engine (optional at runtime; skip scoring if missing)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from react_agent.console_io import configure_stdio, safe_print  # noqa: E402
from react_agent.harness.schema import (  # noqa: E402
    TrajectorySchemaError,
    assert_valid,
    normalize_trajectory,
    validate_trajectory,
)
from react_agent.harness import start_trajectory, finish_trajectory  # noqa: E402

configure_stdio()

FIXTURE = os.path.join(
    os.path.dirname(__file__), "fixtures", "harness_closed_loop.json"
)


def _load_fixture(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert_valid(data)
    return normalize_trajectory(data)


def _mock_agent_trajectory(query: str) -> dict:
    """Produce a valid Format B trajectory via the Harness recorder."""
    t = start_trajectory(query, model="mock-gpt")
    t.add_step(
        1,
        thought="Need docs for list.sort",
        action_name="web_search",
        action_args='{"query": "Python list.sort"}',
        observation="list.sort sorts in place, ascending by default",
    )
    t.add_step(
        2,
        thought="Enough to answer",
    )
    path = finish_trajectory(
        "list.sort() sorts a list in place; use sorted() for a new list."
    )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # Recorder may write empty thought steps — ensure schema pass
    issues = validate_trajectory(data)
    if issues:
        raise TrajectorySchemaError("; ".join(issues))
    data = normalize_trajectory(data)
    if os.path.exists(path):
        os.remove(path)
    return data


def _analyze_tdebug(traj: dict) -> dict | None:
    try:
        from trace_debugger import Analyzer
        from trace_debugger.reader import parse as tdebug_parse
        from trace_debugger.analyzer import FailureType
    except ImportError:
        print("  [skip] trace-debugger not installed")
        return None

    parsed = tdebug_parse(traj)
    analysis = Analyzer().analyze(parsed)
    failures = []
    for pa in analysis.paths:
        for ft in pa.failure_types:
            label = FailureType.LABELS.get(ft, ft)
            failures.append({"type": ft, "label": label})
    main_ok = any(pa.success and pa.is_main for pa in analysis.paths) or (
        bool(parsed.final_answer.strip()) and not analysis.needs_fix
    )
    summary = {
        "num_steps": parsed.num_steps,
        "success": main_ok,
        "needs_fix": analysis.needs_fix,
        "failures": failures,
        "tools": parsed.main_path.tools_used if parsed.main_path else [],
        "assessment": analysis.overall_assessment[:200],
    }
    print(f"  steps={summary['num_steps']} success={summary['success']} "
          f"needs_fix={summary['needs_fix']}")
    print(f"  tools={summary['tools']}")
    if failures:
        print(f"  failures={[f['type'] for f in failures]}")
    else:
        print("  failures=[]")
    return summary


def _score_eval(traj: dict) -> dict | None:
    try:
        from eval_engine.core.trajectory_parser import parse_trajectory
        from eval_engine.core.process_reward import ProcessRewardScorer
    except ImportError:
        print("  [skip] llm-eval-engine not installed")
        return None

    dag = parse_trajectory(traj)

    def mock_judge(_prompt: str) -> dict:
        return {
            "overall_score": 4.5,
            "efficiency_score": 4.0,
            "tool_usage_score": 4.5,
            "reasoning": "Reasonable steps (mock judge)",
            "needs_revision": False,
            "strengths": ["goal-directed tool use"],
            "weaknesses": [],
            "details": [],
        }

    scorer = ProcessRewardScorer(judge_fn=mock_judge)
    report = scorer.score_trajectory(dag, fast_mode=True)
    summary = {
        "overall_score": round(float(report.overall_score), 3),
        "num_steps": report.num_steps,
        "pass_rate": report.pass_rate,
    }
    print(f"  score={summary['overall_score']} steps={summary['num_steps']} "
          f"pass_rate={summary['pass_rate']}")
    return summary


def run(traj: dict) -> int:
    print("=== Harness closed-loop demo ===")
    print("Step 0: Schema validate")
    issues = validate_trajectory(traj)
    if issues:
        print("  FAIL:", "; ".join(issues))
        return 1
    print("  OK (Format B, 1-based steps)")

    print("Step 1: Trace Debugger analysis")
    tdebug = _analyze_tdebug(traj)

    print("Step 2: Eval Engine process reward")
    evaluation = _score_eval(traj)

    print("Step 3: Summary")
    out = {
        "session_id": traj.get("session_id"),
        "query": traj.get("query"),
        "schema_ok": True,
        "tdebug": tdebug,
        "eval": evaluation,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    # Soft requirement: at least schema OK; prefer both consumers when installed
    if tdebug is None and evaluation is None:
        print("WARN: neither tdebug nor eval-engine available")
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--fixture",
        nargs="?",
        const=FIXTURE,
        default=None,
        help="Use offline fixture (default if no other mode)",
    )
    group.add_argument(
        "--mock-agent",
        action="store_true",
        help="Generate trajectory via Harness recorder",
    )
    group.add_argument(
        "--live",
        metavar="PATH",
        help="Load an existing trajectory JSON",
    )
    args = parser.parse_args(argv)

    if args.mock_agent:
        traj = _mock_agent_trajectory("How to use Python list.sort?")
    elif args.live:
        traj = _load_fixture(args.live)
    else:
        path = args.fixture or FIXTURE
        print(f"fixture: {path}")
        traj = _load_fixture(path)

    return run(traj)


if __name__ == "__main__":
    raise SystemExit(main())
