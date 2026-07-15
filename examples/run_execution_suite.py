"""跑 Execution 任务集（offline_tools / agent）并可选发布 docs 快照。

用法：
  python examples/run_execution_suite.py
  python examples/run_execution_suite.py --modes agent --publish
  python examples/run_execution_suite.py --modes offline_tools,agent --stem execution_all_YYYYMMDD
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# 加载 .env（API Key）
try:
    from react_agent.llm import _load_dotenv
    _load_dotenv()
except Exception:
    pass

from react_agent.eval.execution_scorer import (  # noqa: E402
    report_to_markdown,
    run_execution_suite,
)


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description="Execution suite (offline + agent)")
    parser.add_argument("--dataset", default=None)
    parser.add_argument(
        "--modes",
        default="offline_tools",
        help="逗号分隔: offline_tools,agent（默认仅 offline_tools）",
    )
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--stem", default=None)
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if "agent" in modes:
        import os
        os.environ.setdefault("REACT_AGENT_SKIP_RAG", "1")
        os.environ.setdefault("REACT_AGENT_SANDBOX_PREWARM", "0")
        os.environ.setdefault("REACT_AGENT_DISABLE_MCP", "1")
        os.environ.setdefault("LLM_PROVIDER", "deepseek")
        if not (os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")):
            print("ERROR: agent 模式需要 DEEPSEEK_API_KEY 或 OPENAI_API_KEY", file=sys.stderr)
            return 2

    report = run_execution_suite(args.dataset, modes=modes)
    s = report["summary"]
    print("=" * 55)
    print(f"  Execution Suite modes={modes}")
    print(f"  {s['passed']}/{s['total']}  pass_rate={s['pass_rate']}%")
    print("=" * 55)
    for r in report["results"]:
        icon = "OK" if r["passed"] else ("SKIP" if r.get("skipped") else "FAIL")
        print(f"  [{icon}] {r['id']} ({r.get('mode')}): {r.get('reason', '')}")

    if args.publish:
        default_stem = "execution_snapshot_" + datetime.now().strftime("%Y%m%d")
        if modes == ["agent"]:
            default_stem = "execution_agent_snapshot_" + datetime.now().strftime("%Y%m%d")
        stem = args.stem or default_stem
        docs = ROOT / "docs"
        snap_dir = docs / "snapshots"
        docs.mkdir(exist_ok=True)
        snap_dir.mkdir(exist_ok=True)
        json_path = snap_dir / f"{stem}.json"
        md_path = docs / f"{stem}.md"
        report["meta"] = {
            "git": _git_sha(),
            "reproduce_cmd": (
                f"python examples/run_execution_suite.py --modes {','.join(modes)} --publish"
            ),
        }
        json_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        notes = [
            f"git: `{_git_sha()}`",
            f"archived_json: `docs/snapshots/{json_path.name}`",
            f"reproduce: `python examples/run_execution_suite.py --modes {','.join(modes)} --publish`",
        ]
        md = report_to_markdown(report, title=f"Execution 公开快照（{stem}）")
        parts = md.split("\n", 2)
        extra = "\n".join(f"- {n}" for n in notes) + "\n"
        if len(parts) >= 3:
            md = parts[0] + "\n\n" + extra + parts[2]
        else:
            md = md + "\n" + extra
        md_path.write_text(md, encoding="utf-8")
        print(f"\nPublished:\n  {md_path}\n  {json_path}")

    return 0 if s.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
