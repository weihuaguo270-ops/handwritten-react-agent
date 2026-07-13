"""发布可公开的评测快照（Markdown + 归档 JSON）。

用法：
  # 从已有 JSON 报告生成 docs 快照（无需 API，可复现）
  python examples/publish_eval_snapshot.py --from-report src/react_agent/eval/reports/eval_20260713_140216.json

  # 先跑 capability 再发布（需 DEEPSEEK_API_KEY）
  python examples/publish_eval_snapshot.py --run capability

  # 只跑扩大后的新用例（更快验证）
  python examples/publish_eval_snapshot.py --run capability --only-new
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from react_agent.eval.report import load_report, report_to_markdown  # noqa: E402

NEW_CASE_IDS = {
    "acc_germany_capital",
    "acc_square_15",
    "tool_calc_99_plus_1",
    "reason_three_times",
    "cons_boiling",
    "hallu_wrong_boiling_forbid",
}


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


def publish(
    report_path: Path,
    *,
    dataset_name: str,
    out_stem: str,
    reproduce_cmd: str,
) -> tuple[Path, Path]:
    report = load_report(str(report_path))
    docs = ROOT / "docs"
    snap_dir = docs / "snapshots"
    docs.mkdir(exist_ok=True)
    snap_dir.mkdir(exist_ok=True)

    archived = snap_dir / f"{out_stem}.json"
    shutil.copy2(report_path, archived)

    notes = [
        f"git: `{_git_sha()}`",
        f"source_report: `{report_path.as_posix()}`",
        f"archived_json: `docs/snapshots/{archived.name}`",
        "评分器：capability 用规则打分；功能集用 must_contain/工具检查",
    ]
    md = report_to_markdown(
        report,
        title=f"Capability/Eval 公开快照（{out_stem}）",
        dataset_name=dataset_name,
        extra_notes=notes,
        reproduce_cmd=reproduce_cmd,
    )
    md_path = docs / f"{out_stem}.md"
    md_path.write_text(md, encoding="utf-8")
    return md_path, archived


def run_eval(dataset: str, only_new: bool) -> Path:
    os.environ.setdefault("REACT_AGENT_SKIP_RAG", "1")
    os.environ.setdefault("REACT_AGENT_SANDBOX_PREWARM", "0")

    from react_agent.eval import EvalRunner

    runner = EvalRunner()
    runner.load_dataset(path=dataset)
    if only_new:
        runner.cases = [c for c in runner.cases if c.id in NEW_CASE_IDS]
        print(f"[publish] only-new → {len(runner.cases)} cases")
    if not runner.cases:
        raise SystemExit("没有可运行的用例")
    runner.run_all(provider=os.environ.get("LLM_PROVIDER", "deepseek"))
    runner.print_summary()
    path = Path(runner.save_report())
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="发布评测公开快照")
    parser.add_argument("--from-report", default="", help="已有 eval_*.json 路径")
    parser.add_argument(
        "--run",
        choices=["", "capability", "default"],
        default="",
        help="先跑评测再发布",
    )
    parser.add_argument(
        "--only-new",
        action="store_true",
        help="仅跑 capability 扩容的新用例",
    )
    parser.add_argument(
        "--stem",
        default="",
        help="输出文件名主干，默认 capability_snapshot_YYYYMMDD",
    )
    args = parser.parse_args()

    report_path: Path
    dataset_name: str
    reproduce: str

    if args.run:
        if not (
            os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or (ROOT / ".env").exists()
        ):
            print("缺少 API Key：请配置 .env 或 DEEPSEEK_API_KEY")
            return 2
        # 确保加载 .env
        from react_agent.llm import _load_dotenv

        _load_dotenv(override=True)
        dataset_name = (
            "capability_dataset.json" if args.run == "capability" else "dataset.json"
        )
        reproduce = f"python -m react_agent.eval --dataset {args.run}"
        report_path = run_eval(args.run, only_new=args.only_new and args.run == "capability")
    elif args.from_report:
        report_path = Path(args.from_report)
        if not report_path.is_file():
            report_path = ROOT / args.from_report
        if not report_path.is_file():
            print(f"找不到报告: {args.from_report}")
            return 1
        # 启发式：有 by_capability 则当 capability
        data = json.loads(report_path.read_text(encoding="utf-8"))
        if data.get("by_capability"):
            dataset_name = "capability_dataset.json"
            reproduce = "python -m react_agent.eval --dataset capability"
        else:
            dataset_name = "dataset.json"
            reproduce = "python -m react_agent.eval"
    else:
        parser.print_help()
        return 1

    stem = args.stem or f"capability_snapshot_{datetime.now().strftime('%Y%m%d')}"
    md_path, archived = publish(
        report_path,
        dataset_name=dataset_name,
        out_stem=stem,
        reproduce_cmd=reproduce,
    )
    print(f"Markdown: {md_path}")
    print(f"Archived: {archived}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
