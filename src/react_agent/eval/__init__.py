"""Eval 统一入口

命令行用法：
    python -m react_agent.eval
    python -m react_agent.eval --provider deepseek
    python -m react_agent.eval --dataset capability
    python -m react_agent.eval --dataset capability --capability accuracy
    python -m react_agent.eval --capability all
    python -m react_agent.eval --list

API 用法：
    from react_agent.eval import EvalRunner
    runner = EvalRunner()
    runner.load_dataset("capability")
    runner.run_all()
    runner.print_summary()
    runner.save_report()
"""

import sys
import os
from typing import Optional

from .dataset import (
    load_dataset,
    filter_by_tag,
    filter_by_capability,
    resolve_dataset_path,
    CAPABILITIES,
)
from .runner import run_batch
from .report import generate_report, save_report, list_reports


class EvalRunner:
    """Eval 评测运行器"""

    def __init__(self):
        self.cases = []
        self.raw_results = []
        self.report = {}
        self.provider = None
        self.dataset_path = None

    def load_dataset(
        self,
        path: Optional[str] = None,
        tag: Optional[str] = None,
        capability: Optional[str] = None,
    ):
        """加载测试用例

        path: None/"default"/"capability"/文件路径
        tag: 按 tag 过滤
        capability: 按能力维度过滤（all=所有带 capability 的用例）
        """
        self.dataset_path = resolve_dataset_path(path)
        cases = load_dataset(path)
        if capability:
            cases = filter_by_capability(cases, capability)
            print(
                f"[Eval] 加载 {len(cases)} 条用例"
                f"（capability={capability}, dataset={os.path.basename(self.dataset_path)}）"
            )
        elif tag:
            cases = filter_by_tag(cases, tag)
            print(f"[Eval] 加载 {len(cases)} 条用例（tag={tag}）")
        else:
            tags = sorted({c.tag for c in cases if c.tag})
            caps = sorted({c.capability for c in cases if c.capability})
            print(
                f"[Eval] 加载 {len(cases)} 条用例"
                f"（dataset={os.path.basename(self.dataset_path)}）"
                f" tag={tags}" + (f" capability={caps}" if caps else "")
            )
        self.cases = cases

    def run_all(self, provider: Optional[str] = None, progress: bool = True):
        """运行所有已加载的测试用例"""
        self.provider = provider or os.environ.get("LLM_PROVIDER", "default")

        def _progress(index, total, case_id, status, result):
            if not progress:
                return
            icon = {
                "running": ">",
                "done": "OK",
                "timeout": "TIMEOUT",
                "error": "ERR",
            }.get(status, "?")
            if status == "running":
                print(f"  [{index}/{total}] {icon} {case_id}...")
            elif status in ("done", "timeout", "error"):
                duration = result.get("duration_seconds", 0) if result else 0
                print(f"  [{index}/{total}] {icon} {case_id}  ({duration}s)")

        print(f"\n[Eval] 开始评测: {len(self.cases)} 条用例, provider={self.provider}")
        print(f"[Eval] {'=' * 50}")

        self.raw_results = run_batch(
            self.cases, provider=provider, progress_callback=_progress
        )
        self.report = generate_report(
            self.raw_results, self.cases, provider=self.provider
        )

    def summary(self) -> dict:
        if not self.report:
            return {"total": 0, "passed": 0, "pass_rate": 0}
        return self.report.get("summary", {})

    def save_report(self) -> str:
        return save_report(self.report)

    def print_summary(self):
        s = self.summary()
        if not s.get("total"):
            print("[Eval] 没有评测结果")
            return

        report_id = self.report.get("report_id", "")
        by_tag = self.report.get("by_tag", {})
        by_cap = self.report.get("by_capability", {})

        print(f"\n[Eval] {'=' * 55}")
        print(f"[Eval]  评测报告: {report_id}")
        print(f"[Eval]  Provider: {self.provider}")
        print(f"[Eval] {'=' * 55}")
        print(
            f"  总计: {s['total']} 条  |  通过: {s['passed']}  |  失败: {s['failures']}"
        )
        print(
            f"  通过率: {s['pass_rate']*100:.1f}%  |  评分率: {s.get('score_rate', 0)*100:.1f}%"
        )
        print(
            f"  平均耗时: {s['avg_duration']}s  |  平均步数: {s['avg_steps']}  |  "
            f"平均 tokens: {s['avg_tokens']}"
        )

        # 能力摘要
        cap_lines = []
        if "accuracy_rate" in s:
            cap_lines.append(f"accuracy={s['accuracy_rate']*100:.0f}%")
        if "tool_selection_f1" in s:
            cap_lines.append(f"tool_f1={s['tool_selection_f1']:.2f}")
        if "reasoning_rate" in s:
            cap_lines.append(f"reasoning={s['reasoning_rate']*100:.0f}%")
        if "consistency_rate" in s:
            cap_lines.append(f"consistency={s['consistency_rate']*100:.0f}%")
        if "hallucination_rate" in s:
            cap_lines.append(f"hallucination={s['hallucination_rate']*100:.0f}%")
        if cap_lines:
            print(f"  能力指标: {', '.join(cap_lines)}")

        print(f"  {'-' * 55}")

        if by_cap:
            print("  按 capability 分组:")
            for cap, stats in sorted(by_cap.items()):
                avg = stats.get("avg_metrics", {})
                extra = ""
                if "tool_f1" in avg:
                    extra = f"  F1={avg['tool_f1']:.2f}"
                elif "consistency_rate" in avg:
                    extra = f"  cons={avg['consistency_rate']:.2f}"
                elif "hallucination_rate" in avg:
                    extra = f"  hallu={avg['hallucination_rate']:.2f}"
                print(
                    f"    {cap:<16} {stats['passed']}/{stats['total']}  "
                    f"({stats['pass_rate']*100:.0f}%){extra}"
                )

        if by_tag and not by_cap:
            print("  按 tag 分组:")
            for tag, stats in sorted(by_tag.items()):
                print(
                    f"    {tag:<12} {stats['passed']}/{stats['total']}  "
                    f"({stats['pass_rate']*100:.0f}%)  "
                    f"总耗时: {stats['total_duration']:.0f}s"
                )

        failures = self.report.get("failures", [])
        if failures:
            print(f"  {'-' * 55}")
            print(f"  失败用例 ({len(failures)}):")
            for f in failures[:8]:
                details = f["score"].get("details", {})
                fail_reasons = "; ".join(
                    d.get("reason", "")
                    for dim, d in details.items()
                    if isinstance(d, dict) and not d.get("passed", True)
                )
                print(f"    x {f['case_id']}: {fail_reasons or 'failed'}")
                if f.get("trajectory_file"):
                    print(f"      轨迹: traj_{f['trajectory_file']}.json")
        print(f"[Eval] {'=' * 55}\n")


def main():
    """CLI 入口"""
    provider = None
    tag = None
    dataset = None
    capability = None
    list_only = False

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--provider" and i + 1 < len(args):
            provider = args[i + 1]
            i += 2
        elif args[i] == "--tag" and i + 1 < len(args):
            tag = args[i + 1]
            i += 2
        elif args[i] == "--dataset" and i + 1 < len(args):
            dataset = args[i + 1]
            i += 2
        elif args[i] == "--capability" and i + 1 < len(args):
            capability = args[i + 1]
            i += 2
        elif args[i] == "--list":
            list_only = True
            i += 1
        elif args[i] in ("-h", "--help"):
            print("用法:")
            print("  python -m react_agent.eval [--dataset default|capability|PATH]")
            print("  python -m react_agent.eval --capability accuracy|tool_selection|reasoning|consistency|hallucination|all")
            print("  python -m react_agent.eval --tag tool_local")
            print("  python -m react_agent.eval --provider deepseek")
            print("  python -m react_agent.eval --list")
            print()
            print(f"能力维度: {', '.join(CAPABILITIES)}")
            return
        else:
            i += 1

    if list_only:
        reports = list_reports()
        if not reports:
            print("没有找到评测报告")
            return
        print(f"\n共 {len(reports)} 份评测报告:\n")
        print(f"{'#':<4} {'时间':<20} {'Provider':<14} {'通过率':<8} {'总数':<6} {'失败':<6}")
        print("-" * 70)
        for idx, r in enumerate(reports, 1):
            s = r["summary"]
            print(
                f"{idx:<4} {r['timestamp']:<20} {r['provider']:<14} "
                f"{s.get('pass_rate', 0)*100:.0f}%{'':<4} {s.get('total', 0):<6} "
                f"{s.get('failures', 0):<6}"
            )
        return

    # --capability 默认加载 capability 数据集
    if capability and not dataset:
        dataset = "capability"

    runner = EvalRunner()
    runner.load_dataset(path=dataset, tag=tag, capability=capability)
    if not runner.cases:
        print("[Eval] 没有匹配的测试用例")
        return
    runner.run_all(provider=provider)
    runner.print_summary()
    path = runner.save_report()
    print(f"[Eval] 报告已保存: {path}")


if __name__ == "__main__":
    main()
