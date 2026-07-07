"""Eval 统一入口

命令行用法：
    python -m eval                        # 运行全部测试（默认 provider）
    python -m eval --provider openai      # 使用 OpenAI 运行
    python -m eval --tag local            # 只运行本地工具类测试
    python -m eval --list                 # 列出历史报告

API 用法：
    from eval import EvalRunner
    runner = EvalRunner()
    runner.load_dataset("eval/dataset.json")
    runner.run_all()            # 执行所有用例
    report = runner.report      # 获取报告
    runner.save_report()        # 保存
"""

import sys
import os
from typing import Optional

from .dataset import load_dataset, filter_by_tag
from .runner import run_batch
from .report import generate_report, save_report, list_reports


class EvalRunner:
    """Eval 评测运行器

    用法：
        runner = EvalRunner()
        runner.load_dataset()
        runner.run_all(provider="deepseek")
        print(runner.summary())
        path = runner.save_report()
    """

    def __init__(self):
        self.cases = []
        self.raw_results = []
        self.report = {}
        self.provider = None

    def load_dataset(self, path: Optional[str] = None, tag: Optional[str] = None):
        """加载测试用例

        参数:
            path: dataset.json 路径，默认使用 eval/dataset.json
            tag: 可选，只加载特定 tag 的用例
        """
        cases = load_dataset(path)
        if tag:
            self.cases = filter_by_tag(cases, tag)
            print(f"[Eval] 加载 {len(self.cases)} 条用例（tag={tag}，共 {len(cases)} 条可用）")
        else:
            self.cases = cases
            tags = set(c.tag for c in cases if c.tag)
            print(f"[Eval] 加载 {len(cases)} 条用例，tag: {sorted(tags)}")

    def run_all(self, provider: Optional[str] = None, progress: bool = True):
        """运行所有已加载的测试用例

        参数:
            provider: LLM provider
            progress: 是否实时打印进度
        """
        self.provider = provider or os.environ.get("LLM_PROVIDER", "default")

        def _progress(index, total, case_id, status, result):
            if not progress:
                return
            icon = {"running": "▶", "done": "✓", "timeout": "⏰", "error": "✗"}.get(status, "?")
            if status == "running":
                print(f"  [{index}/{total}] {icon} {case_id}...")
            elif status in ("done", "timeout", "error"):
                duration = result.get("duration_seconds", 0) if result else 0
                print(f"  [{index}/{total}] {icon} {case_id}  ({duration}s)")

        print(f"\n[Eval] 开始评测: {len(self.cases)} 条用例, provider={self.provider}")
        print(f"[Eval] {'=' * 50}")

        self.raw_results = run_batch(self.cases, provider=provider,
                                     progress_callback=_progress)

        self.report = generate_report(self.raw_results, self.cases,
                                      provider=self.provider)

    def summary(self) -> dict:
        """返回评测摘要"""
        if not self.report:
            return {"total": 0, "passed": 0, "pass_rate": 0}
        return self.report.get("summary", {})

    def save_report(self) -> str:
        """保存报告到 eval/reports/，返回文件路径"""
        return save_report(self.report)

    def print_summary(self):
        """打印评测摘要"""
        s = self.summary()
        if not s.get("total"):
            print("[Eval] 没有评测结果")
            return

        report_id = self.report.get("report_id", "")
        by_tag = self.report.get("by_tag", {})

        print(f"\n[Eval] {'=' * 55}")
        print(f"[Eval]  评测报告: {report_id}")
        print(f"[Eval]  Provider: {self.provider}")
        print(f"[Eval] {'=' * 55}")
        print(f"  总计: {s['total']} 条  |  通过: {s['passed']}  |  失败: {s['failures']}")
        print(f"  通过率: {s['pass_rate']*100:.1f}%  |  评分率: {s['score_rate']*100:.1f}%")
        print(f"  平均耗时: {s['avg_duration']}s  |  平均步数: {s['avg_steps']}  |  平均 tokens: {s['avg_tokens']}")
        print(f"  {'─' * 55}")

        if by_tag:
            print(f"  按 tag 分组:")
            for tag, stats in sorted(by_tag.items()):
                print(f"    {tag:<12} {stats['passed']}/{stats['total']}  ({stats['pass_rate']*100:.0f}%)  "
                      f"  总耗时: {stats['total_duration']:.0f}s")

        failures = self.report.get("failures", [])
        if failures:
            print(f"  {'─' * 55}")
            print(f"  失败用例 ({len(failures)}):")
            for f in failures[:5]:
                fail_reasons = "; ".join(
                    d.get("reason", "")
                    for dim, d in f["score"].get("details", {}).items()
                    if not d.get("passed")
                )
                if f.get("trajectory_file"):
                    print(f"    ✗ {f['case_id']}: {fail_reasons}")
                    print(f"      轨迹: traj_{f['trajectory_file']}.json")
                else:
                    print(f"    ✗ {f['case_id']}: {fail_reasons}")
        print(f"[Eval] {'=' * 55}\n")


def main():
    """CLI 入口"""
    # 解析参数
    provider = None
    tag = None
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
        elif args[i] == "--list":
            list_only = True
            i += 1
        else:
            i += 1

    if list_only:
        reports = list_reports()
        if not reports:
            print("没有找到评测报告")
            return
        print(f"\n共 {len(reports)} 份评测报告:\n")
        print(f"{'#':<4} {'时间':<20} {'Provider':<14} {'通过率':<8} {'总数':<6} {'失败':<6} {'平均耗时'}")
        print("-" * 75)
        for i, r in enumerate(reports, 1):
            s = r["summary"]
            print(f"{i:<4} {r['timestamp']:<20} {r['provider']:<14} "
                  f"{s['pass_rate']*100:.0f}%{'':<4} {s['total']:<6} "
                  f"{s['failures']:<6} {s['avg_duration']}s")
        return

    runner = EvalRunner()
    runner.load_dataset(tag=tag)
    if not runner.cases:
        print("[Eval] 没有匹配的测试用例")
        return
    runner.run_all(provider=provider)
    runner.print_summary()
    path = runner.save_report()
    print(f"[Eval] 报告已保存: {path}")


if __name__ == "__main__":
    main()
