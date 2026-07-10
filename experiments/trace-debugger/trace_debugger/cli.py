"""Trace Debugger CLI"""
from __future__ import annotations
import argparse
import sys
import os

from .reader import load, load_recent
from .analyzer import Analyzer
from .reporter import format_report, format_json


def main():
    """CLI 入口 — 支持四种调用方式：

    tdebug <file>                      复盘单条轨迹
    tdebug replay <file>               逐步骤回放
    tdebug scan <dir>                  扫描目录
    tdebug -h                          帮助
    """
    if len(sys.argv) < 2:
        print("用法: tdebug <轨迹.json> | replay <轨迹.json> | scan <目录>")
        print("      tdebug -h  查看帮助")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "-h" or cmd == "--help":
        _print_help()
    elif cmd == "replay" and len(sys.argv) >= 3:
        _cmd_replay(sys.argv[2])
    elif cmd == "scan" and len(sys.argv) >= 3:
        _cmd_scan(sys.argv[2], int(sys.argv[3]) if len(sys.argv) >= 4 else 5)
    elif cmd.endswith(".json"):
        _cmd_analyze(cmd)
    else:
        print(f"未知命令: {cmd}")
        _print_help()
        sys.exit(1)


def _print_help():
    print("Trace Debugger — Agent 执行轨迹复盘分析工具")
    print()
    print("用法:")
    print("  tdebug <file.json>              复盘单条轨迹")
    print("  tdebug replay <file.json>       逐步骤回放")
    print("  tdebug scan <directory> [N]     扫描最新 N 条轨迹")
    print()
    print("示例:")
    print("  tdebug traj_xxx.json")
    print("  tdebug replay traj_xxx.json")
    print("  tdebug scan ./trajectories/ 10")


def _cmd_analyze(filepath: str):
    if not os.path.exists(filepath):
        print(f"文件不存在: {filepath}")
        sys.exit(1)

    traj = load(filepath)
    analyzer = Analyzer()
    analysis = analyzer.analyze(traj)
    print(format_report(analysis))


def _cmd_replay(filepath: str):
    if not os.path.exists(filepath):
        print(f"文件不存在: {filepath}")
        sys.exit(1)

    traj = load(filepath)

    print("=" * 55)
    print(f"  Trace Debugger — 回放模式")
    print(f"  查询: {traj.query[:80]}")
    print("=" * 55)

    for i, step in enumerate(traj.steps):
        input(f"\n按 Enter 查看 Step {step.index}...")
        print(f"\n--- Step {step.index} ---")
        if step.thought:
            print(f"  [思考] {step.thought[:300]}")
        if step.action_name:
            print(f"  [工具] {step.action_name}")
            if step.action_args:
                print(f"  [参数] {step.action_args[:200]}")
        if step.observation:
            print(f"  [返回] {step.observation[:300]}")
        if step.has_error:
            print(f"  ⚠ [错误] {step.error_message[:200]}")
        print(f"  [耗时] {step.duration:.2f}s")

    print(f"\n最终答案: {traj.final_answer[:200]}")
    print("回放完成。")


def _cmd_scan(directory: str, n: int):
    if not os.path.exists(directory):
        print(f"目录不存在: {directory}")
        sys.exit(1)

    trajs = load_recent(directory, n)
    if not trajs:
        print(f"目录中没有轨迹 JSON 文件: {directory}")
        return

    print("=" * 55)
    print(f"  Trace Debugger — 扫描结果")
    print(f"  目录: {directory}")
    print(f"  最近 {len(trajs)} 条轨迹")
    print("=" * 55)

    for i, traj in enumerate(trajs):
        analyzer = Analyzer()
        analysis = analyzer.analyze(traj)
        icon = "✅" if "无错误" in analysis.overall_assessment else "⚠️"
        print(f"\n  [{i+1}] {icon} {traj.session_id}")
        print(f"      {traj.query[:80]}")
        print(f"      {traj.total_duration:.1f}s / {traj.total_steps} 步 / {traj.model}")
        print(f"      {analysis.overall_assessment}")
