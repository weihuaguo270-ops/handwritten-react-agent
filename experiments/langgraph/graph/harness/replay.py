"""Replay — LangGraph 版轨迹重放

与手写版 replay.py 共享同一个 trajectories/ 目录，
所以无论轨迹是用手写版还是 LangGraph 版记录的，都能回放。

用法：
    # 终端命令
    python -m graph.harness.replay              # 列出所有轨迹
    python -m graph.harness.replay --latest      # 回放最新
    python -m graph.harness.replay 1             # 回放编号 1
    python -m graph.harness.replay --step 1      # 步进模式回放编号 1

    # 代码调用
    from harness.replay import Replay
    r = Replay()
    recordings = r.list_recordings()
    r.play(recordings[0]["filepath"], step_by_step=True)
"""

import json
import os
import sys
import glob
from typing import Optional

# 和手写版共享 trajectories/ 目录
TRAJECTORY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "trajectories",
)


class Replay:
    """轨迹重放器

    可以列出手写版和 LangGraph 版记录的所有轨迹，统一回放。
    轨迹内容结构一致（session_id, query, steps[]），不关心来源。
    """

    def __init__(self, directory: str = ""):
        self.directory = directory or TRAJECTORY_DIR

    def list_recordings(self, directory: str = "") -> list[dict]:
        """列出所有轨迹文件

        返回按时间倒序排列的轨迹摘要列表。
        """
        search_dir = directory or self.directory
        os.makedirs(search_dir, exist_ok=True)
        pattern = os.path.join(search_dir, "traj_*.json")
        files = sorted(glob.glob(pattern), reverse=True)
        result = []
        for i, f in enumerate(files, 1):
            try:
                with open(f, encoding="utf-8") as fh:
                    data = json.load(fh)
                result.append({
                    "index": i,
                    "session_id": data.get("session_id", ""),
                    "source": data.get("source", ""),
                    "query": data.get("query", "")[:80],
                    "steps": data.get("total_steps", 0),
                    "duration": data.get("total_duration_seconds", 0),
                    "tokens": data.get("total_tokens_estimated", 0),
                    "model": data.get("model", ""),
                    "timestamp": data.get("timestamp", ""),
                    "filepath": f,
                })
            except (json.JSONDecodeError, OSError):
                continue
        return result

    def load(self, filepath: str) -> dict:
        """加载一条完整的轨迹数据"""
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)

    def play(self, filepath: str, step_by_step: bool = False):
        """回放一条轨迹

        参数:
            filepath: 轨迹 JSON 文件路径
            step_by_step: 如果为 True，每步暂停等待用户按 Enter

        返回:
            完整的轨迹字典
        """
        data = self.load(filepath)
        self._display(data, step_mode=step_by_step)
        return data

    @staticmethod
    def _display(data: dict, step_mode: bool = False):
        """格式化打印轨迹内容"""
        separator = "=" * 68
        source = data.get("source", "")

        print(separator)
        print(f"  🎯 {data.get('query', '（无查询）')}")
        print(separator)
        print(f"  🆔 {data.get('session_id', '')}  |  来源: {source}")
        print(f"  📅 {data.get('timestamp', '')}  |  🤖 {data.get('model', '')}")
        print(f"  📊 {data.get('total_steps', 0)} 步  |  ⏱ {data.get('total_duration_seconds', 0)}s  |  💰 ~{data.get('total_tokens_estimated', 0)} tokens")
        print()

        system_prompt = data.get("system_prompt_preview", "")
        if system_prompt:
            print(f"  📋 System Prompt 预览:")
            for line in system_prompt.split("\n"):
                print(f"     {line}")
            print()

        steps = data.get("steps", [])
        if not steps:
            print("  （无步骤记录）")
            return

        for step_index, step in enumerate(steps):
            step_number = step.get("step", "?")
            thought = step.get("thought", "")
            actions = step.get("actions", [])
            duration = step.get("duration_seconds", 0)
            tokens = step.get("tokens_estimated", 0)
            tool_call_count = step.get("tool_call_count", len(actions))

            print(f"  ─── Step {step_number} ({duration}s, ~{tokens}t, {tool_call_count} 工具调用) ───")

            if thought:
                for line in thought.split("\n"):
                    print(f"    💭 {line}")

            if actions:
                for action_index, action in enumerate(actions, 1):
                    action_name = action.get("name", "")
                    action_args = action.get("arguments", "")
                    observation = action.get("observation", "")
                    action_duration = action.get("duration_seconds", 0)
                    action_tokens = action.get("tokens_estimated", 0)

                    args_preview = action_args[:300]
                    obs_preview = observation[:500]

                    print(f"    🔧 [{action_index}] {action_name}({args_preview})")
                    print(f"       ⏱ {action_duration}s  |  ~{action_tokens}t")
                    if obs_preview:
                        print(f"       📥 {obs_preview}")
                        if len(observation) > 500:
                            print(f"          ...（共 {len(observation)} 字符）")
            print()

            if step_mode and step_index < len(steps) - 1:
                try:
                    input("    按 Enter 继续下一步...")
                except (EOFError, KeyboardInterrupt):
                    return

        final_answer = data.get("final_answer", "")
        if final_answer:
            print(separator)
            print(f"  ✅ 最终答案:")
            print(f"     {final_answer[:800]}")
            if len(final_answer) > 800:
                print(f"     ...（共 {len(final_answer)} 字符）")
            print(separator)


def main():
    """CLI 入口：python -m graph.harness.replay"""
    replay = Replay()
    recordings = replay.list_recordings()

    if not recordings:
        print("没有找到轨迹文件")
        print(f"（请先运行 graph/main.py，轨迹保存在 {TRAJECTORY_DIR}）")
        return

    if len(sys.argv) == 1:
        print(f"\n共 {len(recordings)} 条轨迹:\n")
        print(f"{'#':<4} {'来源':<8} {'时间':<20} {'模型':<18} {'步数':<6} {'耗时':<8} {'查询'}")
        print("-" * 90)
        for t in recordings:
            print(f"{t['index']:<4} {t['source']:<8} {t['timestamp']:<20} "
                  f"{t['model']:<18} {t['steps']:<6} {t['duration']:<8.1f} {t['query']}")
        print(f"\n用法: python -m graph.harness.replay <编号>")
        print(f"      python -m graph.harness.replay --latest")
        print(f"      python -m graph.harness.replay --step <编号>")
        return

    step_mode = "--step" in sys.argv
    target = None
    for arg in sys.argv[1:]:
        if arg == "--latest":
            target = recordings[0] if recordings else None
        elif arg == "--step":
            continue
        elif arg.isdigit():
            idx = int(arg)
            if 1 <= idx <= len(recordings):
                target = recordings[idx - 1]
        else:
            for t in recordings:
                if t["session_id"] == arg:
                    target = t
                    break

    if target is None:
        print(f"未找到匹配的轨迹，可用编号 1-{len(recordings)}")
        return

    replay.play(target["filepath"], step_by_step=step_mode)


if __name__ == "__main__":
    main()
