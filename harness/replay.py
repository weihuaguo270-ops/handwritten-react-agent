"""Replay — 轨迹重放

回放 Recorder 模块记录的 JSON 轨迹文件，支持列表、逐步骤、按 ID 查找。

用法:
    python -m harness.replay                         # 列出所有轨迹
    python -m harness.replay --latest                 # 重放最新
    python -m harness.replay --step 1                 # 逐步骤查看
"""

import json
import os
import sys
import glob
from typing import Optional

TRAJECTORY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trajectories")


class Replay:
    """轨迹重放器"""

    def __init__(self, directory: str = ""):
        self.directory = directory or TRAJECTORY_DIR

    def list_recordings(self, directory: str = "") -> list[dict]:
        pattern = os.path.join(directory or self.directory, "traj_*.json")
        files = sorted(glob.glob(pattern), reverse=True)
        result = []
        for i, f in enumerate(files, 1):
            try:
                with open(f, encoding="utf-8") as fh:
                    data = json.load(fh)
                result.append({
                    "index": i,
                    "session_id": data.get("session_id", ""),
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
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)

    def play(self, filepath: str, step_by_step: bool = False):
        data = self.load(filepath)
        self._display(data, step_mode=step_by_step)
        return data

    @staticmethod
    def _display(data: dict, step_mode: bool = False):
        sep = "=" * 65
        print(sep)
        print(f"  🎯 {data.get('query', '（无查询）')}")
        print(sep)
        print(f"  🆔 {data.get('session_id', '')}")
        print(f"  📅 {data.get('timestamp', '')}  |  🤖 {data.get('model', '')}")
        print(f"  📊 {data.get('total_steps', 0)} 步  |  ⏱ {data.get('total_duration_seconds', 0)}s  |  💰 ~{data.get('total_tokens_estimated', 0)} tokens\n")

        sp = data.get("system_prompt_preview", "")
        if sp:
            print(f"  📋 System Prompt 预览:")
            print(f"     {sp[:150].replace(chr(10), chr(10)+'     ')}\n")

        steps = data.get("steps", [])
        if not steps:
            print("  （无步骤记录）")
            return

        for s in steps:
            step_num = s.get("step", "?")
            thought = s.get("thought", "")
            action = s.get("action", {})
            observation = s.get("observation", "")
            duration = s.get("duration_seconds", 0)
            tokens = s.get("tokens_estimated", 0)

            print(f"  ─── Step {step_num} ({duration}s, ~{tokens}t) ───")
            if thought:
                for line in thought.split("\n"):
                    print(f"    💭 {line}")
            if action:
                name = action.get("name", "")
                args = action.get("arguments", "")
                print(f"    🔧 {name}({args[:200]})")
            if observation:
                obs_preview = observation[:300]
                print(f"    📥 {obs_preview}")
                if len(observation) > 300:
                    print(f"       ...（共 {len(observation)} 字符）")
            print()
            if step_mode:
                input("    按 Enter 继续下一步...")

        final = data.get("final_answer", "")
        if final:
            print(sep)
            print(f"  ✅ 最终答案:")
            print(f"     {final[:500]}")
            if len(final) > 500:
                print(f"     ...（共 {len(final)} 字符）")
            print(sep)


# 便捷类名
Replayer = Replay


def main():
    r = Replay()
    recordings = r.list_recordings()

    if not recordings:
        print("没有找到轨迹文件")
        print(f"（请先运行 react_loop.py，轨迹保存在 {TRAJECTORY_DIR}）")
        return

    if len(sys.argv) == 1:
        print(f"\n共 {len(recordings)} 条轨迹:\n")
        print(f"{'#':<4} {'时间':<20} {'模型':<18} {'步数':<6} {'耗时':<8} {'查询'}")
        print("-" * 90)
        for t in recordings:
            print(f"{t['index']:<4} {t['timestamp']:<20} {t['model']:<18} "
                  f"{t['steps']:<6} {t['duration']:<8.1f} {t['query']}")
        print(f"\n用法: python -m harness.replay <编号>")
        print(f"      python -m harness.replay --latest")
        print(f"      python -m harness.replay --step <编号>")
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

    r.play(target["filepath"], step_by_step=step_mode)


if __name__ == "__main__":
    main()
