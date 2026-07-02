"""
轨迹重放器 — 回放 Harness 记录的 ReAct 轨迹

用法:
    python replay.py                          # 列出所有轨迹
    python replay.py <session_id 或编号>       # 重放指定轨迹
    python replay.py --latest                 # 重放最新一条
    python replay.py --step                   # 逐步骤模式（按回车一步步看）
"""

import json
import os
import sys
import glob

# ============================================================
# 1. 查找轨迹文件
# ============================================================

TRAJECTORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trajectories")


def list_trajectories() -> list[dict]:
    """列出所有轨迹文件，按时间倒序"""
    pattern = os.path.join(TRAJECTORY_DIR, "traj_*.json")
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


def load_trajectory(filepath: str) -> dict:
    """加载单个轨迹文件"""
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 2. 展示函数
# ============================================================

SEP = "=" * 65


def show_trajectory(data: dict, step_mode: bool = False):
    """完整展示一条轨迹"""
    # 标题
    print(SEP)
    print(f"  🎯 {data.get('query', '（无查询）')}")
    print(SEP)

    # 元信息
    print(f"  🆔 {data.get('session_id', '')}")
    print(f"  📅 {data.get('timestamp', '')}  |  🤖 {data.get('model', '')}")
    print(f"  📊 {data.get('total_steps', 0)} 步  |  ⏱ {data.get('total_duration_seconds', 0)}s  |  💰 ~{data.get('total_tokens_estimated', 0)} tokens")
    print()

    # system prompt 预览
    sp = data.get("system_prompt_preview", "")
    if sp:
        print(f"  📋 System Prompt 预览:")
        print(f"     {sp[:150].replace(chr(10), chr(10)+'     ')}")
        print()

    # 逐步骤展示
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

        # Thought
        if thought:
            # 缩进展示，保持可读性
            for line in thought.split("\n"):
                print(f"    💭 {line}")

        # Action
        if action:
            name = action.get("name", "")
            args = action.get("arguments", "")
            print(f"    🔧 {name}({args[:200]})")

        # Observation
        if observation:
            obs_preview = observation[:300]
            print(f"    📥 {obs_preview}")
            if len(observation) > 300:
                print(f"       ...（共 {len(observation)} 字符）")

        print()

        if step_mode:
            input("    按 Enter 继续下一步...")

    # 最终答案
    final = data.get("final_answer", "")
    if final:
        print(SEP)
        print(f"  ✅ 最终答案:")
        print(f"     {final[:500]}")
        if len(final) > 500:
            print(f"     ...（共 {len(final)} 字符）")
        print(SEP)


# ============================================================
# 3. 主入口
# ============================================================

def main():
    trajectories = list_trajectories()

    if not trajectories:
        print("没有找到轨迹文件")
        print(f"（请先运行 react_loop.py，轨迹保存在 {TRAJECTORY_DIR}）")
        return

    # 没有参数 → 列出所有
    if len(sys.argv) == 1:
        print(f"\n共 {len(trajectories)} 条轨迹:\n")
        print(f"{'#':<4} {'时间':<20} {'模型':<18} {'步数':<6} {'耗时':<8} {'查询'}")
        print("-" * 90)
        for t in trajectories:
            print(f"{t['index']:<4} {t['timestamp']:<20} {t['model']:<18} "
                  f"{t['steps']:<6} {t['duration']:<8.1f} {t['query']}")
        print(f"\n用法: python replay.py <编号>  # 查看具体轨迹")
        print(f"      python replay.py --latest  # 最新一条")
        print(f"      python replay.py --step <编号>  # 逐步骤查看")
        return

    # 解析参数
    step_mode = "--step" in sys.argv
    target = None
    for arg in sys.argv[1:]:
        if arg == "--latest":
            target = trajectories[0] if trajectories else None
        elif arg == "--step":
            continue
        elif arg.isdigit():
            idx = int(arg)
            if 1 <= idx <= len(trajectories):
                target = trajectories[idx - 1]
        else:
            # 按 session_id 匹配
            for t in trajectories:
                if t["session_id"] == arg:
                    target = t
                    break

    if target is None:
        print(f"未找到匹配的轨迹，可用编号 1-{len(trajectories)}")
        return

    data = load_trajectory(target["filepath"])
    show_trajectory(data, step_mode=step_mode)


if __name__ == "__main__":
    main()
