"""runner — 批量执行 Agent + 集成 Recorder 记录轨迹

对每一条测试用例，通过 subprocess 调用 react_loop.py，
并支持 consistency_runs>1 时的重复执行。
"""

import subprocess
import sys
import os
import time
import json
import glob
from typing import Optional

# src/react_agent/ 与项目根（含 llm_config.json / .env）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
REACT_LOOP = os.path.join(BASE_DIR, "react_loop.py")
TRAJECTORY_DIR = os.path.join(BASE_DIR, "trajectories")


def run_single_case(question: str, timeout: int = 60,
                    provider: Optional[str] = None,
                    max_steps: Optional[int] = None) -> tuple:
    """执行单条测试用例

    返回:
        (stdout, trajectory_dict_or_None, exit_code, duration)
    """
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    if provider:
        env["LLM_PROVIDER"] = provider
    # 保证可 import react_agent，并让 llm 能找到项目根配置
    src_dir = os.path.dirname(BASE_DIR)
    env["PYTHONPATH"] = os.pathsep.join(
        [src_dir, env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    # 评测子进程关闭沙箱预热，避免每条用例多一次冷启动
    env["REACT_AGENT_SANDBOX_PREWARM"] = "0"
    # 长跑可靠性：评测默认开 ToolGuard / 自修；步数由用例透传
    env.setdefault("REACT_AGENT_TOOL_GUARD", "1")
    env.setdefault("REACT_AGENT_SELF_REPAIR", "1")
    # 评测默认关 MCP，避免 get_time→get_current_time 等工具名漂移造成假阴性
    env.setdefault("REACT_AGENT_DISABLE_MCP", "1")
    if max_steps is not None:
        env["REACT_AGENT_MAX_STEPS"] = str(int(max_steps))

    before_files = _list_traj_files()

    start_time = time.time()
    try:
        # cwd 用项目根，便于加载 llm_config.json / .env
        cmd = [sys.executable, REACT_LOOP]
        if max_steps is not None:
            cmd.extend(["--max-steps", str(int(max_steps))])
        cmd.append(question)
        result = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace",
            cwd=PROJECT_ROOT if os.path.isdir(PROJECT_ROOT) else BASE_DIR,
            env=env,
        )
        stdout = result.stdout.strip()
        if result.stderr and not stdout:
            stdout = result.stderr.strip()
        exit_code = result.returncode
    except subprocess.TimeoutExpired:
        stdout = f"[Eval] 超时 ({timeout}s)"
        exit_code = -1
    except Exception as e:
        stdout = f"[Eval] 执行异常: {e}"
        exit_code = -2

    duration = round(time.time() - start_time, 2)

    after_files = _list_traj_files()
    new_files = sorted(f for f in after_files if f not in before_files)
    trajectory = None
    if new_files:
        latest_file = new_files[-1]
        try:
            with open(latest_file, encoding="utf-8") as fh:
                trajectory = json.load(fh)
        except (json.JSONDecodeError, OSError):
            pass

    return stdout, trajectory, exit_code, duration


def _list_traj_files() -> set:
    if not os.path.exists(TRAJECTORY_DIR):
        return set()
    return set(glob.glob(os.path.join(TRAJECTORY_DIR, "traj_*.json")))


def run_batch(cases: list, provider: Optional[str] = None,
              progress_callback=None) -> list:
    """批量运行测试用例

    对 consistency_runs > 1 的用例会重复执行，结果写入
    result["run_results"]，供 capability_scorer 计算一致率。
    """
    results = []
    total = len(cases)

    for i, case in enumerate(cases):
        case_id = case.id or f"case_{i+1}"
        if progress_callback:
            progress_callback(i + 1, total, case_id, "running", None)

        timeout = getattr(case, "timeout", 60)
        runs = max(1, int(getattr(case, "consistency_runs", 1) or 1))
        # 仅 consistency capability 强制多跑；其它 capability 若设了 runs 也尊重
        if getattr(case, "capability", None) == "consistency":
            runs = max(runs, 2)

        run_results = []
        total_duration = 0.0
        last_exit = 0
        timed_out = False

        case_max_steps = getattr(case, "max_steps", None)
        for r_i in range(runs):
            stdout, trajectory, exit_code, duration = run_single_case(
                case.question, timeout=timeout, provider=provider,
                max_steps=case_max_steps,
            )
            total_duration += duration
            last_exit = exit_code
            if exit_code == -1:
                timed_out = True
            run_results.append({
                "stdout": stdout,
                "trajectory": trajectory,
                "exit_code": exit_code,
                "duration_seconds": duration,
            })
            if progress_callback and runs > 1:
                progress_callback(
                    i + 1, total, f"{case_id}#{r_i+1}/{runs}",
                    "timeout" if exit_code == -1 else ("error" if exit_code != 0 else "done"),
                    {"duration_seconds": duration},
                )

        primary = run_results[-1]
        result = {
            "case_id": case_id,
            "question": case.question,
            "stdout": primary["stdout"],
            "trajectory": primary["trajectory"],
            "exit_code": last_exit,
            "duration_seconds": round(total_duration, 2),
            "timed_out": timed_out,
            "runs": runs,
        }
        if runs > 1:
            result["run_results"] = run_results

        if progress_callback:
            if timed_out:
                status = "timeout"
            elif last_exit == 0:
                status = "done"
            else:
                status = "error"
            progress_callback(i + 1, total, case_id, status, result)

        results.append(result)

    return results
