"""runner — 批量执行 Agent + 集成 Recorder 记录轨迹

核心逻辑：
    对每一条测试用例，调用 Agent 执行，同时记录轨迹到 trajectories/。
    不直接调 Agent 的循环逻辑——而是复用已有的 subprocess 调用方式（通过 react_loop.py），
    确保 runner 和实际使用环境一致（不是模拟调用）。

数据流：
    for each test_case:
        subprocess(react_loop.py "<question>")  →  stdout + trajectory file
        scorer.score(stdout, trajectory, test_case)  →  Result
    report.aggregate(results)  →  eval/reports/eval_xxx.json
"""

import subprocess
import sys
import os
import time
import json
import glob
from typing import Optional

# 引用 repo 根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REACT_LOOP = os.path.join(BASE_DIR, "react_loop.py")
TRAJECTORY_DIR = os.path.join(BASE_DIR, "trajectories")


def run_single_case(question: str, timeout: int = 60,
                    provider: Optional[str] = None) -> tuple[str, Optional[dict]]:
    """执行单条测试用例

    通过 subprocess 调用 react_loop.py，捕获输出。
    执行完成后从 trajectories/ 中找到最新生成的轨迹文件。

    参数:
        question: 用户问题
        timeout:  超时秒数
        provider: 指定 LLM provider（None=使用默认）

    返回:
        (stdout, trajectory_dict_or_None)
    """
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    if provider:
        env["LLM_PROVIDER"] = provider

    # 记录运行前的轨迹文件列表，用于识别本次运行新生成的
    before_files = _list_traj_files()

    start_time = time.time()
    try:
        result = subprocess.run(
            [sys.executable, REACT_LOOP, question],
            capture_output=True, text=True, timeout=timeout,
            cwd=BASE_DIR,
            env=env,
        )
        stdout = result.stdout.strip()
        exit_code = result.returncode
    except subprocess.TimeoutExpired:
        stdout = f"[Eval] 超时 ({timeout}s)"
        exit_code = -1
    except Exception as e:
        stdout = f"[Eval] 执行异常: {e}"
        exit_code = -2

    duration = round(time.time() - start_time, 2)

    # 找到本次运行新生成的轨迹文件
    after_files = _list_traj_files()
    new_files = [f for f in after_files if f not in before_files]
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
    """返回 trajectories/ 下所有 JSON 文件路径的 set"""
    if not os.path.exists(TRAJECTORY_DIR):
        return set()
    return set(glob.glob(os.path.join(TRAJECTORY_DIR, "traj_*.json")))


def run_batch(cases: list, provider: Optional[str] = None,
              progress_callback=None) -> list:
    """批量运行测试用例

    参数:
        cases: TestCase 列表
        provider: LLM provider
        progress_callback: 可选回调函数，用于报告进度
            def callback(index, total, case_id, status, result_or_error)

    返回:
        list[dict] — 每条用例一个结果字典
    """
    results = []
    total = len(cases)

    for i, case in enumerate(cases):
        case_id = case.id or f"case_{i+1}"
        if progress_callback:
            progress_callback(i + 1, total, case_id, "running", None)

        timeout = case.timeout
        stdout, trajectory, exit_code, duration = run_single_case(
            case.question, timeout=timeout, provider=provider,
        )

        result = {
            "case_id": case_id,
            "question": case.question,
            "stdout": stdout,
            "trajectory": trajectory,
            "exit_code": exit_code,
            "duration_seconds": duration,
            "timed_out": exit_code == -1,
        }

        if progress_callback:
            status = "timeout" if exit_code == -1 else ("error" if exit_code != 0 else "done")
            progress_callback(i + 1, total, case_id, status, result)

        results.append(result)

    return results
