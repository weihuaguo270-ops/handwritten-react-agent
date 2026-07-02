"""
轨迹记录器（Harness Engineering）

将 ReAct Loop 的每一步（thought/action/observation）结构化记录到 JSON 文件。
可用于调试、回放、面试展示、token 消耗统计。

记录内容:
  - session_id / 时间戳 / 模型名
  - 每步的 thought、调用的工具名和参数、工具返回结果
  - 最终答案
  - 预估 token 消耗和耗时

每次对话自动保存在 trajectories/ 目录下。
"""

import json
import os
import time
import random
import string
from typing import Optional


TRAJECTORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trajectories")


# ============================================================
# 1. 工具函数
# ============================================================

def _generate_session_id() -> str:
    """生成唯一的会话 ID：时间戳 + 4 位随机字符"""
    ts = time.strftime("%Y%m%d_%H%M%S")
    rand = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"{ts}_{rand}"


def _ensure_dir(path: str):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)


# ============================================================
# 2. Trajectory 数据类
# ============================================================

class Trajectory:
    """一次 ReAct 会话的完整轨迹

    用法:
        traj = Trajectory(query="现在纽约几点", model="deepseek-v4-flash")
        traj.add_step(step=1, thought="需要搜索", action="web_search", arguments="...")
        traj.add_observation(step=1, observation="纽约凌晨 02:47")
        traj.set_final_answer("纽约凌晨 02:47")
        path = traj.save()
    """

    def __init__(self, query: str, model: str = "", system_prompt: str = ""):
        self.session_id = _generate_session_id()
        self.query = query
        self.model = model
        self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.system_prompt = system_prompt[:200] if system_prompt else ""
        self.steps: list[dict] = []
        self.final_answer = ""
        self.total_tokens_estimated = 0
        self._start_time = time.time()
        self._step_durations: dict[int, float] = {}

    def start_step(self, step: int):
        """记录一步的开始时间"""
        self._step_durations[step] = time.time()

    def add_step(self, step: int, thought: str = "",
                 action_name: str = "", action_args: str = "",
                 observation: str = "", tokens: int = 0):
        """添加一步记录"""
        entry = {
            "step": step,
            "thought": thought[:500] if thought else "",
            "duration_seconds": round(time.time() - self._step_durations.get(step, time.time()), 2),
        }
        if action_name:
            entry["action"] = {"name": action_name, "arguments": action_args[:300]}
        if observation:
            entry["observation"] = observation[:500]
        if tokens:
            entry["tokens_estimated"] = tokens
            self.total_tokens_estimated += tokens
        self.steps.append(entry)

    def add_thought(self, step: int, thought: str):
        """单独记录 thought（当步没有调工具时）"""
        self._update_step(step, thought=thought[:500])

    def add_tool_call(self, step: int, name: str, arguments: str,
                      result: str, duration: float = 0):
        """记录工具调用及其结果"""
        self._update_step(step,
                          action={"name": name, "arguments": arguments[:300]},
                          observation=result[:500])

    def _update_step(self, step: int, **kwargs):
        """更新某一步的字段（如果该步已存在则追加工具调用）"""
        for s in self.steps:
            if s["step"] == step:
                # 如果已有 action，说明这一步调了多个工具，追加到 actions 列表
                if "action" in kwargs and "action" in s:
                    if "actions" not in s:
                        s["actions"] = [s.pop("action")]
                    s["actions"].append(kwargs.pop("action"))
                s.update(kwargs)
                return
        # 不存在则新建
        entry = {"step": step}
        entry.update(kwargs)
        self.steps.append(entry)

    def set_final_answer(self, answer: str):
        """设置最终答案"""
        self.final_answer = answer[:1000] if answer else ""

    def to_dict(self) -> dict:
        """导出为字典"""
        duration = round(time.time() - self._start_time, 2)
        return {
            "session_id": self.session_id,
            "query": self.query[:500],
            "model": self.model,
            "timestamp": self.timestamp,
            "system_prompt_preview": self.system_prompt,
            "total_steps": len(self.steps),
            "total_duration_seconds": duration,
            "total_tokens_estimated": self.total_tokens_estimated,
            "final_answer": self.final_answer,
            "steps": self.steps,
        }

    def save(self, directory: Optional[str] = None) -> str:
        """保存轨迹到 JSON 文件

        参数:
            directory: 保存目录，默认 trajectories/

        返回:
            保存的文件路径
        """
        save_dir = directory or TRAJECTORY_DIR
        _ensure_dir(save_dir)
        filename = f"traj_{self.session_id}.json"
        filepath = os.path.join(save_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return filepath


# ============================================================
# 3. 全局实例
# ============================================================

_current_trajectory: Optional[Trajectory] = None


def start_trajectory(query: str, model: str = "", system_prompt: str = "") -> Trajectory:
    """开始记录一次新会话的轨迹"""
    global _current_trajectory
    _current_trajectory = Trajectory(query=query, model=model, system_prompt=system_prompt)
    return _current_trajectory


def current_trajectory() -> Optional[Trajectory]:
    """获取当前正在记录的轨迹"""
    return _current_trajectory


def finish_trajectory(final_answer: str = "") -> Optional[str]:
    """结束当前会话，保存轨迹文件

    返回:
        文件路径，如果没有活跃的会话则返回 None
    """
    global _current_trajectory
    if _current_trajectory is None:
        return None
    _current_trajectory.set_final_answer(final_answer)
    filepath = _current_trajectory.save()
    _current_trajectory = None
    return filepath
