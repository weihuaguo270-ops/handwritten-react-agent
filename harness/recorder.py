"""Recorder — 轨迹记录

记录 ReAct Loop 每一步的 thought / action / observation / result，
最终写入 JSON 文件，可供 Replay 模块回放。
"""

import json
import os
import time
import random
import string
from typing import Optional

TRAJECTORY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trajectories")


def _generate_session_id() -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    rand = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"{ts}_{rand}"


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


class Trajectory:
    """一次 ReAct 会话的完整轨迹"""

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
        self._step_durations[step] = time.time()

    def add_step(self, step: int, thought: str = "",
                 action_name: str = "", action_args: str = "",
                 observation: str = "", tokens: int = 0):
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
        self._update_step(step, thought=thought[:500])

    def add_tool_call(self, step: int, name: str, arguments: str,
                      result: str, duration: float = 0):
        self._update_step(step,
                          action={"name": name, "arguments": arguments[:300]},
                          observation=result[:500])

    def _update_step(self, step: int, **kwargs):
        for s in self.steps:
            if s["step"] == step:
                if "action" in kwargs and "action" in s:
                    if "actions" not in s:
                        s["actions"] = [s.pop("action")]
                    s["actions"].append(kwargs.pop("action"))
                s.update(kwargs)
                return
        entry = {"step": step}
        entry.update(kwargs)
        self.steps.append(entry)

    def set_final_answer(self, answer: str):
        self.final_answer = answer[:1000] if answer else ""

    def to_dict(self) -> dict:
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
        save_dir = directory or TRAJECTORY_DIR
        _ensure_dir(save_dir)
        filename = f"traj_{self.session_id}.json"
        filepath = os.path.join(save_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return filepath


_current_trajectory: Optional[Trajectory] = None


def start_trajectory(query: str, model: str = "", system_prompt: str = "") -> Trajectory:
    global _current_trajectory
    _current_trajectory = Trajectory(query=query, model=model, system_prompt=system_prompt)
    return _current_trajectory


def current_trajectory() -> Optional[Trajectory]:
    return _current_trajectory


def finish_trajectory(final_answer: str = "") -> Optional[str]:
    global _current_trajectory
    if _current_trajectory is None:
        return None
    _current_trajectory.set_final_answer(final_answer)
    filepath = _current_trajectory.save()
    _current_trajectory = None
    return filepath
