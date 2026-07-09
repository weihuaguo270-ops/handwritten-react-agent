"""Recorder — LangGraph 版本轨迹记录器

专门为 LangGraph 的 StateGraph 执行模式设计。
LangGraph 的节点函数是黑盒，Harness 做的是「在节点执行前后插桩」。

记录流程：
  1. 调用 start_trajectory(query, model, system_prompt) → 初始化
  2. Agent 每走一步，记录 thought（LLM 回复）和 action（工具调用）
  3. 调用 finish(final_answer) → 标记结束
  4. 调用 save() → 写 JSON 文件

与手写版 recorder.py 的区别：
  - 手写版：直接的 Python 过程式调用（react_loop 中一步步记录）
  - 本版：设计为被 LangGraph 节点函数调用，节点函数通过 config 传 harness 实例
"""

import json
import os
import time
import random
import string
from typing import Optional

# 轨迹文件存到 repo/trajectories/（和手写版共享目录）
TRAJECTORY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "trajectories",
)


def _generate_session_id() -> str:
    """生成唯一会话 ID，如 '20260707_143022_xk9m'"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    random_suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"{timestamp}_{random_suffix}"


def _ensure_directory(path: str):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)


class TrajectoryRecorder:
    """一次 LangGraph Agent 会话的完整轨迹记录器

    用法：
        recorder = TrajectoryRecorder(query="...", model="...")
        recorder.record_thought(step=1, thought="用户想问什么", tokens=100)
        recorder.record_action(step=1, action_name="web_search", ...)
        recorder.set_final_answer("答案是42")
        filepath = recorder.save()
    """

    def __init__(self, query: str, model: str = "", system_prompt: str = ""):
        self.session_id = _generate_session_id()
        self.query = query
        self.model = model
        self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.system_prompt = system_prompt[:500] if system_prompt else ""
        self.steps: list[dict] = []
        self.final_answer = ""
        self.total_tokens_estimated = 0
        self._source = "graph"  # 标记来自 LangGraph 版本

        # 当前正在记录的 step（一个 step = thought + 0~N 次工具调用）
        self._current_step: int = 0
        self._current_thought: str = ""
        self._current_thought_tokens: int = 0
        self._current_actions: list[dict] = []
        self._step_start_time: float = time.time()
        self._action_count: int = 0

    # ── 记录接口 ──

    def record_thought(self, step: int, thought: str = "", tokens: int = 0):
        """记录 LLM 的一次回复（thought）

        如果一个 step 有多次 LLM 调用（tool_calls → 继续），
        后续的 thought 会追加到当前 step。
        """
        if step != self._current_step:
            # 新 step 开始，先提交上一步
            self._commit_current_step()
            self._current_step = step
            self._step_start_time = time.time()
            self._current_actions = []
            self._action_count = 0

        self._current_thought = thought[:1000] if thought else ""
        self._current_thought_tokens = tokens
        self.total_tokens_estimated += tokens

    def record_action(self, step: int, action_name: str = "",
                      action_args: str = "", observation: str = "",
                      duration_seconds: float = 0, tokens: int = 0):
        """记录一次工具调用（action → observation）

        如果一个 tool_call 返回了多个 ToolMessage（一个 tool 对应一个），
        多次调用 record_action 会追加到同一个 step 的 actions 列表中。
        """
        if step != self._current_step:
            # 工具调用与当前 step 不匹配时，自动同步
            self._commit_current_step()
            self._current_step = step
            self._step_start_time = time.time()
            self._current_actions = []
            self._action_count = 0

        self._current_actions.append({
            "name": action_name,
            "arguments": action_args[:500] if action_args else "",
            "observation": observation[:1000] if observation else "",
            "duration_seconds": round(duration_seconds, 3),
            "tokens_estimated": tokens,
        })
        self._action_count += 1
        self.total_tokens_estimated += tokens

    def set_final_answer(self, answer: str):
        """设置最终答案"""
        self.final_answer = answer[:2000] if answer else ""

    # ── 内部方法 ──

    def _commit_current_step(self):
        """将当前缓存的 step 提交为最终的 step 条目"""
        if self._current_step == 0:
            return

        step_entry = {
            "step": self._current_step,
            "duration_seconds": round(time.time() - self._step_start_time, 3),
            "thought": self._current_thought,
            "tokens_estimated": self._current_thought_tokens,
        }
        if self._current_actions:
            step_entry["actions"] = self._current_actions
            step_entry["tool_call_count"] = self._action_count

        self.steps.append(step_entry)

    def to_dict(self) -> dict:
        """导出为可序列化的字典"""
        self._commit_current_step()  # 确保最后一步也被提交
        total_duration = round(time.time() - self._step_start_time, 2) if self.steps else 0

        return {
            "session_id": self.session_id,
            "source": self._source,
            "query": self.query[:500],
            "model": self.model,
            "timestamp": self.timestamp,
            "system_prompt_preview": self.system_prompt,
            "total_steps": len(self.steps),
            "total_duration_seconds": total_duration,
            "total_tokens_estimated": self.total_tokens_estimated,
            "final_answer": self.final_answer,
            "steps": self.steps,
        }

    def save(self, directory: Optional[str] = None) -> str:
        """将轨迹写入 JSON 文件，返回文件路径"""
        save_dir = directory or TRAJECTORY_DIR
        _ensure_directory(save_dir)
        filename = f"traj_{self.session_id}.json"
        filepath = os.path.join(save_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return filepath
