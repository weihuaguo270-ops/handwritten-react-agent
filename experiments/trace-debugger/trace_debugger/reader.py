"""reader — 读取并解析 Harness 轨迹 JSON

支持 Handwritten Agent 和 LangGraph 版两种轨迹格式。
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Step:
    """轨迹中的单步"""
    index: int
    thought: str = ""
    action_name: str = ""
    action_args: str = ""
    observation: str = ""
    duration: float = 0.0
    tokens: int = 0
    has_error: bool = False          # 工具调用是否报错
    error_message: str = ""          # 错误信息

    @property
    def is_action(self) -> bool:
        return bool(self.action_name)

    @property
    def is_final(self) -> bool:
        return "FINAL ANSWER" in self.thought.upper()

    @property
    def is_thought(self) -> bool:
        return bool(self.thought.strip()) and not self.is_final

    @property
    def summary(self) -> str:
        if self.is_final:
            return f"输出答案: {self.thought[:80]}"
        if self.is_action:
            return f"调工具: {self.action_name}({self.action_args[:60]})"
        if self.thought:
            return f"思考: {self.thought[:80]}"
        return f"步骤 {self.index}"


@dataclass
class Path:
    """Agent 执行的一条路径

    一条路径 = 从开始到 FINAL ANSWER（或终止）的连续步骤。
    简单任务只有 1 条路径，复杂任务（如 ToT 多次推理）可能有多条。
    """
    steps: list[Step] = field(default_factory=list)
    success: bool = False
    final_answer: str = ""
    is_main_path: bool = False       # 是否最终输出答案的路径

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def tools_used(self) -> list[str]:
        return [s.action_name for s in self.steps if s.action_name]

    @property
    def has_errors(self) -> bool:
        return any(s.has_error for s in self.steps)

    @property
    def error_summary(self) -> list[str]:
        return [s.error_message for s in self.steps if s.has_error]


@dataclass
class Trajectory:
    """完整的执行轨迹"""
    session_id: str
    query: str
    model: str
    timestamp: str
    steps: list[Step]
    paths: list[Path]
    final_answer: str
    total_duration: float
    metadata: dict = field(default_factory=dict)

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def num_paths(self) -> int:
        return len(self.paths)

    @property
    def main_path(self) -> Optional[Path]:
        for p in self.paths:
            if p.is_main_path:
                return p
        return self.paths[-1] if self.paths else None

    @property
    def failed_paths(self) -> list[Path]:
        return [p for p in self.paths if not p.success and not p.is_main_path]


def load(filepath: str) -> Trajectory:
    """从 JSON 文件加载轨迹

    参数:
        filepath: JSON 文件路径

    返回:
        Trajectory
    """
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    return parse(data)


def parse(data: dict) -> Trajectory:
    """解析轨迹字典为 Trajectory 对象"""
    raw_steps = data.get("steps", [])
    steps = []

    for s in raw_steps:
        step_num = s.get("step", 0)
        thought = s.get("thought", "")
        action = s.get("action", {}) or {}
        observation = s.get("observation", "")
        duration = s.get("duration_seconds", 0.0)
        tokens = s.get("tokens_estimated", 0)

        action_name = action.get("name", "") if isinstance(action, dict) else ""
        action_args = action.get("arguments", action.get("args", "")) if isinstance(action, dict) else ""

        # 参数截断：Python 代码只保留关键行，其他参数截断前 80 字符
        action_args_display = str(action_args)
        if len(action_args_display) > 100:
            if action_name in ("execute_python", "execute_command", "run_code"):
                # 代码类参数：只显示前 2 行 + 行数
                lines = action_args_display.split("\\n")
                short = "\\n".join(lines[:2])
                action_args_display = f"{short}\\n... ({len(lines)} lines total)"
            else:
                action_args_display = action_args_display[:80] + "..."

        # 检测错误
        has_error = False
        error_msg = ""
        if observation:
            obs_lower = observation.lower()
            if any(kw in obs_lower for kw in ["error", "异常", "traceback", "stderr"]):
                has_error = True
                # 错误消息精简
                lines = observation.strip().split("\n")
                meaningful = [l for l in lines if l.strip()]
                if meaningful:
                    error_msg = meaningful[0][:150]
                else:
                    error_msg = observation[:150]

        steps.append(Step(
            index=step_num,
            thought=thought,
            action_name=action_name,
            action_args=action_args_display,
            observation=observation[:300],
            duration=duration or 0.0,
            tokens=tokens,
            has_error=has_error,
            error_message=error_msg,
        ))

    # 路径划分：当前 Harness 格式是连续记录，暂按单路径处理
    # 后续扩展：当 ToT/orchestrator 产生多条路径时，根据特征切分
    paths = [Path(steps=steps, success=True, is_main_path=True,
                  final_answer=data.get("final_answer", ""))]

    return Trajectory(
        session_id=data.get("session_id", ""),
        query=data.get("query", ""),
        model=data.get("model", ""),
        timestamp=data.get("timestamp", ""),
        steps=steps,
        paths=paths,
        final_answer=data.get("final_answer", ""),
        total_duration=data.get("total_duration_seconds", 0.0),
        metadata={
            "system_prompt_preview": data.get("system_prompt_preview", ""),
            "total_tokens_estimated": data.get("total_tokens_estimated", 0),
        },
    )


def load_recent(directory: str, n: int = 5) -> list[Trajectory]:
    """加载最近的 N 条轨迹

    参数:
        directory: 轨迹目录
        n: 加载条数

    返回:
        list[Trajectory]
    """
    if not os.path.exists(directory):
        return []
    files = sorted(
        [f for f in os.listdir(directory) if f.endswith(".json")],
        reverse=True,
    )[:n]
    result = []
    for f in files:
        try:
            result.append(load(os.path.join(directory, f)))
        except Exception:
            continue
    return result
