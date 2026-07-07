"""Harness — LangGraph Agent 的观测 + 安全 + 回放层

LangGraph 的节点执行是黑盒：call_model 返回了什么 thought？
tools_node 调用了什么工具、花了多久？跑完后没法复盘。

Harness 解决这三个问题：
  1. Recording（记录） — 每一步的 thought/action/observation/duration/tokens
  2. Sandbox（安全）   — 工具调用在子进程执行，崩溃不拖死主图
  3. Replay（回放）    — 读 JSON 轨迹文件，步进或整段观看

集成方式：
  from harness import Harness
  harness = Harness()
  harness.start_trajectory(query="...", model="...")

  在 LangGraph 节点中，用 context 传 harness 实例：
  call_model(state, config)  → 捕获 LLM response 后记录
  tools_node(state, config)  → 捕获每个 tool_call 后记录
"""

from .recorder import TrajectoryRecorder
from .sandbox import Sandbox
from .replay import Replay


class Harness:
    """统一 Harness 层

    同时管理轨迹记录、沙箱隔离和回放调试。

    用法：
        harness = Harness(sandbox_enabled=False)
        harness.start_trajectory(query="今天日期", model="deepseek-chat")

        # 运行 Agent...
        # 在 call_model 节点后：
        harness.record_thought(step=1, thought="用户想知道日期", tokens=120)

        # 在 tools_node 节点后：
        harness.record_action(
            step=1,
            action_name="get_current_time",
            action_args="{}",
            observation="2026-07-07 14:30:00",
            duration_seconds=0.8,
            tokens=50,
        )

        # 完成后：
        harness.finish(final_answer="今天是 2026-07-07")
        path = harness.save()
        print(f"轨迹已保存: {path}")
    """

    def __init__(self, sandbox_enabled: bool = False, sandbox_timeout: int = 30):
        self.recorder: TrajectoryRecorder | None = None
        self.sandbox = Sandbox(enabled=sandbox_enabled, timeout=sandbox_timeout)
        self.replayer = Replay()

    # ── 轨迹记录 ──

    def start_trajectory(self, query: str, model: str = "",
                         system_prompt: str = "") -> "Harness":
        """开始一次新的轨迹记录"""
        self.recorder = TrajectoryRecorder(
            query=query, model=model, system_prompt=system_prompt,
        )
        return self

    def record_thought(self, step: int, thought: str = "", tokens: int = 0) -> "Harness":
        """记录 LLM 的思考/回复（thought → action 的第一步）"""
        if self.recorder:
            self.recorder.record_thought(step, thought, tokens)
        return self

    def record_action(self, step: int, action_name: str = "",
                      action_args: str = "", observation: str = "",
                      duration_seconds: float = 0, tokens: int = 0) -> "Harness":
        """记录一次工具调用（action → observation）"""
        if self.recorder:
            self.recorder.record_action(
                step, action_name, action_args,
                observation, duration_seconds, tokens,
            )
        return self

    def finish(self, final_answer: str = "") -> "Harness":
        """标记轨迹结束"""
        if self.recorder:
            self.recorder.set_final_answer(final_answer)
        return self

    def save(self) -> str:
        """保存轨迹到 JSON 文件，返回路径"""
        if self.recorder:
            return self.recorder.save()
        return ""

    def add_unsafe_tool(self, tool_name: str) -> "Harness":
        """注册不应在沙箱中运行的快速工具（如 get_current_time）"""
        self.sandbox.add_unsafe_tool(tool_name)
        return self

    def is_sandboxed(self, tool_name: str) -> bool:
        """判断某个工具是否应当在沙箱中执行"""
        return self.sandbox.should_sandbox(tool_name)

    def run_sandboxed(self, tool_call: dict) -> str:
        """在沙箱子进程中执行工具调用"""
        return self.sandbox.run(tool_call)

    # ── 重放 ──

    def list_recordings(self, directory: str = "") -> list:
        """列出所有轨迹文件"""
        return self.replayer.list_recordings(directory)

    def playback(self, filepath: str, step_by_step: bool = False):
        """回放一条轨迹"""
        return self.replayer.play(filepath, step_by_step=step_by_step)
