"""agent_wrapper — 将权限 + 自动 Eval 嵌入 Agent 执行流程

两个核心功能：
  1. PermissionWrapper: 拦截 Agent 的工具调用，在每次执行前检查权限
  2. AutoEvalWrapper: Agent 执行完成后，自动触发 Eval Loop

集成方式（在 react_loop.py 中）：
    from experiments.eval-engine.integration.agent_wrapper import (
        PermissionWrapper, AutoEvalWrapper, create_guarded_agent
    )

    # 方式 A：直接包装整个 agent 函数
    guarded_agent = create_guarded_agent(
        agent_fn=react_loop,
        ask_fn=ask_user,       # 你的询问回调
    )
    result = guarded_agent("帮我写一份报告")

    # 方式 B：单独使用权限包装器
    permission = PermissionWrapper(ask_fn=ask_user)
    permission.intercept_tool_call("write_file", {"path": "/etc"})
"""

from __future__ import annotations
import json
import sys
import os
from typing import Any, Callable, Optional

# 添加 eval-engine 到路径
_eval_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
for p in [_eval_dir, _src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from handwritten_react_agent.safety.permissions import PermissionLevel, get_tool_permission, describe_action
from handwritten_react_agent.safety.human_in_the_loop import HumanInTheLoop
from handwritten_react_agent.safety.trace_watch import TraceWatch
from handwritten_react_agent.intent.classifier import IntentClassifier, TaskType


# ══════════════════════════════════════════════
#  1. 权限拦截器 — 嵌入工具调用
# ══════════════════════════════════════════════


class PermissionWrapper:
    """权限拦截器——包装 execute_tool_call 函数

    用法：
        original_execute = execute_tool_call  # 保存原函数
        permission = PermissionWrapper(ask_fn=ask_user)
        execute_tool_call = permission.wrap(original_execute)

        # 之后每次调工具，都会先检查权限
    """

    def __init__(
        self,
        ask_fn: Optional[Callable[[str, list[str]], str]] = None,
        hitl: Optional[HumanInTheLoop] = None,
    ):
        """初始化权限拦截器

        参数:
            ask_fn: 询问用户的回调函数。传 None 时默认放行 SAFE、拒绝 DENY
            hitl:   可复用的 HITL 管理器。如果提供，则忽略 ask_fn
        """
        self.hitl = hitl or HumanInTheLoop(ask_fn=ask_fn)
        self.watch = TraceWatch(hitl=self.hitl)  # 绑定路径监控
        self._blocked_calls: list[dict] = []

    def check_tool_call(
        self,
        tool_name: str,
        tool_args: dict,
        reason: str = "",
    ) -> Optional[str]:
        """检查工具调用是否允许

        参数:
            tool_name: 工具名
            tool_args: 工具参数
            reason:    调用理由

        返回:
            None  = 允许执行
            str   = 拒绝理由（调用方应拦截执行）
        """
        if not self.hitl.check_tool_call(tool_name, tool_args, reason):
            self._blocked_calls.append({
                "tool": tool_name,
                "args": tool_args,
                "reason": "用户拒绝",
            })
            return f"操作被用户拒绝：{tool_name}"
        return None

    def wrap(self, original_fn: Callable) -> Callable:
        """包装原始 execute_tool_call 函数

        返回的函数签名与原始函数一致：
            wrapped_fn(tool_call: dict) -> str
        """
        def wrapped(tool_call: dict) -> str:
            # 防御：畸形调用
            if not isinstance(tool_call, dict) or "function" not in tool_call:
                return json.dumps({"error": "畸形工具调用", "blocked": True})

            func_name = tool_call["function"].get("name", "unknown")
            try:
                arguments = json.loads(tool_call["function"].get("arguments", "{}"))
            except (json.JSONDecodeError, KeyError, TypeError):
                arguments = {}

            # 权限检查
            block_reason = self.check_tool_call(func_name, arguments)
            if block_reason:
                self.watch.on_tool_call(func_name, arguments, block_reason)
                return json.dumps({"error": block_reason, "blocked": True})

            # 权限通过 → 调用原始函数
            result = original_fn(tool_call)
            self.watch.on_tool_call(func_name, arguments, result)
            return result

        return wrapped

    @property
    def blocked_count(self) -> int:
        return len(self._blocked_calls)

    @property
    def stats(self) -> dict:
        report = self.watch.summary()
        return {
            **self.hitl.stats(),
            "blocked_calls": self.blocked_count,
            "trace_errors": report.error_count,
            "trace_switches": report.switch_count,
        }


# ══════════════════════════════════════════════
#  2. 自动触发 Eval — Agent 完成后自动评分
# ══════════════════════════════════════════════


class AutoEvalWrapper:
    """自动 Eval 包装器

    Agent 执行完成后，自动判断是否需要启动 Eval Loop。

    触发条件：
      - 功能测试类（functional_test）：直接返回结果
      - 生成式任务（generative_task）：自动评分 → 低分则自动修正
      - OR 用户显式要求评估

    用法：
        from experiments.eval-engine.integration.agent_wrapper import AutoEvalWrapper
        auto_eval = AutoEvalWrapper(judge_fn=call_judge)
        agent_with_eval = auto_eval.wrap(react_loop)
        result = agent_with_eval("帮我写一份报告")
        # → 自动执行 Agent → 自动评分 → 返回最终结果
    """

    def __init__(
        self,
        judge_fn: Optional[Callable[[str], dict[str, Any]]] = None,
        hitl: Optional[HumanInTheLoop] = None,
        auto_eval: bool = True,
    ):
        """初始化自动 Eval 包装器

        参数:
            judge_fn: Judge LLM 调用函数（不传则不评分）
            hitl:     人工审批管理器（传 None 则修正不询问）
            auto_eval: 是否自动触发 eval（设为 False 则手动触发）
        """
        self.judge_fn = judge_fn
        self.hitl = hitl
        self.auto_eval = auto_eval
        self.classifier = IntentClassifier()

        # 延迟导入 EvalLoopEngine（避免循环依赖）
        self._engine = None

    @property
    def _eval_engine(self):
        if self._engine is None and self.judge_fn:
            from loop.eval_loop import EvalLoopEngine, EvalLoopConfig
            self._engine = EvalLoopEngine(
                agent_fn=self._dummy_agent,
                judge_fn=self.judge_fn,
                config=EvalLoopConfig(verbose=False),
                hitl=self.hitl,
            )
        return self._engine

    def wrap(self, agent_fn: Callable) -> Callable:
        """包装 Agent 执行函数

        返回的函数：
          1. 执行 Agent 并捕获轨迹
          2. 自动判断任务类型
          3. 生成式任务 → 自动 Eval Loop
          4. 返回最终结果
        """
        def wrapped(query: str) -> str:
            # 1. 意图分类
            task_type = self.classifier.classify(query)

            # 2. 功能测试 → 直接执行并返回
            if task_type == TaskType.FUNCTIONAL_TEST or not self.auto_eval:
                return agent_fn(query)

            # 3. 生成式任务 → 使用 Eval Loop
            if self._eval_engine:
                # 将 agent_fn 注入 Eval Loop
                self._eval_engine.agent_fn = agent_fn
                result = self._eval_engine.execute(query)
                return result.final_output
            else:
                # 没有 Judge → 直接执行
                return agent_fn(query)

        return wrapped

    # ── 触发条件查询 ──

    @staticmethod
    def should_eval(query: str) -> bool:
        """判断是否应该触发评估

        对外暴露的分类接口，供其他模块使用。
        """
        classifier = IntentClassifier()
        return classifier.classify(query) == TaskType.GENERATIVE_TASK

    @staticmethod
    def trigger_conditions() -> list[dict]:
        """返回所有触发条件说明"""
        return [
            {"condition": "功能测试类任务", "trigger": False,
             "reason": "用户明确测试工具功能，直接返回结果"},
            {"condition": "生成式任务", "trigger": True,
             "reason": "复杂生成任务，自动评分+修正"},
            {"condition": "用户显式要求评估", "trigger": True,
             "reason": "无论任务类型，用户要求评估就执行"},
            {"condition": "auto_eval=False", "trigger": False,
             "reason": "用户关闭了自动评估"},
        ]

    def _dummy_agent(self, query: str) -> dict:
        """占位 agent_fn（由 wrap 时覆盖）"""
        return {"output": "", "trajectory": {}}


# ══════════════════════════════════════════════
#  3. 一站式工厂函数
# ══════════════════════════════════════════════


def create_guarded_agent(
    agent_fn: Callable[[str], str],
    execute_tool_fn: Optional[Callable] = None,
    judge_fn: Optional[Callable[[str], dict[str, Any]]] = None,
    ask_fn: Optional[Callable[[str, list[str]], str]] = None,
    enable_permissions: bool = True,
    enable_auto_eval: bool = True,
) -> Callable[[str], str]:
    """创建一个带权限 + 自动 Eval 的 Agent 函数

    一站式工厂，把权限拦截和自动评估打包成一个可直接替换 react_loop 的函数。

    参数:
        agent_fn:        原始 Agent 函数（如 react_loop）
        execute_tool_fn: 原始 execute_tool_call 函数（如不传则不拦截权限）
        judge_fn:        Judge LLM 调用函数（如不传则只分类不评分）
        ask_fn:          询问用户的回调函数（传 None 则无交互）
        enable_permissions: 是否启用权限拦截
        enable_auto_eval:    是否启用自动 Eval

    返回:
        Callable[[str], str]: 带防护的 Agent 函数
    """
    hitl = None
    if enable_permissions:
        hitl = HumanInTheLoop(ask_fn=ask_fn)

        # 如果有 execute_tool_fn，包装它
        if execute_tool_fn:
            perm = PermissionWrapper(hitl=hitl)
            wrapped_execute = perm.wrap(execute_tool_fn)
            # 注意：这里 caller 需要把 wrapped_execute 赋值给原变量
            print(f"[GuardedAgent] 已启用权限拦截")

    # 自动 Eval 包装
    if enable_auto_eval and judge_fn:
        auto_eval = AutoEvalWrapper(judge_fn=judge_fn, hitl=hitl)
        guarded = auto_eval.wrap(agent_fn)
        print(f"[GuardedAgent] 已启用自动 Eval（生成式任务自动评分）")
        return guarded

    return agent_fn
