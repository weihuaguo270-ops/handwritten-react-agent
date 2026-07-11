"""eval_loop — Eval Loop 自适应循环引擎

核心工作流（生成类任务）：

    用户输入
        │
        ▼
    IntentClassifier → "generative_task"
        │
        ▼
    ╔══════════════════════════════════════════╗
    ║         Eval Loop（循环直到通过）          ║
    ║                                          ║
    ║  Agent 执行 → 轨迹解析 → Process Reward   ║
    ║       │                    │             ║
    ║       │               ┌───┴───┐          ║
    ║       │             全部达标  有低分项     ║
    ║       │               │       │          ║
    ║       │               ▼       ▼          ║
    ║       │            输出结果  打包修正指令  ║
    ║       │               │       │          ║
    ║       │               │       ▼          ║
    ║       │               │    LLM 重新生成   ║
    ║       │               │     → 回到起点    ║
    ║       │               │                  ║
    ║       └───────────────┘                  ║
    ╚══════════════════════════════════════════╝

关键设计：
  - max_iterations: 防止无限循环（默认 3 次）
  - min_improvement: 防止震荡（两次迭代总提升 < 此值 → 停止）
  - 每次循环记录 healing_log，最后完整报告含自愈过程
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from handwritten_react_agent.intent.classifier import IntentClassifier, TaskType
from core.trajectory_parser import (
    parse_trajectory,
    StepsDAG,
    dag_to_text,
)
from core.process_reward import (
    ProcessRewardScorer,
    ProcessRewardReport,
    analyze_error_propagation,
    pack_revision_instructions,
)
from handwritten_react_agent.safety.human_in_the_loop import HumanInTheLoop


@dataclass
class EvalLoopResult:
    """Eval Loop 的最终输出

    属性:
        query:         用户原始输入
        task_type:     识别出的任务类型
        final_output:  最终输出（Agent 的最终回复）
        report:        Process Reward 评分报告
        iterations:    循环次数（1 = 一次通过）
        healing_log:   自愈过程记录
        error_analysis: 错误传播分析
        passed:        最终是否通过
    """
    query: str
    task_type: str
    final_output: str
    report: ProcessRewardReport
    iterations: int
    healing_log: list[dict]
    error_analysis: dict[str, Any]
    passed: bool


@dataclass
class EvalLoopConfig:
    """Eval Loop 配置"""
    max_iterations: int = 3          # 最大循环次数
    min_improvement: float = 0.1     # 最小改进幅度（低于此停止震荡）
    fast_mode_threshold: int = 5     # 步数少于这个值用快速模式
    verbose: bool = True             # 是否打印过程


class EvalLoopEngine:
    """Eval Loop 引擎

    用法：
        engine = EvalLoopEngine(
            agent_fn=run_agent,       # Agent 执行函数
            judge_fn=call_judge_llm,  # Judge LLM 调用函数
        )
        result = engine.execute("帮我写一份 AI 行业分析报告")

        if result.passed:
            print(result.final_output)
        else:
            print("质量未达标，报告如下：")
            print(result.report)
    """

    def __init__(
        self,
        agent_fn: Callable[[str], dict[str, Any]],
        judge_fn: Callable[[str], dict[str, Any]],
        config: Optional[EvalLoopConfig] = None,
        intent_classifier: Optional[IntentClassifier] = None,
        hitl: Optional[HumanInTheLoop] = None,
    ):
        """初始化引擎

        参数:
            agent_fn:          Agent 执行函数
            judge_fn:          Judge LLM 调用函数
            config:            循环配置
            intent_classifier: 意图分类器（默认新建）
            hitl:              人工审批管理器。传 None 时不检查权限
        """
        self.agent_fn = agent_fn
        self.judge_fn = judge_fn
        self.config = config or EvalLoopConfig()
        self.classifier = intent_classifier or IntentClassifier()
        self.hitl = hitl

    def execute(self, user_input: str) -> EvalLoopResult:
        """执行完整的 Eval Loop

        参数:
            user_input: 用户输入

        返回:
            EvalLoopResult
        """
        # 1. 意图分类
        task_type = self.classifier.classify(user_input)
        self._log(f"意图分类: {task_type}")

        # 2. 功能测试类 → 单次执行 + 返回
        if task_type == TaskType.FUNCTIONAL_TEST:
            return self._execute_functional(user_input)

        # 3. 生成类 → Eval Loop
        return self._execute_generative(user_input)

    def _execute_functional(self, query: str) -> EvalLoopResult:
        """功能测试类：单次执行 + 直接返回"""
        agent_output = self.agent_fn(query)
        trajectory = agent_output.get("trajectory", {})

        # 解析并快速评分
        try:
            dag = parse_trajectory(trajectory)
            scorer = ProcessRewardScorer(judge_fn=self.judge_fn)
            report = scorer.score_trajectory(dag, fast_mode=True)
        except Exception:
            report = ProcessRewardReport(
                query=query,
                per_step=[],
                overall_score=0,
                num_steps=0,
                num_scored=0,
                num_failed_steps=0,
                error_sources=[],
                needs_revision=False,
                healing_log=[],
                dag_summary={},
            )

        return EvalLoopResult(
            query=query,
            task_type=TaskType.FUNCTIONAL_TEST,
            final_output=agent_output.get("output", ""),
            report=report,
            iterations=1,
            healing_log=[],
            error_analysis={},
            passed=True,
        )

    def _execute_generative(self, query: str) -> EvalLoopResult:
        """生成类任务：Eval Loop 自适应循环"""
        scorer = ProcessRewardScorer(judge_fn=self.judge_fn)
        healing_log: list[dict] = []
        prev_overall_score = 0.0
        oscillation_count = 0

        for iteration in range(1, self.config.max_iterations + 1):
            self._log(f"迭代 {iteration}/{self.config.max_iterations}")

            # a. Agent 执行
            agent_output = self.agent_fn(query)
            trajectory = agent_output.get("trajectory", {})
            output_text = agent_output.get("output", "")

            # b. 轨迹解析 → DAG
            try:
                dag = parse_trajectory(trajectory)
            except ValueError as e:
                self._log(f"轨迹解析失败: {e}")
                continue

            # c. Process Reward 评分
            fast = dag.num_steps < self.config.fast_mode_threshold
            report = scorer.score_trajectory(dag, fast_mode=fast)

            # d. 错误传播分析
            error_analysis = analyze_error_propagation(report, dag)

            # e. 记录迭代日志
            log_entry = {
                "iteration": iteration,
                "overall_score": report.overall_score,
                "needs_revision": report.needs_revision,
                "num_failed_steps": report.num_failed_steps,
                "error_sources": report.error_sources,
                "final_output_preview": output_text[:200],
            }
            healing_log.append(log_entry)

            self._log(
                f"  总分: {report.overall_score:.3f}, "
                f"失败步骤: {report.num_failed_steps}, "
                f"需修正: {report.needs_revision}"
            )

            # f. 检查是否达标
            if not report.needs_revision:
                self._log(f"✅ 迭代 {iteration} 通过")
                return EvalLoopResult(
                    query=query,
                    task_type=TaskType.GENERATIVE_TASK,
                    final_output=output_text,
                    report=report,
                    iterations=iteration,
                    healing_log=healing_log,
                    error_analysis=error_analysis,
                    passed=True,
                )

            # g. 检测震荡：分数没怎么提升
            improvement = report.overall_score - prev_overall_score
            if iteration > 1 and improvement < self.config.min_improvement:
                oscillation_count += 1
                if oscillation_count >= 2:
                    self._log(f"⚠ 检测到震荡（改进幅度 {improvement:.3f} < {self.config.min_improvement}），停止循环")
                    return EvalLoopResult(
                        query=query,
                        task_type=TaskType.GENERATIVE_TASK,
                        final_output=output_text,
                        report=report,
                        iterations=iteration,
                        healing_log=healing_log,
                        error_analysis=error_analysis,
                        passed=False,
                    )

            prev_overall_score = report.overall_score

            # h. 用户确认：修正指令注入前先问用户
            if self.hitl:
                if not self.hitl.check_direction(
                    "修正指令注入",
                    details=f"迭代 {iteration} 中有 {report.num_failed_steps} 步不合格，"
                            f"将注入修正指令让 Agent 重试",
                ):
                    self._log(f"⏸ 用户拒绝修正指令注入，停止循环")
                    return EvalLoopResult(
                        query=query,
                        task_type=TaskType.GENERATIVE_TASK,
                        final_output=output_text,
                        report=report,
                        iterations=iteration,
                        healing_log=healing_log,
                        error_analysis=error_analysis,
                        passed=False,
                    )

            # i. 打包修正指令
            fix_instructions = pack_revision_instructions(report, dag)
            self._log(f"  生成修正指令，共 {len(fix_instructions)} 字符")

            # j. 用户确认：重新执行步骤前问用户
            if self.hitl:
                if not self.hitl.check_direction(
                    "重新执行步骤",
                    details=f"将重新执行 Agent，第 {iteration + 1} 次尝试",
                ):
                    self._log(f"⏸ 用户拒绝重新执行，停止循环")
                    return EvalLoopResult(
                        query=query,
                        task_type=TaskType.GENERATIVE_TASK,
                        final_output=output_text,
                        report=report,
                        iterations=iteration,
                        healing_log=healing_log,
                        error_analysis=error_analysis,
                        passed=False,
                    )

            # k. 将修正指令注入 Agent 的 query（让 Agent 重试时知道哪里错了）
            revised_query = (
                f"【原任务】{query}\n\n"
                f"【前次反馈】以下步骤质量不达标，请修正：\n"
                f"{fix_instructions}\n\n"
                f"请基于以上反馈重新生成。保留之前做得好的部分，只修复标记的问题。"
            )
            query = revised_query

        # 达到最大迭代次数仍未通过
        self._log(f"⚠ 达到最大迭代次数 {self.config.max_iterations}，最后一次结果如下")
        return EvalLoopResult(
            query=query,
            task_type=TaskType.GENERATIVE_TASK,
            final_output=output_text,
            report=report,
            iterations=self.config.max_iterations,
            healing_log=healing_log,
            error_analysis=analyze_error_propagation(report, dag),
            passed=False,
        )

    def _log(self, msg: str) -> None:
        if self.config.verbose:
            print(f"[EvalLoop] {msg}")
