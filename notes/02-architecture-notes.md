# 架构笔记

## 运行时全链路（文本版）

```
query 输入
├── 普通入口 → 直接 react_loop()
└── Orchestrator 入口
      ├── plan() → Planner 分解任务
      ├── run_worker() → 每个子任务独立走 react_loop()
      └── synthesize() → 汇总

react_loop 内:
  step 0: system prompt 三层拼接（base → ROLE_MANAGER → COT）
  step 0: Memory 检索
  step 0: Harness 开始记录
  ReAct Loop 每步:
    call_llm → thought + tool_calls
    ├── 无 tool_calls → 检查 FINAL ANSWER / 隐式答案 / 寒暄 → 结束
    └── 有 tool_calls
          ├── TOOL_REGISTRY（本地/MCP）→ 执行 → Harness 记录
          ├── ToT 是一个普通工具（tot_reasoning），不是系统层
          └── CONTEXT.manage() 检查 token 用量
```

## 模块间关系

| 层级 | 模块 | 控制范围 |
|------|------|---------|
| 任务层 | Planner → Orchestrator | 把一句话拆成多个子任务 |
| 对话层 | ReAct Loop | 思考→行动→观察循环 |
| 推理层 | CoT / ToT | 增强 LLM 思考质量 |

## Agent = LLM + Harness

Harness 三层统一设计：

| 子系统 | 文件 | 职责 |
|--------|------|------|
| Recorder | harness/recorder.py | 轨迹记录（原 harness.py） |
| Sandbox | harness/sandbox.py | 子进程隔离执行 |
| Replay | harness/replay.py | 离线逐步回放 |

统一入口：

```python
from harness import Harness
h = Harness()
h.start_session(query, model, prompt)
h.add_step(step, thought=..., action_name=..., result=...)
h.save()
```

## 各模块触发时机

| 模块 | 触发条件 | 介入位置 |
|------|---------|---------|
| Memory | react_loop 每次启动时 | 拼接进 system prompt |
| CoT | react_loop 启动时、system prompt 构建阶段 | 向 system prompt 末尾追加推理指令 |
| ToT | 仅当 LLM 选择 tot_reasoning 工具时 | ReAct Loop 内工具执行阶段 |
| Planner / Orchestrator | only when user query → Orchestrator.execute() | 外部包装 |
| Context | ReAct Loop 每步结束后 | messages.append 之后 |
| Harness | react_loop 进入/退出/每步工具调用 | 持久化到 trajectories/*.json |
