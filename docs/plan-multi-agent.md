# 多 Agent 协作 - 学习计划

## 目标
在 react_loop.py 上实现 Orchestrator-Worker 模式

## 今天 - 单体链式调用
用户输入多子句 -> LLM 拆任务 -> 每个子任务跑独立 ReAct Loop -> 合并

## 明天 - 独立 Orchestrator
独立的规划器，拆任务 -> 分发 -> 汇总

## 后天 - Worker 隔离
每个 Worker 独立工具集、独立记忆、独立对话历史
