# ReAct Agent Framework

[![CI](https://github.com/weihuaguo270-ops/react-agent/actions/workflows/test.yml/badge.svg)](https://github.com/weihuaguo270-ops/react-agent/actions/workflows/test.yml) [![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

** ReAct Agent 框架，双实现路线** — 手写运行时用于深入理解和完全控制，LangGraph 版用于生产部署。14 个模块覆盖 RAG、MCP 工具集成、多 Agent 编排、执行录制回放和安全护栏。

## 架构概览

框架提供两种互补的 ReAct Agent 实现：

| 维度 | 手写版（`src/`） | LangGraph 版（`experiments/langgraph/`） |
|------|-----------------|----------------------------------------|
| **依赖** | Python 标准库 + LLM API | LangChain + LangGraph |
| **目标** | 完全透明，每行代码可控 | 可扩展部署 |
| **状态管理** | 手动管理 | LangGraph 内置 graph 状态 |

### 执行流程

```
query 输入
  │
  ├── 普通入口 → react_loop()
  │     │
  │     ├── Step 0: 构建 system prompt（base + 角色注入 + CoT 策略）
  │     ├── Step 1: LLM 调用 → thought/action
  │     ├── Step 2: 工具执行（权限检查）
  │     ├── Step 3: 观察结果集成
  │     └── 循环直至输出最终答案
  │
  └── Orchestrator 入口
        ├── plan() → 任务分解（带依赖追踪）
        ├── run_worker() → 每个子任务独立运行 react_loop()
        └── synthesize() → 汇总结果
```

### 模块清单

```
react_agent/
│
├── react_loop.py        核心 ReAct 循环（thought → action → observation）
├── llm.py               LLM Provider 抽象（多 provider 切换）
├── tools/               工具注册 + 内置工具集
│   ├── web_search.py    网络搜索
│   ├── fetch_page.py    页面内容提取
│   ├── execute_python.py Python 沙箱执行
│   ├── calculator.py    计算器
│   └── ...
├── context.py           上下文管理
├── memory.py            对话记忆
├── cot.py               Chain-of-Thought 策略注入
├── tot.py               Tree-of-Thought 工具集成
├── prompts.py           System Prompt 构建
├── rag.py               检索增强生成
│
├── orchestrator.py      多 Agent 任务分解 + 汇总
├── planner.py           任务规划 + 依赖解析
├── mcp_client.py        MCP 协议客户端
│
├── eval/                评估与评分
│   ├── runner.py        批量评估
│   ├── scorer.py        评分函数
│   ├── dataset.py       数据集加载
│   └── report.py        报告生成
│
├── harness/             执行录制与回放
│   ├── recorder.py      完整轨迹录制
│   ├── replay.py        逐步骤回放
│   └── sandbox.py       隔离执行沙箱
│
├── safety/              安全与权限
│   ├── permissions.py   四级权限系统（SAFE/NOTIFY/CONFIRM/DENY）
│   ├── human_in_the_loop.py 人工审批
│   └── trace_watch.py   执行轨迹监控
│
├── intent/              任务分类
│   └── classifier.py    意图识别（功能测试 vs 生成式任务）
│
├── dashboard/           实时执行可视化
│
└── resilience.py        错误处理与重试
```

## 核心功能

### 多 Provider LLM 支持

环境变量切换，无需改代码：

```bash
export LLM_PROVIDER=deepseek
export LLM_PROVIDER=openai
export LLM_PROVIDER=anthropic
```

`llm_config.json` 中配置各 provider 的 API key、base URL 和模型名。

### 权限安全系统

四级权限控制工具调用：

| 等级 | 行为 | 适用场景 |
|------|------|---------|
| SAFE | 自动放行 | web_search、calculator |
| NOTIFY | 记录 + 继续 | fetch_page（外部域名） |
| CONFIRM | 询问用户 | write_file、execute_python |
| DENY | 拦截 | rm -rf、敏感路径 |

### 执行轨迹录制

完整录制每步 thought/action/observation，支持事后分析和回放：

```python
from react_agent.harness.recorder import current_trajectory

result = react_loop("分析这份数据")
trajectory = current_trajectory()

# 逐步骤回放
from react_agent.harness.replay import replay_trajectory
replay_trajectory(trajectory)
```

### RAG 与 MCP 集成

- **RAG**：文档摄入、分块、嵌入、检索，可配置向量存储
- **MCP Client**：连接外部 MCP 服务器，自动发现并调用工具

### 多 Agent 编排

复杂任务自动分解为带依赖的子任务：

```python
from react_agent.orchestrator import Orchestrator

orc = Orchestrator()
plan = orc.plan("调研并撰写 AI 趋势报告")
# → [task_1: 搜索趋势, task_2: 分析数据, task_3: 撰写报告]
#   task_2 依赖 task_1，task_3 依赖 task_1 + task_2

results = orc.execute(plan)
```

## LangGraph 版（`experiments/langgraph/`）

基于 LangChain/LangGraph 的图计算实现，适用于生产环境中需要框架集成的场景。功能等价于手写版：可配置 Agent 图、上下文管理、MCP 工具、RAG pipeline、执行录制、记忆管理、多 Agent 编排。

## 快速开始

```bash
pip install -e .
cp .env.example .env
# 编辑 .env 填入 API key

# 运行
python -m react_agent "法国的首都是什么？"

# 启动 Web 面板
python -m react_agent.dashboard.server
```

## 环境要求

- Python 3.10+
- LLM API key
- LangChain + LangGraph（experiments/langgraph/ 需要，可选）

## 相关项目

- [llm-eval-engine](https://github.com/weihuaguo270-ops/llm-eval-engine) — LLM 评估框架
- [transformer-attention](https://github.com/weihuaguo270-ops/transformer-attention) — NumPy/PyTorch Transformer Attention 实现
- [trace-debugger](https://github.com/weihuaguo270-ops/trace-debugger) — Agent 执行轨迹分析工具

## License

MIT
