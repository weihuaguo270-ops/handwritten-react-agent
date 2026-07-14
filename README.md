# ReAct Agent Framework

[![CI](https://github.com/weihuaguo270-ops/react-agent/actions/workflows/test.yml/badge.svg)](https://github.com/weihuaguo270-ops/react-agent/actions/workflows/test.yml) [![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**ReAct Agent 学习实现，双实现路线** — 手写运行时用于深入理解和完全控制，LangGraph 版用于对照框架集成写法。覆盖 RAG、MCP、多 Agent 编排、执行录制回放、权限提示与内置评测等模块。

## 范围与定位（可招聘叙事）

| 是 | 不是 |
|----|------|
| 教学用 ReAct / Harness / Eval 串联实验台 | 生产级 Agent 运行时或安全产品 |
| 模块广度优先（便于学习完整链路） | 单一职责、可交接的微服务切分 |
| 离线 CI + 可选真实 LLM 冒烟 | SLA / 不可信代码隔离保证 |

跨仓职责：**本仓** = Agent 执行 + capability 规则打分；**llm-eval-engine** = Process Reward / 人机校准；**trace-debugger** = 轨迹启发式复盘。共享轨迹约定见 [`schemas/harness_trajectory.schema.json`](schemas/harness_trajectory.schema.json)。

## 架构概览

本仓库提供两种互补的 ReAct Agent 实现（教学/实验用途）：

| 维度 | 手写版（`src/`） | LangGraph 版（`experiments/langgraph/`） |
|------|-----------------|----------------------------------------|
| **依赖** | Python 标准库 + LLM API | LangChain + LangGraph |
| **目标** | 完全透明，每行代码可控 | 对照框架图编排写法 |
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
│   ├── execute_python.py 子进程执行 Python（超时；非安全沙箱）
│   ├── calculator.py    计算器
│   └── ...
├── context.py           上下文管理
├── memory.py            对话记忆
├── cot.py               Chain-of-Thought 策略注入
├── tot.py               Tree-of-Thought 工具集成
├── prompts.py           System Prompt 构建
├── rag.py               检索增强生成（可选依赖，见下方安装）
│
├── orchestrator.py      多 Agent 任务分解 + 汇总
├── planner.py           任务规划 + 依赖解析
├── mcp_client.py        MCP 协议客户端
├── mcp_config.py        MCP 启动参数（mcp_servers.json / 环境变量）
│
├── eval/                评估与评分（任务 capability 规则打分；Process Reward 见 llm-eval-engine）
│   ├── runner.py        批量评估（支持 consistency 重复跑）
│   ├── scorer.py        功能 4 维评分
│   ├── capability_scorer.py  能力评估（准确率/工具/推理/一致性/幻觉）
│   ├── dataset.py       数据集加载
│   ├── dataset.json     功能验证集
│   ├── capability_dataset.json  能力评估集
│   └── report.py        报告生成（含 by_capability）
│
├── harness/             执行录制与回放
│   ├── recorder.py      完整轨迹录制
│   ├── replay.py        逐步骤回放
│   └── sandbox.py       工具子进程 + 超时（崩溃隔离，非安全边界）
│
├── safety/              权限提示（学习用）
│   ├── permissions.py   工具名表四级提示（SAFE/NOTIFY/CONFIRM/DENY）
│   ├── human_in_the_loop.py 人工审批
│   └── trace_watch.py   执行轨迹监控
│
├── intent/              任务分类
│   └── classifier.py    意图识别（功能测试 vs 生成式任务）
│
├── dashboard/           实时执行可视化（server.py + index.html）
│
└── resilience.py        错误处理与重试
```

## 核心功能

### 多 Provider LLM 支持

优先读取项目根目录 `.env` / `llm_config.json`（`.env` 会覆盖系统里残留的旧 API Key），也可用环境变量切换 provider：

```bash
export LLM_PROVIDER=deepseek   # 或 openai / anthropic
```

### 权限与沙箱（学习级，非生产隔离）

工具调用采用 **名称表 + 可选参数规则** 的四级提示策略（`safety/permissions.py`）：

| 等级 | 行为 | 适用场景（示例） |
|------|------|------------------|
| SAFE | 自动放行 | web_search、calculator |
| NOTIFY | 记录后继续 | 部分读信息工具 |
| CONFIRM | 询问用户 | write_file、execute_python |
| DENY | 默认拦截 | 表内登记的破坏性工具名 |

说明（诚实边界）：
- **不是** OS / 容器级沙箱；未知工具名默认不在 DENY 表内。
- `execute_python` / `harness/sandbox` 主要是 **子进程 + 超时**，用于隔离崩溃与卡住，**不能**当作安全边界（代码仍可访问本机网络/文件权限范围内资源）。
- 危险 shell 字符串（如 `rm -rf`）**不会**被逐字解析拦截；请勿用生产不可信代码跑本项目的执行工具。

`harness/sandbox.py` 支持 `off` / `auto` / `on`；子进程内禁止再次预热沙箱，避免递归拉起进程。

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

- **RAG**：`pip install -e ".[rag]"` 后可用；未安装时 Agent 仍可跑，`rag_query` 会提示缺少依赖
- **MCP**：默认仅尝试连接便携命令 `uvx mcp-server-time`。本机路径 / filesystem server 请复制 [`mcp_servers.example.json`](mcp_servers.example.json) → `mcp_servers.json`（已 gitignore），或设置 `REACT_AGENT_MCP_CONFIG` / CLI `--mcp ...`

### 多 Agent 编排

复杂任务自动分解为带依赖的子任务：

```python
from react_agent.orchestrator import Orchestrator

orc = Orchestrator()
# execute 内部会先 plan（分解子任务与依赖），再按层级执行并 synthesize
results = orc.execute("调研并撰写 AI 趋势报告")
# 也可单独查看计划：orc.plan("调研并撰写 AI 趋势报告")
```

## LangGraph 版（`experiments/langgraph/`）

基于 LangChain/LangGraph 的图计算实验实现，用于对照「手写循环 vs 框架编排」。覆盖可配置 Agent 图、上下文管理、MCP 工具、RAG pipeline、执行录制、记忆管理、多 Agent 编排；与手写版能力接近，但未做严格等价性测试。

## 快速开始

```bash
pip install -e ".[test]"          # 核心 + pytest
pip install -e ".[rag,test]"      # 需要本地语义检索时再装
cp .env.example .env
# 编辑 .env 填入 API key
# 可选：cp mcp_servers.example.json mcp_servers.json

# 运行
python -m react_agent "法国的首都是什么？"

# 启动 Web 面板
python -m react_agent.dashboard.server
```

## 评测

功能验证（工具/关键词/步数）与能力评估（规则打分）共用 `EvalRunner`：

| 指标 | capability | 含义 |
|------|------------|------|
| 准确率 | `accuracy` | 最终答案命中 `expected_answer` |
| 工具选择 | `tool_selection` | 工具精确率 / 召回率 / F1 |
| 多步推理 | `reasoning` | 检查点 + 最终答案 |
| 一致性 | `consistency` | 同题多次运行答案一致率 |
| 幻觉率 | `hallucination` | 禁止错误主张 + 可选 grounded |

```bash
# 功能验证集
python -m react_agent.eval

# 能力评估全集
python -m react_agent.eval --dataset capability

# 只跑某一能力维度
python -m react_agent.eval --capability accuracy
python -m react_agent.eval --capability tool_selection

# 查看历史报告
python -m react_agent.eval --list
```

报告保存在 `src/react_agent/eval/reports/`（本地跑批产物，部分日期文件被 gitignore）。  
**对外公开**请用发布脚本写入 `docs/` + `docs/snapshots/`：

```bash
python examples/publish_eval_snapshot.py --from-report <eval_*.json>
python examples/publish_eval_snapshot.py --run capability   # 需 API Key
```

索引见 [docs/EVAL_INDEX.md](docs/EVAL_INDEX.md)。

### 最近一次公开结果（学习用途，样本量有限）

| 报告 | 日期 | 结果 | 说明 |
|------|------|------|------|
| [Capability 快照](docs/capability_snapshot_20260713.md) | 2026-07-13 | **18/18（100%）** | 规则打分五维；[归档 JSON](docs/snapshots/capability_snapshot_20260713.json) |
| [扩容新用例](docs/capability_newcases_20260713.md) | 2026-07-13 | **5/6（83%）** | 新增 6 条；1 例未调 calculator（诚实失败） |
| [功能向整理](docs/eval_report_20260713.md) | 2026-07-13 | **23/26（88%）** | DeepSeek；3 例角色关键词检测过严 |
| Capability 集规模 | 2026-07-13 | **24 条** | 原 18 + 新 6；索引见 [EVAL_INDEX](docs/EVAL_INDEX.md) |

与 [llm-eval-engine](https://github.com/weihuaguo270-ops/llm-eval-engine) 的 Process Reward / 人机校准打通见 `examples/agent_to_eval.py`；校准快照在 eval-engine 的 `docs/calibration_snapshot_*.md`。

### Harness 轨迹 Schema + 闭环 Demo

三仓共用 **Format B** 轨迹约定（1-based `step`，工具参数优先 `arguments` 字符串）：

| 产物 | 路径 |
|------|------|
| JSON Schema | [`schemas/harness_trajectory.schema.json`](schemas/harness_trajectory.schema.json) |
| 校验 / 归一化 | `react_agent.harness.schema` |
| 离线 fixture | `examples/fixtures/harness_closed_loop.json` |
| 一键 demo | `python examples/harness_closed_loop.py` |

闭环：`Agent 记录 → Trace Debugger 失败分类 → Eval Engine Process Reward`（CI `integration` job 会 clone 两仓并跑 demo）。

```bash
pip install -e ../trace-debugger -e ../llm-eval-engine   # 本地旁路仓
python examples/harness_closed_loop.py --fixture
python examples/harness_closed_loop.py --mock-agent
```

## 测试

```bash
# 离线单测（含 capability_scorer、resilience）
pytest tests/ -q

# 全模块脚本测试（不依赖 LLM）
python test_all.py

# 真实 LLM：CI 冒烟子集 / 全量（无 Key 时自动 skip）
pytest tests/test_real_llm.py -v -m real_llm_smoke
pytest tests/test_real_llm.py -v -m real_llm
```

### CI 与真实 LLM

| Job | 触发 | 行为 |
|-----|------|------|
| lint / test / integration | push、PR | 离线；不消耗 API |
| **Real LLM (smoke)** | push、PR（且已配置 Secret） | 跑 `real_llm_smoke`（事实问答 / 计算器 / 多步推理）；**失败会使该 job 红** |
| **Real LLM (full)** | Actions → Run workflow → suite=`full` | 全量 `real_llm` |

在仓库 **Settings → Secrets and variables → Actions** 添加 `DEEPSEEK_API_KEY`（与本地 `.env` 同名即可）。未配置时 **Real LLM gate** 会标记无 Key，smoke/full job 显示为 **Skipped**，不影响离线 CI。

也可本地写入 Secret（勿把 Key 提交进 Git）：

```bash
# 从 .env 读取一行写入 GitHub（需已 gh auth login）
gh secret set DEEPSEEK_API_KEY --repo weihuaguo270-ops/react-agent < <(grep '^DEEPSEEK_API_KEY=' .env | cut -d= -f2-)
```

仓库忽略本地 `llm_config.json`；CI / 新环境会回退到已提交的 [`llm_config.example.json`](llm_config.example.json)（Key 仍只来自环境变量 / Secret）。

## 环境要求

- Python 3.10+
- LLM API key（运行 Agent / 真实评测时需要）
- LangChain + LangGraph（仅 `experiments/langgraph/` 需要，可选）

## 相关项目

- [llm-eval-engine](https://github.com/weihuaguo270-ops/llm-eval-engine) — LLM 评估实验框架（Process Reward）
- [transformer-attention](https://github.com/weihuaguo270-ops/transformer-attention) — Attention 教学实现
- [trace-debugger](https://github.com/weihuaguo270-ops/trace-debugger) — 轨迹分析小工具

## License

MIT

## 贡献与安全

见 [CONTRIBUTING.md](CONTRIBUTING.md) / [SECURITY.md](SECURITY.md)。
