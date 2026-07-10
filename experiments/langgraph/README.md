# LangGraph 版本 — 手写 Agent 的框架重构

`experiments/langgraph/` 将手写版 Agent（`src/handwritten_react_agent/`）的核心能力用 **LangChain + LangGraph** 重新实现。这是一个"对照实验"——同一套 Agent 能力，手写版 vs 框架版，展示我对两者的理解深度。

## 设计思路

```
手写版（src/）              LangGraph 版（experiments/langgraph/）
─────────────────────       ─────────────────────────────────────
手写 ReAct Loop             StateGraph（call_model ⇄ tools）
urllib 直接调 API           ChatOpenAI（复用 llm_config.json）
TOOL_REGISTRY 字典           @tool 装饰器
字符串 prompt 拼接           角色模板 + SystemMessage
subprocess 沙箱              harness/sandbox.py（同思路实现）
手写轨迹记录器               harness/recorder.py（同格式）
```

两者**不共享代码**——LangGraph 版从零实现，只在配置层（`llm_config.json`）和轨迹文件格式上保持一致，便于 A/B 对比。

## 文件映射

| 手写模块 | LangGraph 文件 | 对照点 |
|---------|---------------|--------|
| `react_loop.py` | `graph/agent.py` | 手写 Loop vs StateGraph 节点+条件边 |
| `orchestrator.py` | `graph/orchestrator.py` | supervisor → worker → join 编排 |
| `llm.py` | `graph/llm.py` | urllib 直接请求 vs ChatOpenAI SDK |
| `tools/` | `graph/tools.py` | 字典注册 vs @tool 装饰器 |
| `prompts.py` + `cot.py` | `graph/prompts.py` | 字符串拼接 vs LangChain 模板 |
| `context.py` | `graph/context.py` | 同策略（truncate/drop/summarize） |
| `rag.py` | `graph/rag.py` | FAISS + HuggingFace Embeddings |
| `memory.py` | `graph/memory.py` | 同策略（双阈值语义去重） |
| `mcp_client.py` | `graph/mcp.py` | MCP JSON-RPC over stdio |
| `harness/recorder.py` | `graph/harness/recorder.py` | 同格式轨迹记录 |
| `harness/sandbox.py` | `graph/harness/sandbox.py` | 子进程隔离 + 白名单 |
| `harness/replay.py` | `graph/harness/replay.py` | 轨迹回放 CLI |

## 架构

### 单 Agent（`graph/agent.py`）

```
                    ┌──────────┐
                    │ 用户输入   │
                    └────┬─────┘
                         ▼
                  ┌──────────────┐
                  │  call_model   │  ← LLM 推理 + thought 记录
                  └──────┬───────┘
                         │
                    ┌────┴────┐
                    │ tool_calls?│
                    └────┬────┘
                   /              \
              有 tool_calls     无 tool_calls
                  │                  │
                  ▼                  ▼
          ┌──────────────┐   ┌──────────────┐
          │  tools_node   │   │ context_manage│ → extract_memory → END
          └──────┬───────┘   └──────────────┘
                 │
                 └──→ call_model（回到循环）
```

### 多 Agent 编排（`graph/orchestrator.py`）

```
  supervisor                   worker                    join
┌──────────────┐       ┌──────────────┐          ┌──────────┐
│  用 LLM 分解   │       │  子任务1     │          │ 合并结果  │
│  用户请求为    │ ───→ │  子任务2     │ ───────→ │ 输出最终  │
│  子任务列表    │       │  ...        │          │ 汇总     │
└──────────────┘       └──────────────┘          └──────────┘
                           每个 Worker 内部
                          是独立 StateGraph
```

- **supervisor** — 将用户请求分解为子任务列表，分析依赖关系
- **worker** — 按依赖层级执行，同层并行，有依赖的等待前置任务
- **join** — 合并所有 Worker 结果为最终汇总，含工具隔离（每个 Worker 只暴露相关工具）

## 运行方式

```bash
# 单 Agent（单次查询，含 Harness 轨迹记录）
cd experiments/langgraph/graph
python main.py "现在几点了？"

# 单 Agent（交互模式）
python main.py
# 然后输入查询，输入 /bye 退出

# 多 Agent 编排
python orchestrator.py "帮我查时间并计算 50*30"

# 轨迹回放
python -m harness.replay

# 指定 LLM Provider
LLM_PROVIDER=deepseek python main.py "写一段 Python 代码"
```

## 和手写版的对照价值

| 维度 | 手写版 | LangGraph 版 |
|------|--------|-------------|
| 循环实现 | `for step in range(max_steps)` | `StateGraph` 图执行 |
| 条件路由 | `if tool_calls` 分支 | `conditional_edge` 条件边 |
| 状态管理 | 列表追加 `messages.append(...)` | `operator.add` 自动合并 |
| 工具注册 | `TOOL_REGISTRY[name] = func` | `@tool` 装饰器 + `bind_tools` |
| 批量执行 | 无 | `ThreadPoolExecutor` 并行 Worker |
| 可检查点 | 无（全局变量） | `MemorySaver` 检查点 |

## 快速验证

```bash
# 快速验证
python orchestrator.py "现在几点了？"
python orchestrator.py "帮我查时间并计算 50*30"
```
