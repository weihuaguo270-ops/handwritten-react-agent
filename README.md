# handwritten-react-agent

> 从零手写 LLM Agent Runtime — 不依赖 LangChain、AutoGPT 等框架，只用 Python 标准库 + LLM API + BGE Embedding。
> 覆盖工具调用、记忆、RAG、多 Agent 编排、推理增强、评测 Harness 与安全沙箱全链路。

## 架构总览

项目按**层级**组织，每个层级解决不同粒度的问题：

| 层级 | 模块 | 职责 |
|------|------|------|
| **① 任务层** | `planner.py` → `orchestrator.py` | 把一句话需求拆成多个子任务，按 DAG 调度执行 |
| **② 对话层** | `react_loop.py`（ReAct Loop） | 每个子任务走"思考→行动→观察"循环 |
| **③ 推理层** | `cot.py` / `tot.py` | 增强 LLM 在"思考"这一步的推理质量 |
| **④ 旁路层** | `memory.py` / `rag.py` / `context.py` / `harness.py` / `sandbox.py` / `replay.py` | 贯穿全流程：记忆读写、文档检索、上下文控制、轨迹记录、安全执行、重放调试 |

**执行流程：**

```
用户输入
  → [任务层] Planner 拆成子任务 → Orchestrator 按 DAG 调度
     → [对话层] 每个子任务进入 ReAct Loop
        → [推理层] Thought 步可增强为 CoT / ToT
        → Action 步调工具
        → Observation 步看结果
        → 循环直到完成
     ← 子任务结果汇总
  ← 最终答案

## 快速开始

### 1. 配置 API Key

```bash
# Windows
set DEEPSEEK_API_KEY=sk-xxx
# Linux / Mac
export DEEPSEEK_API_KEY='sk-xxx'
```

### 2. 运行

```bash
# 交互模式
python react_loop.py

# 单次查询
python react_loop.py "现在几点？"
```

### 3. 运行测试

```bash
# 单元测试（无需 API Key）
python test_all.py

# 端到端测试（需 API Key）
python eval.py
```

## 核心模块

### ReAct Loop（react_loop.py）

Agent 主循环：**思考 → 行动 → 观察 → 再思考 → 最终答案**。每步支持：

- 多 tool_call 并行执行
- 上下文漂移检测与寒暄兜底
- 最大步数限制（默认 10 步）
- 交互模式与单次运行模式

### 推理增强（cot.py / tot.py）

CoT（思维链）和 ToT（思维树）是两个互补的推理增强层：

| 维度 | CoT | ToT |
|------|-----|-----|
| 路径 | 单条推理链 | 多条路径并行搜索 |
| 策略 | 4 种自动切换 | BFS / DFS + 评分剪枝 |
| 成本 | 0 次额外 LLM 调用 | 每步 N 次生成 + N 次评估 |
| 适用 | 常规问题 | 多方案比较、规划类问题 |

CoT 支持 4 种策略，根据用户问题关键词自动选择：

| 策略 | 触发场景 | 做法 |
|------|---------|------|
| Zero-shot | 通用、搜索类 | 加一句"请逐步思考" |
| Few-shot Math | 数学/计算 | 给 2 个带步骤的数学示例 |
| Few-shot Reasoning | 逻辑推理 | 给 2 个三段论推理示例 |
| Structured | 复杂长问题（≥40字符，≥3个逗号） | 强制按"分析→拆解→步骤→验证"框架思考 |

### 任务规划与多 Agent（planner.py / orchestrator.py）

Planner 负责将复杂请求分解为子任务并分析依赖关系；Orchestrator 负责按 DAG 调度执行。

```
用户 → Planner 分解 → 拓扑排序 → 第1层(并行) → 第2层(串行) → 完成
```

**Planner 分层分解：** 模板匹配（零 LLM 调用）优先，未命中才走 LLM 兜底。

**Worker 隔离：** 每个 Worker 只暴露当前任务所需工具，避免 LLM 选择错误。

**上下文传递：** `_build_context()` 将前置任务结果注入后置任务 prompt，避免重复搜索。

### 记忆与 RAG（memory.py / rag.py）

| 模块 | 存储 | 检索 | 遗忘 |
|------|------|------|------|
| Memory | BGE 512 维向量 → memory.json | 余弦相似度 Top-3 | LRU（超 100 条） |
| RAG | 文档分块 → BGE 索引 → rag_index.json | 余弦相似度 Top-K（min_score=0.25） | 索引启动时重建 |

**记忆写入方式：**
- 手动：说"记住 xxx"
- 自动：每次对话后 LLM 提取事实性信息

**记忆删除方式：** 精确 / 关键词 / 语义 / 全部，4 级删除。

### 上下文管理（context.py）

| 策略 | 做法 | 额外 LLM 成本 |
|------|------|--------------|
| **auto（默认）** | LLM 根据对话状态选择最优策略 | 超限时 1 次 |
| truncate | 从最早非 system 消息开始删 | 0 |
| drop | 仅删已执行的 tool_call→tool_result 对 | 0 |
| summarize | 将早期对话压缩为摘要 | 1 次 |

### Harness / Sandbox / Replay（harness.py / sandbox.py / replay.py）

- **Harness：** 每步 thought/action/observation/token_usage 持久化为 JSON
- **Sandbox：** subprocess + timeout 隔离不可信代码，AST 白名单安全解析
- **Replay：** `python replay.py --latest` 从轨迹文件逐步回放

## 已实现工具

| 工具名 | 功能 | 来源 |
|--------|------|------|
| `get_current_time(tz)` | 获取指定时区时间 | MCP |
| `calculator(expression)` | 计算数学表达式（AST 安全解析） | 内置 |
| `web_search(query)` | 搜索互联网 | AnySearch |
| `fetch_page(url)` | 读取网页正文 | 内置 |
| `summarize(text)` | 自动文字摘要 | 内置 |
| `rag_query(query, top_k)` | 从本地文档库检索知识 | BGE |
| `tot_reasoning(problem)` | 思维树多路径推理 | tot.py |
| `switch_cot_strategy(s)` | 切换 CoT 推理策略 | cot.py |
| `switch_role(role)` | 切换 AI 角色风格 | prompts.py |
| `switch_context_strategy(s)` | 切换上下文管理策略 | context.py |
| `toggle_sandbox(enabled)` | 开启/关闭沙箱隔离 | sandbox.py |
| `read_text_file(path)` | 读取文件内容 | MCP |
| `write_file(path, content)` | 写入文件 | MCP |
| `edit_file(path, edits)` | 行级文件编辑 | MCP |
| `list_directory(path)` | 列出目录内容 | MCP |
| `create_directory(path)` | 创建目录 | MCP |
| `move_file(src, dst)` | 移动/重命名文件 | MCP |
| `search_files(pattern)` | 搜索文件 | MCP |
| `get_file_info(path)` | 获取文件元信息 | MCP |
| `directory_tree(path)` | 递归目录树 | MCP |

## 项目结构

```
├── react_loop.py       # 主循环（ReAct + 工具 + 记忆 + 交互）
├── cot.py              # 思维链（4 种策略自动切换）
├── tot.py              # 思维树（BFS/DFS 双搜索模式）
├── planner.py          # 任务规划器（模板+LLM 分层分解）
├── orchestrator.py     # 多 Agent DAG 调度
├── prompts.py          # 角色注入（5 种风络）
├── context.py          # 上下文窗口管理（4 种策略）
├── harness.py          # 轨迹记录器（JSON 持久化）
├── sandbox.py          # 子进程沙箱隔离
├── replay.py           # 轨迹重放器
├── mcp_client.py       # MCP 协议客户端
├── rag.py              # RAG 检索增强生成
├── memory.py           # 语义记忆系统
├── eval.py             # 端到端评测（12 个测试用例）
├── test_all.py         # 单元测试（46 项，无需 API Key）
├── trajectories/       # 轨迹文件
├── README.md
└── LICENSE
```

## 安装

```bash
# 开发模式安装
pip install -e .

# 或仅安装依赖
pip install numpy scikit-learn sentence-transformers
```

## 依赖

- Python 3.8+
- numpy
- scikit-learn
- sentence-transformers

## 后续计划

- [x] MCP 协议支持
- [x] 多 Agent 协作
- [x] RAG 文档检索
- [x] 思维链（CoT）
- [x] 思维树（ToT）
- [x] DAG 任务规划
- [x] 角色注入
- [x] 上下文窗口管理
- [x] Harness / Sandbox / Replay
- [ ] Agent Web 界面
- [ ] 沙箱子进程预热缓存

## License

MIT
