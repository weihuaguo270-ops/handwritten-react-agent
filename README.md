# handwritten-react-agent

手写 ReAct Agent — 零框架实现 LLM Agent 核心循环，支持交互模式、工具调用、语义记忆、文档知识检索（RAG）、思维链（CoT）、思维树（ToT）和 DAG 任务调度。

## 概述

从零实现 ReAct（Reasoning + Acting）循环，不依赖 LangChain、AutoGPT 等框架，只用 Python 标准库 + LLM API + BGE Embedding。

## 安装

### 方式一：开发模式安装（推荐）

```bash
cd D:\agent_learning\repo
pip install -e .
```

安装后可直接使用 `agent` 命令启动交互模式。

### 方式二：仅安装依赖

```bash
pip install numpy scikit-learn sentence-transformers
```

## 快速开始

### 1. 配置 API Key（必需）

⚠️ **启动检查**：程序启动时会自动检查 API Key 是否配置，若未配置会显示错误并退出：
```
错误：未配置 DEEPSEEK_API_KEY，也没有在 react_loop.py 中设置 fallback API_KEY。
```

推荐通过**环境变量**配置，避免 API Key 被误提交到 Git：

```bash
# Windows（命令行）
set DEEPSEEK_API_KEY=sk-xxx

# Windows（PowerShell）
$env:DEEPSEEK_API_KEY="sk-xxx"

# Linux / Mac
export DEEPSEEK_API_KEY='sk-xxx'
```

也可以直接修改 `react_loop.py` 第 66 行，将 `os.environ.get(...)` 的第二个参数改为你的 Key：

```python
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "这里填入你的Key")
BASE_URL = "https://api.deepseek.com"      # DeepSeek 官方 API 地址
MODEL = "deepseek-v4-flash"                 # DeepSeek V4 Flash
```

### 2. 运行

**方式一：使用 agent 命令（推荐，需先 pip install -e . 安装）**

```bash
# 交互模式
agent

# 单次查询
agent "现在几点？"

# 并行多任务
agent --parallel "搜索AI新闻并且总结量子计算进展"
```

**方式二：直接运行 Python 文件**

```bash
# 交互模式
python react_loop.py

# 单次查询
python react_loop.py "现在纽约几点？"

# 指定 MCP 服务器
python react_loop.py --mcp "uvx mcp-server-time" "现在几点？"
```

**方式三：模块方式运行**

```bash
python -m repo
```

交互模式：

```
==================================================
  Agent 交互模式已启动
  ==================================================
  可用工具：get_time / calculator / web_search / fetch_page / summarize / rag_query
  可问：时间、计算、搜索新闻、读网页、总结内容、查文档知识
  记忆：说 "记住..." 保存信息，输入 '记忆' 查看
  退出：输入 'exit' 或 '退出'
  ==================================================

你 > 记住我叫小明，学计算机的
你 > 我学的什么专业？
你 > 记忆
你 > exit
```

单次运行：

```bash
venv\Scripts\python react_loop.py "现在几点了？"
```

## 核心机制

### ReAct Loop

```
用户输入 → 调用 LLM → LLM 输出思考(Thought)
                      → 需要工具？→ 调用工具(Action)
                                    → 工具返回结果(Observation)
                                    → 继续调用 LLM 分析结果
                      → 已足够信息？→ 输出最终答案(Final Answer)
```

### 思维链（CoT）- cot.py

在 ReAct Loop 的思考阶段注入 CoT 策略，让 LLM 先逐步推理再回答。

**支持的策略：**

| 策略 | 适用场景 | 做法 |
|------|---------|------|
| Zero-shot | 通用 | 在 system prompt 末尾加"请一步一步思考" |
| Few-shot Math | 数学/计算 | 给 2 个带详细步骤的数学推理示例 |
| Few-shot Reasoning | 逻辑推理 | 给 2 个三段论/条件推理示例 |
| Structured | 复杂问题 | 强制按"分析→拆解→步骤→验证"框架思考 |

**自动策略选择：** 根据查询关键词自动判断用哪个策略，无需手动指定。

**集成方式：** `react_loop.py` 启动时通过 `COT.inject(base_prompt, query=user_query)` 将 CoT prompt 注入到 system prompt 末尾。

### 思维树（ToT）- tot.py

CoT 只有一条推理链，ToT 同时探索多条链，评估每条的质量，砍掉差的，保留好的。

```
CoT:  思考A → 思考B → 思考C → 答案（一条路走到黑）
ToT:          ┌→ 路径A1 → 评估:6分 → 继续
      思考A ─→ 路径A2 → 评估:8分 → 继续 → 选出最佳
              └→ 路径A3 → 评估:2分 → 剪掉 ✂️
```

**搜索策略：**
- **BFS（广度优先）**：每层保留 beam_width 条路径，逐层扩展（默认）
- **DFS（深度优先）**：一条路到底，不行再回溯

**核心参数：**

| 参数 | 默认值 | 含义 |
|------|--------|------|
| beam_width | 3 | 每层保留多少条路径 |
| branch_factor | 3 | 每个节点生成多少个候选 |
| max_depth | 5 | 最大搜索深度 |

**适用场景：** 开放方案设计、多方案比较、规划类问题。简单问题用 CoT 更高效。

### 任务规划（Planner）- planner.py + orchestrator.py

LLM 自动分解复杂任务为子任务，分析任务间的依赖关系，按 DAG（有向无环图）调度执行。

```
用户: "搜索今天和明天的天气，对比温差"
              ↓
[Planner] 分解为 3 个子任务，2 个执行层级:
  #1: 搜索今天北京天气（无依赖）
  #2: 搜索明天北京天气（无依赖）
  #3: 对比温差（依赖 #1, #2）

调度:
  第1层: [#1, #2]  ← 无依赖，可并行
  第2层: [#3]      ← 等前两个完成再执行
```

**依赖分析流程：**

1. `Planner.plan()` → LLM 输出结构化工单（`task_N: 描述 | depends_on: N, M`）
2. `Planner.schedule()` → 拓扑排序，计算执行层级
3. `Orchestrator.execute()` → 按层级调度：同层并行、层间串行
4. `_build_context()` → 前置任务的结果自动注入到后置任务的 prompt

### 记忆系统

基于 BGE-small-zh-v1.5 的语义记忆，支持自动提取与遗忘。

#### 增加记忆

| 方式 | 触发 | 说明 |
|------|------|------|
| 手动 | 说 `记住...` | BGE 转 512 维向量 → 持久化到 memory.json |
| 自动 | 每次对话后 | LLM 自动判断并提取事实性信息（姓名、职业、爱好等）|

#### 删除记忆

| 方式 | 命令 | 匹配方式 |
|------|------|---------|
| 精确 | `忘记 xxx` | 精确匹配事实文字 |
| 关键词 | `忘记 张三` | 关键词包含匹配（`"张三" in "用户的姓名是张三"`）|
| 语义 | `忘记 xxx` | BGE 余弦相似度 > 0.4（兜底）|
| 全部 | `删除所有记忆` | 清空 memory.json |

#### 自动遗忘

记忆超过 100 条时，按使用频率 + 最后使用时间自动移除最不常用的。

#### 使用计数

每次命中检索自动增加访问计数，最长未使用的记忆优先被遗忘。

#### 查看

输入 `'记忆'` 查看所有已保存的记忆。

## 已实现工具

| 工具名 | 功能 | 数据源 |
|--------|------|--------|
| `get_current_time(tz)` | 获取指定时区时间 | MCP mcp-server-time |
| `convert_time(...)` | 时区转换 | MCP mcp-server-time |
| `get_time()` | 获取当前时间（MCP连接时自动隐藏） | 本地 |
| `calculator(expression)` | 计算数学表达式 | Python eval |
| `web_search(query)` | 搜索互联网新闻 | AnySearch 搜索引擎 |
| `fetch_page(url)` | 读取网页正文 | 维基API/HTML提取 |
| `summarize(text)` | 自动提取摘要 | 抽取式 |
| `rag_query(query, top_k)` | 从本地文档库检索知识 | BGE 语义搜索（RAG） |
| `switch_cot_strategy(strategy)` | 运行时切换 CoT 推理策略 | cot.py |
| `tot_reasoning(problem)` | 使用思维树进行多路径推理 | tot.py |
| `switch_role(role)` | 切换 AI 角色风格 | prompts.py |
| `switch_context_strategy(strategy)` | 切换上下文窗口管理策略 | context.py |
| `toggle_sandbox(enabled)` | 开启/关闭工具沙箱隔离 | sandbox.py |
| `read_text_file(path)` | 读取文件内容 | MCP server-filesystem |
| `write_file(path, content)` | 写入文件 | MCP server-filesystem |
| `edit_file(path, edits)` | 编辑文件（行级替换）| MCP server-filesystem |
| `list_directory(path)` | 列出目录内容 | MCP server-filesystem |
| `create_directory(path)` | 创建目录 | MCP server-filesystem |
| `move_file(src, dst)` | 移动/重命名文件 | MCP server-filesystem |
| `search_files(pattern)` | 搜索文件 | MCP server-filesystem |
| `get_file_info(path)` | 获取文件元信息 | MCP server-filesystem |
| `directory_tree(path)` | 递归目录树 | MCP server-filesystem |

### Pipeline 示例

```
用户: 搜索AI Agent的维基百科词条，打开第一条，总结内容
  → web_search("AI Agent Wikipedia")       # 搜索
  → fetch_page("en.wikipedia.org/...")     # 读全文
  → summarize(全文)                         # 总结
  → LLM 综合回答                             # 输出
```

## 自动化评测

```bash
# 单元测试（无需 API Key）
python test_all.py

# 端到端集成测试（需要 API Key）
venv\Scripts\python eval.py
```

`test_all.py` 覆盖 46 项单元测试（CoT、ToT、Planner、RoleManager、Context、Harness、Sandbox、Replay）。
`eval.py` 覆盖 12 个端到端测试用例。

## 项目结构

```
handwritten-react-agent/
├── react_loop.py    # 主代码（ReAct Loop + 工具 + 记忆 + 交互模式）
├── cot.py           # 思维链（CoT）策略模块
├── tot.py           # 思维树（ToT）推理模块
├── planner.py       # 任务规划器（LLM 驱动分解 + 依赖分析）
├── orchestrator.py  # 多 Agent 协作（DAG 调度，按依赖层级执行）
├── prompts.py       # 角色 Prompt 模板库（5 种角色风格）
├── context.py       # 上下文窗口管理（截断/丢弃/摘要/LLM 自动选择）
├── harness.py       # 轨迹记录器（每一步写入 JSON 文件）
├── sandbox.py       # 工具沙箱隔离（子进程执行，崩溃不炸主进程）
├── replay.py        # 轨迹重放器（回放 Harness 记录的步骤）
├── mcp_client.py    # MCP 协议模块（JSON-RPC 2.0 over stdio）
├── rag.py           # RAG 检索增强生成模块（文档分块/向量化/语义搜索）
├── memory.py        # 语义记忆模块（增删查 + 自动遗忘）
├── eval.py          # 端到端自动化评测（12 个测试用例）
├── test_all.py      # 单元测试（46 项，无需 API Key）
├── trajectories/    # 轨迹文件目录（自动生成，不提交）
├── memory.json      # 记忆持久化（自动生成）
├── rag_index.json   # RAG 知识库索引（自动生成，不提交）
├── README.md
└── LICENSE
```

## 依赖

- Python 3.8+
- numpy
- scikit-learn
- sentence-transformers（首次加载约 13 秒）

## RAG 文档检索

Agent 启动时自动索引项目目录下的 `.py`、`.md` 文件，存入 RAG 知识库。对话中 Agent 自主判断何时调用 `rag_query` 查询本地文档。

```
用户: "mcp_client.py 是做什么的？"

Agent → rag_query("mcp_client.py 功能")
       → 检索到相关代码片段
       → 结合上下文回答
```

基于 BGE-small-zh-v1.5 语义搜索，支持段落级分块、去重、余弦相似度筛选（min_score=0.25）。

## 多 Agent 协作（Orchestrator-Worker）- DAG 调度

将复杂请求自动拆分为多个子任务，分析任务间的依赖关系，按 DAG（有向无环图）调度执行——同层并行、层间串行。

### 示例

```
用户: 搜索今天AI领域的最新新闻，然后用中文总结，最后写一个Twitter帖子

[Orchestrator] 分解为 3 个子任务，3 个执行层级:
  #1: 搜索今天AI领域的最新新闻        （无依赖）
  #2: 用中文总结搜索到的AI新闻        （依赖 #1）
  #3: 写一个关于AI新闻的Twitter帖子   （依赖 #2）

调度:
  第1层: #1 搜索新闻         ← 先执行
  第2层: #2 总结新闻         ← 等#1完成
  第3层: #3 写Twitter帖子    ← 等#2完成

[层级 1/3] #1 → 搜索AI新闻成功
[层级 2/3] #2 → 收到#1结果作为上下文 → 总结完成
[层级 3/3] #3 → 收到#2结果 → 输出帖子
```

### Worker 隔离

每个 Worker 只能看到自己需要的工具，避免 LLM 选错：

| 子任务 | 暴露的工具数 | 可用的工具 |
|--------|------------|-----------|
| 查询纽约时间 | 4/20 | get_current_time, convert_time, web_search, fetch_page |
| 查看文件大小 | 10/20 | read_text_file, write_file, list_directory, get_file_info ... |

分类规则（关键词匹配，无需额外 LLM 调用）：

| 分类 | 触发关键词 | 包含工具 |
|------|-----------|---------|
| time | 时间、时区、当前时间、纽约 | get_current_time, convert_time |
| file | 文件、目录、大小、读写 | 全部 filesystem 工具 |
| web | 搜索、网页、新闻、查询 | web_search, fetch_page |
| calc | 计算、数学 | calculator |
| summary | 总结、摘要、概括 | summarize |

### 上下文传递

后置任务自动收到前置任务的结果作为参考信息，无需重复搜索。

### 并行执行

同层无依赖的任务自动并行执行，使用 `concurrent.futures.ThreadPoolExecutor`，每个 Worker 独立线程 + 独立工具快照。

## 实现

所有模块可独立导入使用：

```python
# CoT 思维链
from cot import COT
system_prompt = COT.inject(base_prompt, query="一个篮球120元...")

# ToT 思维树
from tot import ToT
result = ToT().solve("复杂问题", llm_call=my_llm)

# Planner 任务规划
from planner import Planner
tasks = Planner().plan("搜索并总结新闻", llm_call=my_llm)

# Orchestrator DAG 调度
from orchestrator import Orchestrator
o = Orchestrator(call_llm, react_loop)
o.execute("搜索AI新闻并写帖子")

# Context 上下文管理
from context import CONTEXT
messages = CONTEXT.manage(messages)  # 在每步后调用

# Harness 轨迹记录
from harness import start_trajectory, finish_trajectory
start_trajectory(query, model, system_prompt)
# ... 运行 ReAct Loop ...
path = finish_trajectory(final_answer)

# Replay 重放（命令行）
# python replay.py --latest
```

## 上下文窗口管理（Context Engineering）

防止 ReAct Loop 因对话历史过长超出 LLM 上下文限制。每步结束后自动检查 token 用量，超限时按策略处理。

**支持策略：**

| 策略 | 做法 | 开销 |
|------|------|------|
| **auto（默认）** | LLM 根据当前对话内容选择最优策略 | 超限时多 1 次 LLM 调用 |
| truncate | 从最早的非 system 消息开始删 | 0 额外 LLM 调用 |
| drop | 只删除已执行完毕的 tool_call + tool_result 对 | 0 额外 LLM 调用 |
| summarize | 把早期对话压缩成一段摘要 | 1 次 LLM 调用 |

## Harness Engineering

Agent 运行保障层，共五层：

| 层 | 模块 | 职责 |
|---|------|------|
| ① Tool 注册与调用 | `react_loop.py` | TOOL_REGISTRY + MCP 自动路由 |
| ② Observation 回路 | `react_loop.py` | 工具结果自动追加到 messages |
| ③ 沙箱隔离 | `sandbox.py` | 每个工具在独立子进程执行，崩溃不炸主进程 |
| ④ 可观测性 | `harness.py` | 每次对话自动写入 JSON 轨迹文件 |
| ⑤ 重放调试 | `replay.py` | 读轨迹文件逐步回放（`python replay.py --latest`）|

**轨迹文件示例：**

```json
{
  "session_id": "20260702_151849_x8kn",
  "query": "一个篮球120元...",
  "total_steps": 2,
  "steps": [
    {"step": 1, "thought": "先算足球单价...", "action": {"name": "calculator"}},
    {"step": 2, "thought": "FINAL ANSWER: 450元"}
  ],
  "final_answer": "450 元"
}
```

## 后续计划

- [x] MCP 协议支持
- [x] 多 Agent 协作（Orchestrator-Worker）
- [x] RAG 文档检索
- [x] 思维链（CoT）
- [x] 思维树（ToT）
- [x] DAG 任务规划（依赖分析 + 层级调度）
- [x] 角色注入（5 种角色风格）
- [x] 上下文窗口管理（截断/丢弃/摘要/auto）
- [x] Harness Engineering（轨迹记录 + 沙箱 + 重放）
- [ ] Web UI 界面
- [ ] 沙箱子进程启动优化（预热缓存）

## License

MIT
