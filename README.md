# handwritten-react-agent

> 从零手写 LLM Agent Runtime — 不依赖 LangChain、AutoGPT 等框架，只用 Python 标准库 + LLM API + BGE Embedding。
> 覆盖工具调用、记忆、RAG、多 Agent 编排、推理增强、评测 Harness、安全沙箱、Eval 评测管线与 Dashboard 可视化全链路。

## 架构总览

### 运行时全链路

用户输入一条 query 到获得最终答案，各模块介入的时机与触发条件如下：

```
query 输入
  │
  ├── 普通入口（直接） → react_loop()
  │     ↓
  └── Orchestrator 入口
        │
        ├── (A) Orchestrator.execute() 被调用
        │       ↓
        ├── (B) Orchestrator.plan() → Planner.plan() 分解任务
        │       触发条件: LLM 被调用，解析 task_N/depends_on 格式
        │       不需要: 单任务场景（不触发 Planner）
        │
        ├── (C) Orchestrator.run_worker() → 每个子任务独立走 react_loop()
        │       注意: Planner 和 Orchestrator 只在 Orchestrator 入口时触发
        │             普通入口不涉及 Planner/Orchestrator
        │
        └── (D) Orchestrator.synthesize() 汇总各 Worker 结果 → 输出
             触发条件: 所有层级执行完毕


react_loop 入口（普通或 Worker）:
  │
  ├── step -1: LLM 配置加载
  │     llm.LLM_DEFAULT 按 LLM_PROVIDER/llm_config.json 确定 provider
  │     react_loop.py 通过 _current_llm.chat() 统一调用，不直接处理 HTTP
  │     切换模型: 一行 export LLM_PROVIDER=openai，不改代码
  │
  ├── step 0: 构建 system prompt（三层拼接）
  │     (1) base_prompt: ReAct 格式规则
  │     (2) ROLE_MANAGER.inject() → 根据 query 关键词选角色
  │     (3) COT.inject() → 根据 query 关键词选 CoT 策略
  │     ↑ 注意：CoT 只在这里（system prompt 构建时）起作用。
  │       ToT 不在这里，它是作为一个工具被注册的。
  │
  ├── step 0: Memory 检索（如果有记忆）
  │     触发时机: 每次运行 react_loop() 时，Memory 将已保存的事实
  │              拼接进 system prompt 的思路部分。
  │     不触发: memory.json 为空时。
  │
  ├── step 0: Harness 轨迹记录（手写版 harness/recorder.py + LangGraph 版 experiments/langgraph/graph/harness/recorder.py）
  │     触发时机: 每次 react_loop() 被调用时必定执行。
  │     记录内容: session_id / query / model / system_prompt
  │

  └── ReAct Loop:
        for step in range(1, max_steps + 1):
          │
          ├── (a) call_llm(messages, tool_defs)
          │      → LLM 返回: thought + 可选的 tool_calls
          │      → Harness 轨迹记录: Recorder.add_thought(step, LLM_回复) 或 LangGraph 版自动记录
          │
          ├── (b) 检查 tool_calls 是否为空
          │     │
          │     ├── tool_calls == []:
          │     │     ├── 含 FINAL ANSWER: 标记?
          │     │     │   → finish_trajectory(答案) → return 答案
          │     │     ├── 上一步调了工具 + 本次有实质输出?
          │     │     │   → 视为隐式答案 → finish_trajectory() → return
          │     │     ├── 寒暄检测（≥4步无工具 + 内容<10字）?
          │     │     │   → finish_trajectory(空) → return
          │     │     └── 都不是 → continue 进入下一轮
          │     │
          │     └── tool_calls != []:
          │           │
          │           ├── (c) 执行工具调用（每个 tool_call 独立执行）
          │           │
          │           │  TOOL_REGISTRY 查找顺序:
          │           │    1. 本地注册表（TOOL_REGISTRY 字典）
          │           │       ├── SANDBOX.strategy 策略判断:
          │           │       │   ├── off → 全部直接执行
          │           │       │   ├── auto → safe 工具直接，io/cpu 走子进程
          │           │       │   └── on → 全部子进程隔离执行
          │           │
          │           │    2. 不在 TOOL_REGISTRY → 遍历 MCP_CLIENTS
          │           │       ├── 工具名在 MCP 工具列表?
          │           │       │   → mcp_client.call_tool(name, args)
          │           │       │     MCP 通过 JSON-RPC over stdio 通信
          │           │       └── 都不在 → 返回"未知工具"错误
          │           │
          │           │    触发 Harness 轨迹记录:
          │           │      手写版: Recorder.add_tool_call(step, name, args, result)
          │           │      LangGraph 版: 自动记录 action/observation/duration
          │           │
          │           ├── (d) ToT 何时介入？
          │           │     触发时机: LLM 在 tool_calls 里选择了
          │           │               tot_reasoning 工具时才触发。
          │           │     ToT 是一个普通工具，不是系统层模块。
          │           │     过程: tot.solve() 内部多次 call_llm()。
          │           │     结果: 最终字符串返回 → 追加到 messages。
          │           │
          │           └── (e) 工具结果追加到 messages
          │                 messages.append({role: "tool", content: result})
          │
          └── (f) CONTEXT.manage(messages) — 每步结束后执行
                    触发条件: messages 总 token 超过阈值时。
                    策略: auto(默认) / truncate / drop / summarize
                    不触发: token 未超限时。
  │
  └── 输出最终答案（发生位置: react_loop() 的 return 语句）
        输出来源:
        (1) FINAL ANSWER: 正则匹配 → finish_trajectory() → return
        (2) 工具后无工具调用 + 实质内容 → finish_trajectory() → return
        (3) 步数耗尽 → finish_trajectory() → return
```

### 各模块触发时机汇总

| 模块 | 触发条件 | 介入位置 | 数据流向 |
|------|---------|---------|---------|
| Memory | react_loop 每次启动时 | 拼接进 system prompt | memory.json → system prompt |
| Planner | Orchestrator 入口时 | Orchestrator.plan() | LLM → 子任务列表 |
| Orchestrator | query 发给 Orchestrator.execute() 时 | 外部包装 | query → 子任务 → Worker 汇总 |
| CoT | react_loop 启动时、system prompt 构建阶段 | base_prompt 拼接 | 向 system_prompt 末尾追加推理指令 |
| ToT | 仅当 LLM 选择 tot_reasoning 工具时 | ReAct Loop 内工具执行阶段 | 工具调用 → 内部多轮 LLM 调用 → 结果字符串 |
| LLM 调用 | call_llm() 被调用时 | llm.py → _current_llm.chat() | 按 LLM_PROVIDER 选择配置 → HTTP POST /chat/completions |
| 工具路由 | LLM 返回 tool_calls 时 | TOOL_REGISTRY 查找（tools/ 模块） | 函数名 → tools.TOOL_REGISTRY[name]() 或遍历 MCP_CLIENTS |
| Context | ReAct Loop 每步结束后 | messages.append 之后 | 检查 token → 可选压缩/截断 |
| RAG | react_loop 首次启动时（懒加载） | RAG_INDEX.ingest_directory() → query 时检索 | 文档片段 → BGE 向量 → rag_index.json |
| Harness | react_loop 进入/退出/每步工具调用 | start_trajectory / add_thought / add_tool_call / finish_trajectory | 持久化到 trajectories/*.json |
| Dashboard | 用户主动启动 | 独立 Flask Web 服务（端口 5050） | 读取 trajectories/ → 轨迹回放；读取 eval/reports/ → 评测看板 |
| Eval（评测） | `python -m handwritten_react_agent.eval` 或 Dashboard 📊 页面 | eval/runner → subprocess 调 react_loop.py | 轨迹 → scorer → eval/reports/*.json |
## 快速开始

### 1. 安装与配置

```bash
# 安装项目
pip install -e .

# 配置 API Key（首次运行）
agent --setup

# 或者直接设置环境变量
export DEEPSEEK_API_KEY=sk-xxx
# export LLM_PROVIDER=openai    # 切换 Provider（可选）
```

支持任意 OpenAI 兼容 API（DeepSeek / OpenAI / Ollama / 自定义），通过配置文件 `llm_config.json` 或环境变量切换，无需修改代码。

```json
{
  "default": "deepseek",
  "providers": {
    "deepseek": {
      "base_url": "https://api.deepseek.com",
      "api_key_env": "DEEPSEEK_API_KEY",
      "model": "deepseek-v4-flash"
    }
  }
}
```

### 2. 运行

```bash
# 交互模式（推荐）
agent

# 单次查询
agent "现在几点？"

# 或直接运行模块
python -m handwritten_react_agent.react_loop "现在几点？"
```

### 3. 运行测试

```bash
# 单元测试（无需 API Key，当前 87+ 项覆盖 12 个模块）
python test_all.py

# 端到端评测（需配置 API Key）
python -m handwritten_react_agent.eval

# 按功能模块筛选运行（只测本地工具，30秒完成）
python -m handwritten_react_agent.eval --tag tool_local

# 查看历史评测报告
python -m handwritten_react_agent.eval --list

# 指定 Provider 运行评测
python -m handwritten_react_agent.eval --provider deepseek

# 启动 Dashboard（浏览器查看轨迹 + 评测结果）
python -m handwritten_react_agent.tools.dashboard
```

### 交互模式示例

```
==================================================
  Agent 交互模式已启动
  ==================================================
  可用工具：get_time / calculator / web_search /
           fetch_page / summarize / rag_query
  可问：时间、计算、搜索新闻、读网页、总结内容、查文档知识
  记忆：说"记住..."保存信息，输入'记忆'查看
  退出：输入'exit'或'退出'
  ==================================================

你 > 记住我叫小明，学计算机的
你 > 我学的什么专业？
你 > 记忆
你 > exit
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

### Harness 层（harness/ + experiments/langgraph/graph/harness/）

Harness = Recorder（轨迹记录）+ Sandbox（沙箱隔离）+ Replay（回放调试），对应 Agent = LLM + Harness 中的保障层。

**手写版（handwritten_react_agent/harness/）：**
- **Recorder：** 每步 thought/action/observation/token_usage 持久化为 JSON
- **Sandbox：** 三策略模式——off(全部直接)/auto(自动按风险判断,默认)/on(全部隔离)
- **Replay：** `python -m handwritten_react_agent.harness.replay --latest` 从轨迹文件逐步回放

**LangGraph 版（experiments/langgraph/graph/harness/）：**
在 LangGraph 的 StateGraph 节点函数中，通过 `config["configurable"]["harness"]` 透传 Harness 实例：
- **TrajectoryRecorder：** `call_model` 节点自动记录 thought，`tools_node` 节点自动记录每个 tool_call
- **Sandbox：** 三策略（off/auto/on），独立 `classify_risk()` 判断，MCP/HTTP 默认 safe
- **Replay：** `python -m experiments.langgraph.graph.harness.replay --step` 兼容手写版和 LangGraph 版的轨迹格式

**Dashboard：** 启动后浏览器打开 http://127.0.0.1:5050，左侧对话实时运行，右侧轨迹逐步骤回放，📊 评测页面查看通过率与失败案例。

### Eval 评测管线（eval/）

端到端 Agent 评测系统，替代旧版 `eval.py`：

| 组件 | 文件 | 作用 |
|------|------|------|
| 测试用例集 | `eval/dataset.json` | 26 条用例，覆盖 12 个功能模块 |
| 用例加载器 | `eval/dataset.py` | 从 JSON 加载，支持按 tag 筛选 |
| 批量执行器 | `eval/runner.py` | subprocess 调 react_loop.py + 集成 Recorder 记录轨迹 |
| 评分器 | `eval/scorer.py` | 4 维评分：工具匹配 / 内容匹配 / 步数控制 / 答案存在 |
| 报告生成器 | `eval/report.py` | 汇总报告 → `eval/reports/` 目录 |
| 统一入口 | `eval/__init__.py` | `python -m handwritten_react_agent.eval` 命令行 + EvalRunner API |

**数据流：** 执行 → `trajectories/`（轨迹）→ scorer → `eval/reports/*.json` → Dashboard 📊 页面展示通过率、按 tag 分组统计、失败案例（可点击跳转轨迹回放）。

**CLI 用法：**
```bash
python -m handwritten_react_agent.eval                       # 全部 26 条
python -m handwritten_react_agent.eval --tag tool_local      # 只测本地工具（30秒快速验证）
python -m handwritten_react_agent.eval --list                # 查看历史报告
python -m handwritten_react_agent.eval --provider openai     # 指定 LLM provider
```

### LLM 配置模块（llm.py + llm_config.json）

LLM 调用被抽象为独立模块，支持任意 OpenAI 兼容 API，通过配置文件驱动：

- **多 Provider：** 内置 DeepSeek / OpenAI / Ollama / 自定义，通过 `LLM_PROVIDER` 环境变量切换
- **配置优先：** 环境变量 > 配置文件的 api_key 字段 > 空
- **零改动换模型：** `export LLM_PROVIDER=openai`，不改一行代码
- **统一调用：** react_loop.py 通过 `_current_llm.chat()` 调用，不直接处理 HTTP

### 工具模块（tools/）

所有本地工具集中在 `tools/` 目录，统一注册和管理：

- **独立文件：** 每个工具一个 .py 文件（calculator / web_search / fetch_page / summarize / get_time / dashboard / execute_python）
- **统一入口：** `tools/__init__.py` 自动合并 TOOL_REGISTRY + TOOL_DEFINITIONS
- **即插即用：** 新增工具只需在 tools/ 下加文件，不改主循环
- **兼容性：** 与 RAG / MCP 等外部工具源共存，LLM 通过同一 TOOL_REGISTRY 查找

## 已实现工具

| 工具名 | 功能 | 来源 |
|--------|------|------|
| `execute_python(code, timeout)` | 子进程隔离执行 Python 代码并返回 stdout/stderr（超时保护） | tools/execute_python.py |
| `get_time()` | 获取当前日期时间 | tools/get_time.py |
| `calculator(expression)` | 计算数学表达式（AST 安全解析） | tools/calculator.py |
| `web_search(query)` | 搜索互联网（DuckDuckGo，手写版；AnySearch，graph 版） | tools/web_search.py |
| `fetch_page(url)` | 读取网页正文 | tools/fetch_page.py |
| `summarize(text)` | 自动文字摘要 | tools/summarize.py |
| `rag_query(query, top_k)` | 从本地文档库检索知识 | BGE |
| `tot_reasoning(problem)` | 思维树多路径推理 | tot.py |
| `switch_cot_strategy(s)` | 切换 CoT 推理策略 | cot.py |
| `switch_role(role)` | 切换 AI 角色风格 | prompts.py |
| `switch_context_strategy(s)` | 切换上下文管理策略 | context.py |
| `toggle_sandbox(strategy)` | 切换沙箱三策略（off/auto/on） | harness/sandbox.py |
| `start_dashboard(port)` | 启动 Dashboard Web 界面 | tools/dashboard.py |
| `clear_trajectories(days)` | 清理历史轨迹文件 | harness/recorder.py |
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
├── src/
│   └── handwritten_react_agent/  # 手写版 Agent 核心代码
│       ├── __init__.py           # 包入口
│       ├── react_loop.py         # 主循环 + CLI 入口
│       ├── llm.py                # LLM 调用封装（多 Provider）
│       ├── llm_config.json       # LLM 配置文件模板
│       ├── tools/                # 工具模块
│       │   ├── __init__.py       # 统一注册入口
│       │   ├── get_time.py       # 获取当前时间
│       │   ├── calculator.py     # 数学计算
│       │   ├── web_search.py     # 互联网搜索
│       │   ├── fetch_page.py     # 网页正文提取
│       │   ├── summarize.py      # 文本摘要
│       │   ├── dashboard.py      # Dashboard 启动
│       │   └── execute_python.py # Python 代码执行
│       ├── cot.py                # 思维链（4 种策略）
│       ├── tot.py                # 思维树（BFS/DFS）
│       ├── planner.py            # 任务规划器
│       ├── orchestrator.py       # 多 Agent DAG 调度
│       ├── prompts.py            # 角色注入（5 种风格）
│       ├── context.py            # 上下文窗口管理
│       ├── mcp_client.py         # MCP 协议客户端
│       ├── rag.py                # RAG 检索增强
│       ├── memory.py             # 语义记忆系统
│       ├── eval.py               # [已弃用] 旧版评测
│       ├── eval/                 # Agent Eval 评测管线
│       │   ├── dataset.py        # 用例加载器
│       │   ├── runner.py         # 批量执行
│       │   ├── scorer.py         # 4 维评分
│       │   ├── report.py         # 报告生成
│       │   └── __init__.py       # EvalRunner 入口
│       ├── harness/              # Harness 层
│       │   ├── __init__.py       # 统一入口
│       │   ├── recorder.py       # 轨迹记录
│       │   ├── sandbox.py        # 子进程隔离
│       │   ├── replay.py         # 离线回放
│       │   └── _sandbox_runner.py
│       └── dashboard/            # Dashboard Web 界面
│           ├── server.py         # Flask 服务
│           ├── index.html        # 前端页面
│           └── kill_old.py
├── __main__.py           # pip install 后的入口 (python -m handwritten_react_agent)
├── pyproject.toml        # 项目配置（pip install 用）
├── test_all.py           # 单元测试（87+ 项）
├── README.md
├── LICENSE
├── .gitignore
├── llm_config.json       # LLM 配置（用户编辑）
├── trajectories/         # 轨迹文件（运行时生成）
├── notes/                # 开发笔记
├── experiments/
│   └── langgraph/       # LangChain/LangGraph 全栈版本（与手写版并存）
│       ├── graph/
│       │   ├── agent.py       # 单 Agent StateGraph
│       │   ├── orchestrator.py# 多 Agent 编排
│       │   ├── llm.py         # ChatOpenAI 封装
│       │   ├── tools.py       # @tool 装饰器定义
│       │   ├── rag.py         # FAISS + HuggingFaceEmbeddings
│       │   ├── memory.py      # 语义记忆系统
│       │   ├── prompts.py     # 角色模板 + CoT 注入
│       │   ├── context.py     # 上下文管理
│       │   ├── mcp.py         # MCP 客户端连接
│       │   ├── main.py        # CLI 入口
│       │   └── harness/       # Harness 层（轨迹记录 + 沙箱 + 回放）
│       │       ├── recorder.py
│       │       ├── sandbox.py
│       │       └── replay.py
│       └── __init__.py
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

## LangGraph / LangChain 版本

项目同时提供了 `experiments/langgraph/graph/` 目录下的 **LangChain/LangGraph 全栈版本**，与手写版模块一一对应：

| 手写模块 | LangChain 替代 | experiments/langgraph/graph/ 文件 | 状态 |
|---------|---------------|------------|------|
| `llm.py` | `ChatOpenAI`（读 `llm_config.json` 配置） | `experiments/langgraph/graph/llm.py` | ✅ |
| `tools/` + `TOOL_REGISTRY` | `@tool` 装饰器 | `experiments/langgraph/graph/tools.py` | ✅ |
| `rag.py` | `FAISS` + `HuggingFaceEmbeddings` | `experiments/langgraph/graph/rag.py` | ✅ |
| `memory.py` | 语义记忆（BGE + LRU + 语义去重更新） | `experiments/langgraph/graph/memory.py` | ✅ |
| `prompts.py` + `cot.py` | 角色模板 + CoT 注入（字符串拼接） | `experiments/langgraph/graph/prompts.py` | ✅ |
| `react_loop.py` | `StateGraph` 节点 + 条件边 | `experiments/langgraph/graph/agent.py` | ✅ |
| `orchestrator.py` | `StateGraph` 编排（supervisor → worker → join） | `experiments/langgraph/graph/orchestrator.py` | ✅ |
| `context.py` | `context_manage_node`（token 估算 + 截断） | `experiments/langgraph/graph/agent.py` | ✅ |
| `harness/recorder.py` | 轨迹 JSON 持久化 | `experiments/langgraph/graph/main.py`（基础）+ `experiments/langgraph/graph/harness/recorder.py`（完整） | ✅ |
| `harness/sandbox.py` | subprocess 沙箱隔离 + 白名单 | `experiments/langgraph/graph/harness/sandbox.py` | ✅ |
| `harness/replay.py` | 轨迹回放 CLI | `experiments/langgraph/graph/harness/replay.py` | ✅ |
| `mcp_client.py` | 独立 MCP 客户端（JSON-RPC over stdio） | `experiments/langgraph/graph/mcp.py` | ✅ |
| `main() / __main__.py` | CLI 入口（交互 + Harness 记录） | `experiments/langgraph/graph/main.py` | ✅ |

**运行方式：**
```bash
cd experiments/langgraph/graph
python main.py "现在几点了？"          # 单次查询（含 Harness 轨迹记录）
python main.py                        # 交互模式
# 多 Agent 编排（experiments/langgraph/graph/orchestrator.py）
python orchestrator.py "帮我查时间和计算 50*30"
```

### LangGraph 架构（Harness 集成）

`experiments/langgraph/graph/agent.py` 使用 `StateGraph` 定义 Agent 循环，通过 `config["configurable"]["harness"]` 注入 Harness 实例，节点函数自动记录轨迹：

```
                    ┌──────────┐
                    │ call_model│  ← 调 ChatOpenAI（绑定工具）
                    └────┬─────┘
                         │
                    ┌────▼─────┐
                    │ 有 tool_  │  ← 条件边
                    │  calls?   │
                    └────┬─────┘
                ┌────────┴────────┐
                ▼                 ▼
          ┌──────────┐     ┌──────────────┐
          │   tools   │     │context_manage│  ← 检查消息长度
          │ (执行工具) │     │ (上下文管理)  │     超限时截断
          └────┬─────┘     └──────┬───────┘
               │                  │
               └──→ call_model    │
                              │
                          ┌───▼────┐
                          │extract │  ← 用 LLM 提取事实
                          │_memory │     存入记忆库
                          └───┬────┘
                              │
                          ┌───▼────┐
                          │   END   │
                          └────────┘
```

- **`call_model`** — 调 `ChatOpenAI`（绑定工具），将回复追加到 messages。自动记录 thought 到 Harness（含 token 数）
- **`tools`** — 执行 `@tool` 函数，支持搜索次数限制和沙箱隔离。每个 tool_call 自动记录 action/observation/duration 到 Harness
- **`context_manage`** — 每轮结束时检查消息 token 数，超限则截断最早的非 system 消息
- **`extract_memory`** — 用 LLM 从对话中提取事实，通过 `add_or_update` 语义去重后存入记忆
- **`should_continue`** — 条件边：有 tool_calls → tools，否则 → context_manage → extract_memory → END

### MCP 连接

启动时通过 `experiments/langgraph/graph/mcp.py` 自动连接 `mcp-server-time` 等 MCP 服务器，工具列表合并到 LLM 的工具定义中。同名工具自动跳过（本地优先）。

### 工具列表

| 工具 | 来源 | 功能 |
|------|------|------|
| `get_current_time` | 本地 `@tool` | 系统时间 |
| `calculator` | 本地 `@tool` | 数学计算（AST 安全解析） |
| `web_search` | AnySearch API | 搜索互联网新闻和网页 |
| `web_rag` | AnySearch + 网页抓取 | 搜索并返回网页全文内容 |
| `rag_query` | 本地 FAISS 索引 | 本地项目文档语义检索 |
| `convert_time` | MCP (`mcp-server-time`) | 时区转换 |

### 轨迹记录与 Harness

每次对话后自动保存轨迹 JSON 到 `trajectories/` 目录（保留最近 100 条）。

`experiments/langgraph/graph/harness/` 通过 LangGraph 的 `config["configurable"]["harness"]` 透传：
```python
# agent.py — call_model 节点自动记录 thought
harness.record_thought(step, thought=llm_response.content, tokens=...)

# agent.py — tools_node 节点自动记录 action
harness.record_action(step, tool_name, tool_args, observation, duration)
```

轨迹文件同时兼容手写版和 LangGraph 版，Dashboard 统一回放。

**运行评测：**
```bash
# 终端运行全部 26 条测试用例
python -m eval

# 只测本地工具（30秒），适合快速验证
python -m eval --tag tool_local

# 查看历史报告
python -m eval --list
```

### LangChain 在架构中的角色

LangChain 和 LangGraph 的分工：

```
┌─────────────────────────────────────────────────────────┐
│                    LangGraph                             │
│    ┌──────────────────────────────────────────────────┐  │
│    │  StateGraph                                       │  │
│    │  ┌──────────┐    ┌──────────┐    ┌─────────────┐ │  │
│    │  │call_model│ →→ │  tools   │ →→ │extract_mem. │ │  │
│    │  │          │ ←← │          │    │             │ │  │
│    │  └─────┬────┘    └──────────┘    └──────┬──────┘ │  │
│    │        │   ┌──────────────────────────┐ │        │  │
│    │        └──→│  should_continue(条件边) │─┘        │  │
│    │            └──────────────────────────┘          │  │
│    └──────────────────────────────────────────────────┘  │
│                          │                                │
│           ┌──────────────┴──────────────┐                 │
│           ▼                              ▼                │
│  ┌────────────────┐           ┌──────────────────┐       │
│  │   LangChain    │           │   LangChain      │       │
│  │  ChatOpenAI    │           │  @tool 函数       │       │
│  │  (模型调用)    │           │  (工具执行)       │       │
│  └────────────────┘           └──────────────────┘       │
│                                                           │
│  LangChain 消息层: AIMessage / ToolMessage / SystemMessage│
│  LangChain 检索层: FAISS + HuggingFaceEmbeddings          │
└─────────────────────────────────────────────────────────┘
```

具体组件对应：

| 层面 | 使用的 LangChain 组件 | 所在文件 | 作用 |
|------|---------------------|---------|------|
| **模型层** | `ChatOpenAI` | `experiments/langgraph/graph/llm.py` | 统一调用 DeepSeek / OpenAI / Ollama 等 API |
| **工具层** | `@tool` 装饰器 | `experiments/langgraph/graph/tools.py` | 把普通函数标记为工具，自动生成 tool schema |
| **消息层** | `AIMessage` / `ToolMessage` / `SystemMessage` / `HumanMessage` | `experiments/langgraph/graph/agent.py` | 标准消息类型，供 `MessagesState` 自动管理 |
| **检索层** | `FAISS` + `HuggingFaceEmbeddings` | `experiments/langgraph/graph/rag.py` | 文档向量化存储与语义检索 |

LangGraph 在架构中的角色：

| 层面 | 使用的 LangGraph 组件 | 所在文件 | 作用 |
|------|---------------------|---------|------|
| **循环控制** | `StateGraph` + 节点 + 条件边 | `experiments/langgraph/graph/agent.py` | 定义 Agent 执行流程（call_model → tools → extract_memory） |
| **状态管理** | `AgentState(TypedDict)` + `operator.add` reducer | `experiments/langgraph/graph/agent.py` | 自动追加对话历史，不必手动 append |
| **轨迹记录** | `MemorySaver` checkpointer | `experiments/langgraph/graph/agent.py` | 持久化对话状态，支持多轮会话 |
| **可观测性** | Harness（`config["configurable"]["harness"]` 注入） | `experiments/langgraph/graph/harness/` | call_model 和 tools_node 自动记录 thought/action/observation |

### LangGraph 多 Agent 编排

`experiments/langgraph/graph/orchestrator.py`：

```
         supervisor              worker              join
    ┌──────────────┐      ┌──────────────┐      ┌──────────┐
    │  用 LLM 分解   │      │  子任务1     │      │ 合并结果  │
    │  用户请求为    │ ───→ │  子任务2     │ ───→ │ 输出最终  │
    │  子任务列表    │      │  ...        │      │ 汇总     │
    └──────────────┘      └──────────────┘      └──────────┘
                              每个 Worker 内部
                              是独立 build_agent()

- **`supervisor`** — 用 LLM 将用户请求分解为子任务列表，分析依赖关系（depends_on）
- **`worker`** — 按依赖层级执行，同层无依赖的任务并行（ThreadPoolExecutor），有依赖的任务等待前置任务完成后注入上下文再执行
- **`join`** — 合并所有 Worker 的结果为最终汇总

**Worker 隔离：** 每个 Worker 根据子任务描述（如"计算"、"搜索"）通过 `filter_tools()` 自动筛选只暴露相关工具。例如"计算 2^100"的 Worker 只有 `calculator`，"搜索新闻"的 Worker 只有 `web_search`，互不干扰。

### 语义记忆去重更新（2026-07-06 新增）

手写版 `memory.py` 和 `experiments/langgraph/graph/memory.py` 同时新增了 `add_or_update()` 方法：

| 相似度 | 判断 | 行为 |
|--------|------|------|
| >= 0.85 | 同一事实 | 跳过（已存在） |
| 0.60 ~ 0.85 | 主体相似但内容不同 | 用新内容替换旧条目 |
| < 0.60 | 不同事实 | 作为新条目追加 |

解决用户先说"我叫张三"、后说"我的名字是李四"时的记忆冲突问题。

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
- [x] Agent 轨迹查看器 + 交互面板（dashboard/）
- [x] 沙箱子进程预热缓存
- [x] start_dashboard 工具（Agent 可主动启动 Dashboard）
- [x] Dashboard 关闭按钮 + 自动清理旧进程
- [x] Agent 清理轨迹工具（clear_trajectories）
- [x] Dashboard 轨迹清理弹窗
- [x] 多 Provider LLM 支持（llm.py + llm_config.json）
- [x] 工具集中管理（tools/ 独立目录统一注册）
- [x] 测试覆盖率提升（12个模块，87+ 项测试，无需 API Key）
- [x] Agent Eval 评测管线（eval/ 包，26 条用例覆盖 12 个模块）
- [x] experiments/langgraph/graph/harness/ LangGraph 版轨迹记录 + 沙箱 + 回放
- [x] Eval + Dashboard 打通（从评测失败案例直接跳转到轨迹回放）
- [x] 沙箱三策略（off/auto/on）— Agent 自动按工具风险决定隔离
- [x] MCP/HTTP 未知工具默认 safe，不走沙箱
- [x] agent --setup 配置向导（交互式 API Key 配置）
- [x] RAG + Memory 模型懒加载（--help 从 ~15s 缩短到 ~1.6s）
- [x] LangGraph 版同步优化（懒加载 + 路径修复）
- [x] .env.example 环境变量模板
- [x] monorepo 结构重构（src/ + experiments/）
- [x] execute_python 工具（子进程 Python 代码执行 + 超时保护）
- [x] Orchestrator 共享数据传递（前置 Worker 工具输出自动注入后置 Worker）

## 快速体验

```bash
pip install -e .
agent --setup        # 30 秒配好 API Key
agent "现在几点？"    # 直接使用
```

## License

MIT
