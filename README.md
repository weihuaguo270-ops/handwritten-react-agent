# handwritten-react-agent

> 从零手写 LLM Agent Runtime — 不依赖 LangChain、AutoGPT 等框架，只用 Python 标准库 + LLM API + BGE Embedding。
> 覆盖工具调用、记忆、RAG、多 Agent 编排、推理增强、评测 Harness 与安全沙箱全链路。

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
  ├── step 0: Harness Recorder 开始记录（harness/recorder.py）
  │     触发时机: 每次 react_loop() 被调用时必定执行。
  │     记录内容: session_id / query / model / system_prompt
  │

  └── ReAct Loop:
        for step in range(1, max_steps + 1):
          │
          ├── (a) call_llm(messages, tool_defs)
          │      → LLM 返回: thought + 可选的 tool_calls
          │      → Harness Recorder: add_thought(step, LLM_回复)
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
          │           │       ├── SANDBOX.enabled? → 子进程隔离执行
          │           │       └── 非启用 → 直接 Python 调用函数
          │           │
          │           │    2. 不在 TOOL_REGISTRY → 遍历 MCP_CLIENTS
          │           │       ├── 工具名在 MCP 工具列表?
          │           │       │   → mcp_client.call_tool(name, args)
          │           │       │     MCP 通过 JSON-RPC over stdio 通信
          │           │       └── 都不在 → 返回"未知工具"错误
          │           │
          │           │    触发 Harness 记录:
          │           │      Recorder: add_tool_call(step, name, args, result)
          │           │     （每次工具调用后都记录）
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
| Context | ReAct Loop 每步结束后 | messages.append 之后 | 检查 token → 可选压缩/截断 |
| Harness | react_loop 进入/退出/每步工具调用 | start_trajectory / add_thought / add_tool_call / finish_trajectory | 持久化到 trajectories/*.json |
| Dashboard | 用户主动启动（命令行或 Agent 调用 start_dashboard 工具） | 独立 Flask Web 服务（端口 5050） | 读取 trajectories/ 目录 → 浏览器展示轨迹回放与实时对话 |
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

### Harness 层（harness/）

Harness = Recorder（轨迹记录）+ Sandbox（沙箱隔离）+ Replay（回放调试），对应 Agent = LLM + Harness 中的保障层。

- **Recorder（harness/recorder.py）：** 每步 thought/action/observation/token_usage 持久化为 JSON
- **Sandbox（harness/sandbox.py）：** subprocess + timeout 隔离不可信代码，AST 白名单安全解析；支持启动时预热缓存
- **Replay：** `python -m harness.replay --latest` 从轨迹文件逐步回放

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
| `toggle_sandbox(enabled)` | 开启/关闭沙箱隔离 | harness/sandbox.py |
| `start_dashboard(port)` | 启动 Dashboard Web 界面 | react_loop.py |
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
├── react_loop.py       # 主循环（ReAct + 工具 + 记忆 + 交互）
├── cot.py              # 思维链（4 种策略自动切换）
├── tot.py              # 思维树（BFS/DFS 双搜索模式）
├── planner.py          # 任务规划器（模板+LLM 分层分解）
├── orchestrator.py     # 多 Agent DAG 调度
├── prompts.py          # 角色注入（5 种风络）
├── context.py          # 上下文窗口管理（4 种策略）
├── harness/            # Harness 层（沙箱 + 记录 + 重放）
│   ├── __init__.py     # 统一 Harness 入口
│   ├── recorder.py     # 轨迹记录（原来的 harness.py）
│   ├── sandbox.py      # 子进程隔离（原来的 sandbox.py）
│   └── replay.py       # 离线回放（原来的 replay.py）
├── mcp_client.py       # MCP 协议客户端
├── rag.py              # RAG 检索增强生成
├── memory.py           # 语义记忆系统
├── eval.py             # 端到端评测（12 个测试用例）
├── test_all.py         # 单元测试（46 项，无需 API Key）
├── trajectories/       # 轨迹文件
├── dashboard/          # Agent 交互 + 轨迹回放 Web 界面
│   ├── server.py       # Flask API 服务（端口 5050，含聊天/轨迹/清理/关闭接口）
│   ├── index.html      # 前端页面（左侧聊天面板 + 右侧轨迹回放 + 清理/关闭按钮）
│   └── kill_old.py     # 启动前清除旧进程
├── notes/              # 开发笔记（bug 记录/架构/Dashboard 心路历程）
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
- [x] Agent 轨迹查看器 + 交互面板（dashboard/）
- [x] 沙箱子进程预热缓存
- [x] start_dashboard 工具（Agent 可主动启动 Dashboard）
- [x] Dashboard 关闭按钮 + 自动清理旧进程
- [x] Agent 清理轨迹工具（clear_trajectories）
- [x] Dashboard 轨迹清理弹窗
- [ ] Dashboard 会话详情对比模式

## License

MIT
