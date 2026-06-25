# handwritten-react-agent

手写 ReAct Agent — 零框架实现 LLM Agent 核心循环，支持交互模式、工具调用和语义记忆。

## 概述

从零实现 ReAct（Reasoning + Acting）循环，不依赖 LangChain、AutoGPT 等框架，只用 Python 标准库 + LLM API + BGE Embedding。

## 快速开始

### 1. 创建虚拟环境（推荐）

```bash
python -m venv venv
venv\Scripts\python -m pip install numpy scikit-learn sentence-transformers
```

### 2. 配置 API Key

修改 `react_loop.py` 顶部第 17 行：

```python
API_KEY=''your-key-here''
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-v4-flash"
```

### 3. 运行

```bash
# 交互模式（自动加载 MCP 时间服务器 + 文件系统服务器）
venv\Scripts\python react_loop.py

# 单次查询
venv\Scripts\python react_loop.py "现在纽约几点？"

# 指定 MCP 服务器（覆盖默认）
venv\Scripts\python react_loop.py --mcp "uvx mcp-server-time" "现在几点？"
```

交互模式：

```
==================================================
  Agent 交互模式已启动
  ==================================================
  可用工具：get_time / calculator / web_search / fetch_page / summarize
  可问：时间、计算、搜索新闻、读网页、总结内容
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

### 记忆系统

基于 BGE-small-zh-v1.5 的语义记忆：

- 说 `记住...` 保存信息 → BGE 模型转 512 维向量 → 持久化到 memory.json
- 每次提问自动检索相关记忆（余弦相似度）→ 加入上下文
- 启动时自动加载历史记忆
- 查看记忆：输入 `'记忆'`

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
| `read_text_file(path)` | 读取文件内容 | MCP server-filesystem |
| `write_file(path, content)` | 写入文件 | MCP server-filesystem |
| `edit_file(path, edits)` | 编辑文件（行级替换） | MCP server-filesystem |
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
venv\Scripts\python eval.py
```

5 个测试用例，自动检查工具调用、内容完整性、步数。

## 项目结构

```
handwritten-react-agent/
├── react_loop.py    # 主代码（ReAct Loop + 工具 + 记忆 + 交互模式）
├── mcp_client.py    # MCP 协议模块（JSON-RPC 2.0 over stdio）
├── eval.py          # 自动化评测
├── memory.json      # 记忆持久化（自动生成）
├── README.md
└── LICENSE
```

## 依赖

- Python 3.8+
- numpy
- scikit-learn
- sentence-transformers（首次加载约 13 秒）

## 多 Agent 协作（Orchestrator-Worker）

支持将复杂请求自动拆分为多个子任务，每个子任务由独立的 ReAct Loop 执行，最后汇总结果。

### 示例

```
用户: 现在纽约几点？同时看看mcp_client.py有多大

[Orchestrator] 拆分为 2 个子任务:
  1. 查询纽约的当前时间
  2. 查看mcp_client.py的文件大小

[Worker 1/2] → get_current_time("America/New_York") → 纽约凌晨 02:47
[Worker 2/2] → get_file_info("...mcp_client.py")    → 5,199 字节
[汇总结果]    → 整合答案
```

### 触发条件

用户问题含"同时"、"并且"、"还有"、"另外"、"且"等连接词时自动启用。

### 实现

```python
def multi_agent_chain(user_query):
    # 1. Orchestrator 拆任务（LLM）
    # 2. Workers 依次执行（独立 ReAct Loop）
    # 3. Orchestrator 汇总（LLM 合并结果）
```

## 后续计划

- [ ] LLM 自动提取关键信息（无需手动说"记住"）
- [ ] 记忆遗忘机制（Token 窗口管理）
- [x] MCP 协议支持
- [x] 多 Agent 协作（Orchestrator-Worker）
- [ ] Web UI 界面

## MCP 协议支持

### 架构

```
mcp_client.py          ← 独立 MCP 协议模块（纯标准库）
react_loop.py          ← from mcp_client import MCPClient
                         --mcp 参数 / DEFAULT_MCP_SERVERS 自动加载
```

### 默认 MCP 服务器（启动时自动连接）

| 服务器 | 工具 | 启动方式 |
|--------|------|---------|
| mcp-server-time | `get_current_time`, `convert_time` | `uvx` |
| server-filesystem | 文件读写、目录管理、搜索等 14 个工具 | `npx` |

### 通信协议

JSON-RPC 2.0 over stdin/stdout（UTF-8 编码）

```
Client: initialize → notifications/initialized → tools/list → tools/call
```

## License

MIT
