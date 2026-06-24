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
venv\Scripts\python react_loop.py
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
| `get_time()` | 获取当前时间 | 本地调用 |
| `calculator(expression)` | 计算数学表达式 | Python eval |
| `web_search(query)` | 搜索互联网新闻 | AnySearch 搜索引擎 |
| `fetch_page(url)` | 读取网页正文 | 维基API/HTML提取 |
| `summarize(text)` | 自动提取摘要 | 抽取式 |

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

## 后续计划

- [ ] LLM 自动提取关键信息（无需手动说"记住"）
- [ ] 记忆遗忘机制（Token 窗口管理）
- [ ] MCP 协议支持
- [ ] Web UI 界面

## License

MIT
