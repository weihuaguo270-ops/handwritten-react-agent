# handwritten-react-agent

手写 ReAct Agent — 零框架实现 LLM Agent 核心循环。

## 概述

本项目从零实现 ReAct（Reasoning + Acting）循环，不依赖 LangChain、AutoGPT 等任何 Agent 框架，只用 Python 标准库 + LLM API。

通过手写完整循环，深入理解 Agent 的核心机制：Thought → Action → Observation 的反馈闭环。

## 核心机制

```
用户输入 → 调用 LLM → LLM 输出思考(Thought)
                      → 需要工具？→ 调用工具(Action)
                                    → 工具返回结果(Observation)
                                    → 继续调用 LLM 分析结果
                      → 已足够信息？→ 输出最终答案(Final Answer)
```

### ReAct vs CoT

| | Chain-of-Thought | ReAct |
|---|---|---|
| 信息来源 | 仅限训练数据 | 训练数据 + 外部工具 |
| 能否获取新信息 | ❌ | ✅ |
| 反馈闭环 | ❌ 单向推理 | ✅ 工具结果影响下一步 |
| 适用场景 | 逻辑推理 | 需要操作外部的 Agent |

## 已实现工具

| 工具名 | 功能 | 说明 |
|--------|------|------|
| `get_time()` | 获取当前时间 | 本地调用，无依赖 |
| `calculator(expression)` | 计算数学表达式 | Python eval，安全校验 |
| `web_search(query)` | 搜索维基百科 | 中文优先，英文兜底 |

## 快速开始

### 1. 配置 API Key

修改 `react_loop.py` 顶部：

```python
API_KEY=*** = "https://api.deepseek.com"  # 或其他兼容 API
MODEL = "deepseek-v4-flash"
```

### 2. 运行

```bash
python react_loop.py
```

### 3. 测试用例

当前内置 4 个测试：

1. `现在几点了？` → 调 get_time
2. `计算 (23 + 45) * 2` → 调 calculator
3. `先告诉我时间，再计算 100 / 7` → 连续调两个工具
4. `搜索一下2026年AI Agent的最新发展` → 多次搜索后汇总报告

## 扩展新工具

添加一个新工具只需三步：

1. **写函数**：实现工具逻辑
2. **注册**：加入 `TOOL_REGISTRY`
3. **声明**：在 `TOOL_DEFINITIONS` 中添加 JSON 描述

## 项目结构

```
handwritten-react-agent/
├── react_loop.py          # 主代码（ReAct Loop + 工具实现）
├── README.md
└── LICENSE
```

## 后续计划

- [ ] 交互式对话模式
- [ ] 对话历史记忆
- [ ] Agent Eval 评测框架
- [ ] 更多工具

## License

MIT
