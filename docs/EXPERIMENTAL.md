# Experimental modules

这些模块可跑、可学，**不是**核心 ReAct 学习运行时的一部分。默认工具表不注册实验工具；需要时：

```bash
set REACT_AGENT_EXPERIMENTAL_TOOLS=1
```

| Module | Entry | Notes |
|--------|-------|-------|
| RAG | `rag.py`，`pip install -e ".[rag]"` | 多数评测 `SKIP_RAG=1` |
| MCP | `mcp_client.py` / `--mcp` | 评测默认 `DISABLE_MCP=1` |
| Multi-agent | `orchestrator.py` / `planner.py` | 演示编排，非产品 |
| ToT | `tot.py` | 教学推理工具 |
| Dashboard | `dashboard/` | 本地可视化 |
| LangGraph twin | `experiments/langgraph/` | 对照实现；**无严格等价性测试** |

证据与快照（execution / reliability / κ）见 [`EVAL_INDEX.md`](EVAL_INDEX.md) 与 [`P0_EVIDENCE_MAP.md`](P0_EVIDENCE_MAP.md)，不进本页。
