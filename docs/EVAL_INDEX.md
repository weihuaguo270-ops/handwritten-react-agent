# 公开评测报告索引

本目录存放 **可复现** 的 Agent 评测快照（学习用途，样本量有限）。

## 报告一览

| 报告 | 数据集 | 结果 | 归档 JSON |
|------|--------|------|-----------|
| [capability_snapshot_20260713.md](./capability_snapshot_20260713.md) | capability（当时 18 条） | 18/18（100%） | [snapshots/…](./snapshots/capability_snapshot_20260713.json) |
| [eval_report_20260713.md](./eval_report_20260713.md) | default 功能集 26 条 | 23/26（88%） | 人工整理（见文内失败分析） |
| [capability_newcases_20260713.md](./capability_newcases_20260713.md) | capability 扩容 6 条 | **5/6（83%）** | [snapshots/…](./snapshots/capability_newcases_20260713.json) |
| [execution_snapshot_20260715.md](./execution_snapshot_20260715.md) | execution 离线工具集 8 条 | **8/8（100%）** | [snapshots/…](./snapshots/execution_snapshot_20260715.json) |
| [execution_agent_snapshot_20260715.md](./execution_agent_snapshot_20260715.md) | execution **agent** 端到端 6 条 | **6/6（100%）** | DeepSeek；`DISABLE_MCP=1`；[归档](./snapshots/execution_agent_snapshot_20260715.json) |
| [reliability_snapshot_20260715.md](./reliability_snapshot_20260715.md) | ToolGuard/自修注入对照 4 场景 | **4/4（100%）** | [snapshots/…](./snapshots/reliability_snapshot_20260715.json) |

当前 `capability_dataset.json` 已扩至 **24** 条（原 18 + 新 6）。全量重跑：

```bash
python examples/publish_eval_snapshot.py --run capability --stem capability_snapshot_YYYYMMDD
```

## Execution 成功率

```bash
# 工具层（offline，CI 默认）
python examples/run_execution_suite.py
# 端到端 Agent（需 API Key；评测默认关 MCP 以提高确定性）
set REACT_AGENT_DISABLE_MCP=1
python examples/run_execution_suite.py --modes agent --publish
```

说明：`offline_tools` ≠ `agent`；前者测工具，后者测 LLM 规划+执行。

## Harness 可靠性对照

```bash
python examples/run_reliability_harness.py --publish
```

## 失败归因周报

轨迹失败分布见姊妹仓 [trace-debugger/docs/FAILURE_INDEX.md](https://github.com/weihuaguo270-ops/trace-debugger/blob/master/docs/FAILURE_INDEX.md)（含 **真实 100 条**轨迹扫描）。

## 一键发布（推荐）

```bash
# 1) 从已有 JSON 固化 Markdown + 归档到 docs/snapshots/（无需 API）
python examples/publish_eval_snapshot.py --from-report src/react_agent/eval/reports/eval_XXXX.json

# 2) 现场跑批并发布（需 DEEPSEEK_API_KEY）
set REACT_AGENT_SKIP_RAG=1
python examples/publish_eval_snapshot.py --run capability

# 3) 只验证扩容的新用例
python examples/publish_eval_snapshot.py --run capability --only-new --stem capability_newcases_YYYYMMDD
```

## 与 llm-eval-engine 的分工

| 仓库 | 评测侧重 |
|------|----------|
| **react-agent** | 任务通过率、工具/答案规则打分、capability 五维 |
| **llm-eval-engine** | Process Reward、动态 rubric、人机校准（κ） |

## 诚实边界

- 公开数字绑定具体 `report_id` / 归档 JSON；换模型后需重跑
- 角色类功能用例曾因 `must_contain` 过严出现假阴性（见功能报告）
- 一致性用例会多次调用 LLM，费用与耗时更高
