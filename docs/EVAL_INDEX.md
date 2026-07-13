# 公开评测报告索引

本目录存放 **可复现** 的 Agent 评测快照（学习用途，样本量有限）。

## 报告一览

| 报告 | 数据集 | 结果 | 归档 JSON |
|------|--------|------|-----------|
| [capability_snapshot_20260713.md](./capability_snapshot_20260713.md) | capability（当时 18 条） | 18/18（100%） | [snapshots/…](./snapshots/capability_snapshot_20260713.json) |
| [eval_report_20260713.md](./eval_report_20260713.md) | default 功能集 26 条 | 23/26（88%） | 人工整理（见文内失败分析） |
| [capability_newcases_20260713.md](./capability_newcases_20260713.md) | capability 扩容 6 条 | **5/6（83%）** | [snapshots/…](./snapshots/capability_newcases_20260713.json) |

当前 `capability_dataset.json` 已扩至 **24** 条（原 18 + 新 6）。全量重跑：

```bash
python examples/publish_eval_snapshot.py --run capability --stem capability_snapshot_YYYYMMDD
```

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
