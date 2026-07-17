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
| [execution_agent_snapshot_20260715_v2.md](./execution_agent_snapshot_20260715_v2.md) | agent 扩容 **24** 条（易/中/难各 8） | **24/24（100%）** | 含双工具/禁工具/算法；[归档](./snapshots/execution_agent_snapshot_20260715_v2.json) |
| [execution_agent_snapshot_20260716_v3.md](./execution_agent_snapshot_20260716_v3.md) | agent 再扩至 **36** 条（易8/中12/难16） | **36/36（100%）** | [归档](./snapshots/execution_agent_snapshot_20260716_v3.json) |
| [reliability_snapshot_20260715.md](./reliability_snapshot_20260715.md) | ToolGuard/自修注入对照 4 场景 | **4/4（100%）** | [snapshots/…](./snapshots/reliability_snapshot_20260715.json) |
| [reliability_live_live_20260716.md](./reliability_live_live_20260716.md) | live Guard ON/OFF × 8 场景 | flaky 皆 6/6；**error_obs 0 vs 3** | [归档](./snapshots/reliability_live_live_20260716.json) |
| [reliability_live_live_20260716_v2.md](./reliability_live_live_20260716_v2.md) | live 扩容 **20 flaky + 4 baseline** | flaky 20/20 vs 20/20；**error_obs 0 vs 3.1**；calls **1.0 vs 2.25** | [归档](./snapshots/reliability_live_live_20260716_v2.json) |
| [P0_EVIDENCE_MAP.md](./P0_EVIDENCE_MAP.md) | 四层证据串联 | — | Execution × Reliability × Failure × Judge |
| [daily_smoke/VARIANCE.md](./daily_smoke/VARIANCE.md) | 跨日 smoke | 自动追加 | Actions `daily-smoke`（UTC 01:00） |
| [FAILURE_FLYWHEEL.md](./FAILURE_FLYWHEEL.md) | 失败→动作→复测飞轮 | 真闭环已勾选 | 配合 tdebug 扫描 |
| [flywheel_closed_loop_20260716.md](./flywheel_closed_loop_20260716.md) | 同批 100 条改前/改后 | **llm_offtrack 6→1** | [snapshots/…](./snapshots/flywheel_closed_loop_20260716.json) |
| [public_benchmark_snapshot_offline.md](./public_benchmark_snapshot_offline.md) | GSM8K×10 + HotpotQA×10 | offline 匹配器 20/20 | [归档](./snapshots/public_benchmark_snapshot_offline.json) |
| [public_benchmark_snapshot_agent_20260717.md](./public_benchmark_snapshot_agent_20260717.md) | 同上 · DeepSeek agent | **19/20（95%）** · GSM8K 10/10 · Hotpot 9/10 · Wilson [76.4, 99.1] | [归档](./snapshots/public_benchmark_snapshot_agent_20260717.json) |

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
# 可按难度过滤：--difficulty easy,medium,hard
```

说明：`offline_tools`（现 12 条）≠ `agent`（现 **36** 条，easy8/medium12/hard16）；勿混谈为同一指标。

## 公开 Agent benchmark 子集

冻结 **GSM8K test×10 + HotpotQA validation×10**（非全量榜）：

```bash
# CI / 无 Key：匹配器自检
python examples/run_public_benchmark.py
# 真实 Agent（需 API Key）
set REACT_AGENT_DISABLE_MCP=1
python examples/run_public_benchmark.py --modes agent --publish
```

数据集：`src/react_agent/eval/public_benchmark_subset.json`。agent 数字绑定模型与日期；offline 只证明打分链路。

## Harness 可靠性对照

```bash
python examples/run_reliability_harness.py --publish
python examples/run_reliability_live.py --mock
set REACT_AGENT_DISABLE_MCP=1
python examples/run_reliability_live.py --live --publish
python examples/run_failure_flywheel.py --fixture --publish
python examples/run_flywheel_closed_loop.py --publish
```

证据总图见 [P0_EVIDENCE_MAP.md](./P0_EVIDENCE_MAP.md)。

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
