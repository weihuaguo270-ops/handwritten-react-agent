# Execution Agent 公开快照（execution_agent_snapshot_20260715）

- git: `3cd5e21`
- archived_json: `docs/snapshots/execution_agent_snapshot_20260715.json`
- reproduce: `set REACT_AGENT_DISABLE_MCP=1 && python examples/run_execution_suite.py --modes agent --publish`
- provider: `deepseek`
- **report_id:** `execution_agent_snapshot_20260715`
- **timestamp:** `2026-07-15T08:15:20.319418+00:00`
- **dataset:** `execution_dataset.json`
- **modes:** `agent`
- **通过率:** **6/6（100.0%）**

## 按 mode

| mode | passed | total | rate |
|------|--------|-------|------|
| `agent` | 6 | 6 | 100.0% |

## 按 tag

| tag | passed | total | rate |
|-----|--------|-------|------|
| `agent` | 6 | 6 | 100.0% |
| `calculator` | 3 | 3 | 100.0% |
| `execute_python` | 2 | 2 | 100.0% |
| `execution` | 6 | 6 | 100.0% |
| `get_time` | 1 | 1 | 100.0% |

## 用例明细

- **PASS** `agent_calc_17x19` [agent] — Agent: compute 17*19 with calculator (30.94s) — agent outcome ok
- **PASS** `agent_calc_100_minus_37` [agent] — Agent: compute 100-37 (23.83s) — agent outcome ok
- **PASS** `agent_get_time` [agent] — Agent: get current time via tool (24.34s) — agent outcome ok
- **PASS** `agent_py_sum_1_to_5` [agent] — Agent: execute_python sum 1..5 (28.72s) — agent outcome ok
- **PASS** `agent_calc_8x7` [agent] — Agent: 8*7 via calculator (32.92s) — agent outcome ok
- **PASS** `agent_py_factorial_5` [agent] — Agent: factorial 5 via python (23.02s) — agent outcome ok

## 诚实边界

- 本套为 **端到端 Agent（LLM 规划 + 工具）** 执行验收，绑定模型与日期
- 样本量有限；失败需区分模型失误 / 评分过严 / 工具环境
- 同日首跑曾 **4/6**：开着默认 MCP 时 `get_time` 被替换为 `get_current_time` 导致工具名假阴性；本快照在 `REACT_AGENT_DISABLE_MCP=1` 下重跑为 **6/6**
- 不代表任意任务域的生产 SLA
