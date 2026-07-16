# P0 证据地图（Execution × Reliability × Failure × Judge）

一张页把实习向 P0 证据串起来，避免「各仓各说各话」。

| 层 | 问题 | 证据 | 数字（截至 2026-07-15/16） |
|----|------|------|---------------------------|
| **能不能干成** | 工具 / Agent 任务成功率 | [execution offline](./execution_snapshot_20260715.md) · [agent v3](./execution_agent_snapshot_20260716_v3.md) | offline **12/12**；agent **36/36**（易8/中12/难16） |
| **坏了能不能撑住** | Guard/自修是否有效 | [注入对照](./reliability_snapshot_20260715.md) · [live v2](./reliability_live_live_20260716_v2.md) | 注入 **4/4**；live flaky **n=20**：ON/OFF 皆 100%，**error_obs 0 vs 3.1**，**tool_calls 1.0 vs 2.25** |
| **坏在哪** | 轨迹失败分布 | [tdebug 真实 100 条](https://github.com/weihuaguo270-ops/trace-debugger/blob/master/docs/tdebug_failure_real_20260715.md) · [飞轮闭环](./flywheel_closed_loop_20260716.md) | 同批重扫：`llm_offtrack` **6→1**；duplicate Harness 已拦 |
| **评得清不清** | Judge 与人标一致吗 | [κ offline v3](https://github.com/weihuaguo270-ops/llm-eval-engine/blob/master/docs/calibration_snapshot_20260716_offline.md) · [κ live](https://github.com/weihuaguo270-ops/llm-eval-engine/blob/master/docs/calibration_snapshot_20260716_live.md) · [怎么读](https://github.com/weihuaguo270-ops/llm-eval-engine/blob/master/docs/METRICS_TRUST.md) | **held_out** offline κ=**1.0**（n=11，冻结分）；全量 offline κ≈**0.90** CI[0.75,1.0]；live 全量 κ≈**0.68** |

## 怎么读（简历叙事）

1. **Agent 开发岗**：先甩 execution **36/36** + live reliability ON/OFF 表 → 再链 tdebug 失败分布与飞轮闭环。  
2. **评测岗**：先甩 κ 校准 + tdebug 失败 taxonomy → 再说明 execution 是「任务通过率」另一轨。  
3. **不要合并成一个数字**：offline ≠ agent ≠ Judge 分。

## 一键复跑

```bash
# Execution
python examples/run_execution_suite.py
set REACT_AGENT_DISABLE_MCP=1
python examples/run_execution_suite.py --modes agent --publish

# Reliability
python examples/run_reliability_harness.py --publish
python examples/run_reliability_live.py --mock
python examples/run_reliability_live.py --live --publish

# Failure flywheel（观察→修复→同批对照）
python examples/run_failure_flywheel.py --fixture --publish
python examples/run_flywheel_closed_loop.py --publish
```

## 诚实边界

- 样本量仍属学习级；live 绑定模型与日期  
- flaky live 使用注入超时，证明机制有效，不等于生产故障率  
- κ：**分栏引用** — held_out（协议冻结后）优先于全量 offline；live 与 offline 勿混谈；单人标注、第二标注者 pending  
- execution 报告含 Wilson 95% CI 与 tool/final 分项率，勿只甩「36/36」  
- 飞轮 `llm_offtrack` 下降含假阳性修复；`duplicate` 历史 traj 不变，需新跑才体现 Harness 拦截
