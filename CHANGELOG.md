# Changelog

## Unreleased

### Added
- Execution-based 离线任务集：`execution_dataset.json` + `execution_scorer` + `examples/run_execution_suite.py`（8/8 公开快照）
- **Agent 端到端 execution**：`--modes agent`；扩至 **36** 条（易8/中12/难16，公开 36/36）；评测默认 `REACT_AGENT_DISABLE_MCP=1`
- Harness 可靠性注入对照：`examples/run_reliability_harness.py`（4/4 公开快照）
- **Live 可靠性 ON/OFF**：扩至 **20 flaky + 4 baseline**（error_obs 0 vs 3.1；tool_calls 1.0 vs 2.25）
- 失败飞轮：`examples/run_failure_flywheel.py` + `docs/FAILURE_FLYWHEEL.md`；证据总图 `docs/P0_EVIDENCE_MAP.md`
- **飞轮真闭环**：相邻同参工具拦截（`REACT_AGENT_BLOCK_DUPLICATE_TOOLS`）+ `run_flywheel_closed_loop.py`；同批 100 条 `llm_offtrack` **6→1**
- **跨仓评分契约**：`score_with_eval_engine` 对齐 `extra_contracts`；`EvalIntegrationError` + `tests/test_eval_engine_contract.py`
- **收尾强制 FINAL ANSWER**：预留末步禁工具（`REACT_AGENT_RESERVE_FINAL_STEP`）+ max_steps 后无工具强制总结；回归 `tests/test_final_answer_guard.py`
- **Windows 控制台安全输出**：`console_io.safe_print` + `[PASS]/[FAIL]`；CI 增加 `windows-latest` × 3.10/3.11
- **指标可信度**：execution Wilson CI；Judge 口径统一为 held_out live κ≈0.59（见 eval-engine METRICS_TRUST）
- **Core 收窄**：默认工具表去掉 RAG/ToT/Dashboard；`docs/EXPERIMENTAL.md`；README 去实验/评测混杂
- **CI**：coverage / mypy / pip-audit（venv）
- Harness 长跑：默认接通 `ToolGuard`（超时/重试/熔断）与工具失败自修提示；评测透传 `max_steps`

## 0.1.0 (2026-07-13)

### Added
- Capability 评测：`capability_scorer` + `capability_dataset.json`（准确率/工具/推理/一致性/幻觉）
- `python -m react_agent` / `python -m react_agent.eval` 入口
- 真实 LLM 集成测试（无 Key 时 skip）与 Agent→Eval 对接示例

### Changed
- README 降调为学习实现；沙箱防递归；`.env` 优先加载 API Key
- 项目从 handwritten-react-agent 更名为 react-agent（历史）

### Infrastructure
- GitHub Actions CI（lint + test + eval-engine 集成校验）
