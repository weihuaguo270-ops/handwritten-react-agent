# Harness Trajectory Schema

Shared **Format B** JSON across:

| Repo | Role |
|------|------|
| [react-agent](https://github.com/weihuaguo270-ops/react-agent) | Produce (`harness.recorder`) + validate (`harness.schema`) |
| [trace-debugger](https://github.com/weihuaguo270-ops/trace-debugger) | Analyze failures |
| [llm-eval-engine](https://github.com/weihuaguo270-ops/llm-eval-engine) | Process reward / DAG score |

## Rules (interop)

1. `step` is **1-based** (never emit `0` from new code).
2. Prefer `action.arguments` as a **JSON string**; `args` object is accepted.
3. Prefer singular `action`; use `actions[]` only for multi-tool steps.
4. Required top-level: `session_id`, `query`, `steps`, `final_answer`.
5. Optional top-level `schema_version` (current major: **`1`**). Absent ⇒ treated as major `1`. Incompatible major fails validation.

Wire constant: `react_agent.harness.schema.SCHEMA_VERSION`  
Eval API constant: `react_agent.eval.scorer.EVAL_API_VERSION` (`0.1`, mirrored in llm-eval-engine).

File: [`harness_trajectory.schema.json`](harness_trajectory.schema.json)

Demo: `python examples/harness_closed_loop.py`

契约测试（防 API 漂移）：

- `pytest tests/test_harness_schema.py` — Format B + `schema_version`
- `pytest tests/test_eval_engine_contract.py` — `ProcessRewardScorer.extra_contracts@{EVAL_API_VERSION}`
- `pytest tests/test_tdebug_eval_contract.py` — fixture → schema → tdebug → eval（需 sibling 安装）
