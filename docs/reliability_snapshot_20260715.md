# Harness 可靠性对照（reliability_snapshot_20260715）

- **report_id:** `reliability_20260715_080432`
- **timestamp:** `2026-07-15T08:04:32.675041+00:00`
- **kind:** `harness_reliability_injected`
- **场景通过:** **4/4（100.0%）**
- **git:** `3cd5e21`

## 对照表

| scenario | Guard/自修效果 | passed |
|----------|----------------|:------:|
| `flaky_timeout_recover` | timeout 可恢复：前 2 次失败，第 3 次成功 | yes |
| `hard_timeout_block` | ToolGuard ON 时对慢工具做超时截断 | yes |
| `self_repair_attach` | 失败结果附着自修提示；正常结果不误触发 | yes |
| `rate_limit_guard` | 1 分钟内同工具超限调用被阻断 | yes |

## 明细

### `flaky_timeout_recover`

- ON: `{"recovered": true, "calls": 3, "output": "recovered_ok", "error": null}`
- OFF: `{"recovered": false, "calls": 1, "output": "", "error": "timeout"}`
- delta: `{"on_better": true, "on_calls": 3, "off_calls": 1}`

### `hard_timeout_block`

- ON: `{"blocked_or_timed_out": true, "elapsed_s": 2.5, "output": "{\"error\": \"\\u8d85\\u65f6 (1s)\", \"blocked\": false, \"retry_exhausted\": true}"}`
- OFF: `{"blocked_or_timed_out": false, "elapsed_s": 0.25, "output": "too_late", "note": "OFF 路径用短睡代理，表示无超时截断会返回结果"}`
- delta: `{"on_blocks": true, "off_returns_result": true}`

### `self_repair_attach`

- ON: `{"error_detected": true, "hint_attached": true, "clean_not_flagged": true, "passed": true}`
- OFF: `{"note": "关闭自修时仅透传原始错误，无提示层（逻辑上 attach=False）", "hint_attached": false, "passed": true}`
- delta: `{"self_repair_available": true}`

### `rate_limit_guard`

- ON: `{"first": "ok", "second": "ok", "third": "{\"error\": \"工具 web_search 调用频繁\", \"blocked\": true}", "blocked": true, "passed": true}`
- OFF: `{"note": "无 Guard 时三次均可成功（此处不实际调用）", "would_block": false}`
- delta: `{"guard_blocks_burst": true}`

## 复现

```bash
python examples/run_reliability_harness.py --publish
```

## 诚实边界

- 本报告是 **注入故障单元对照**，不是数百步线上长跑
- 与 live Agent execution 成功率是不同指标，勿合并宣传
