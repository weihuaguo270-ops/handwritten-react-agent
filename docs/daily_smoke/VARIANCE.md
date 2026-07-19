# Daily smoke variance（跨日）

自动由 `examples/run_daily_smoke.py` + GitHub Actions `daily-smoke` 追加。
默认 **offline / mock**（不耗 API）；带 Key 时可选 `--with-agent`。

| date (UTC) | git | exec offline | exec ok | reliability harness | reliability mock | agent smoke | overall |
|------------|-----|-------------:|:-------:|:-------------------:|:----------------:|:-----------:|:-------:|
| 2026-07-17 | `ff06d08` | 12/12 | PASS | PASS | PASS | skip | PASS |
| 2026-07-17 | `556c9da` | 12/12 | PASS | PASS | PASS | skip | PASS |
| 2026-07-18 | `71fd005` | 12/12 | PASS | PASS | PASS | skip | PASS |
| 2026-07-19 | `b5ee1b0` | 12/12 | PASS | PASS | PASS | skip | PASS |

## 怎么读

- 看的是**跨日是否稳定**，不是再刷一次公开大快照。
- `agent smoke` 默认 skip；只有 workflow / 本地显式开 `--with-agent` 才跑。
- 复现：`python examples/run_daily_smoke.py`
