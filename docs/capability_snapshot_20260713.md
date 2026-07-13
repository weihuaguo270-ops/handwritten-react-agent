# Capability/Eval 公开快照（capability_snapshot_20260713）

- **report_id:** `eval_20260713_140216`
- **timestamp:** 2026-07-13T14:02:16
- **provider:** `default`
- **dataset:** `capability_dataset.json`
- **结果:** **18/18（100.0%）**
- **avg_duration:** 29.12s · **avg_steps:** 1.6

**能力摘要:** accuracy=100%, tool_f1=1.00, reasoning=100%, consistency=100%, hallucination_rate=0%

## 按 capability

| 维度 | 通过 | 通过率 | 附注 |
|------|:----:|:------:|------|
| accuracy | 4/4 | 100% |  |
| consistency | 3/3 | 100% | cons=1.00 |
| hallucination | 4/4 | 100% | hallu=0.00 |
| reasoning | 3/3 | 100% |  |
| tool_selection | 4/4 | 100% | F1=1.00 |

## 逐条结果

| case_id | capability/tag | 结果 | 耗时(s) |
|---------|----------------|:----:|--------:|
| acc_france_capital | accuracy | PASS | 5.65 |
| acc_lu_xun_year | accuracy | PASS | 11.27 |
| acc_calc_product | accuracy | PASS | 7.78 |
| acc_water_boiling | accuracy | PASS | 6.18 |
| tool_calc_required | tool_selection | PASS | 7.07 |
| tool_time_required | tool_selection | PASS | 11.71 |
| tool_search_weather | tool_selection | PASS | 36.83 |
| tool_time_then_calc | tool_selection | PASS | 32.91 |
| reason_lily_pad | reasoning | PASS | 21.98 |
| reason_ages | reasoning | PASS | 21.97 |
| reason_chain_calc | reasoning | PASS | 26.52 |
| cons_lu_xun | consistency | PASS | 71.39 |
| cons_capital | consistency | PASS | 72.59 |
| cons_math | consistency | PASS | 73.43 |
| hallu_wrong_year_forbid | hallucination | PASS | 24.03 |
| hallu_wrong_capital_forbid | hallucination | PASS | 27.69 |
| hallu_calc_grounded | hallucination | PASS | 22.74 |
| hallu_no_fake_nobel | hallucination | PASS | 42.35 |

## 如何复现

```bash
python -m react_agent.eval --dataset capability
python examples/publish_eval_snapshot.py --from-report <json路径>
```

> 学习用途快照：样本量有限，不代表生产基准。

## 备注

- git: `15cc5f5`
- source_report: `src/react_agent/eval/reports/eval_20260713_140216.json`
- archived_json: `docs/snapshots/capability_snapshot_20260713.json`
- 评分器：capability 用规则打分；功能集用 must_contain/工具检查
