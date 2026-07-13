# Capability/Eval 公开快照（capability_newcases_20260713）

- **report_id:** `eval_20260713_181154`
- **timestamp:** 2026-07-13T18:11:54
- **provider:** `deepseek`
- **dataset:** `capability_dataset.json`
- **结果:** **5/6（83.3%）**
- **avg_duration:** 28.64s · **avg_steps:** 1.0

**能力摘要:** accuracy=100%, tool_f1=0.00, reasoning=100%, consistency=100%, hallucination_rate=0%

## 按 capability

| 维度 | 通过 | 通过率 | 附注 |
|------|:----:|:------:|------|
| accuracy | 2/2 | 100% |  |
| consistency | 1/1 | 100% | cons=1.00 |
| hallucination | 1/1 | 100% | hallu=0.00 |
| reasoning | 1/1 | 100% |  |
| tool_selection | 0/1 | 0% | F1=0.00 |

## 逐条结果

| case_id | capability/tag | 结果 | 耗时(s) |
|---------|----------------|:----:|--------:|
| acc_germany_capital | accuracy | PASS | 31.74 |
| acc_square_15 | accuracy | PASS | 20.61 |
| tool_calc_99_plus_1 | tool_selection | FAIL | 24.24 |
| reason_three_times | reasoning | PASS | 25.02 |
| cons_boiling | consistency | PASS | 47.05 |
| hallu_wrong_boiling_forbid | hallucination | PASS | 23.18 |

## 失败用例

- `tool_calc_99_plus_1`: actual=[], groups=[['calculator']], P=0.00 R=0.00 F1=0.00, sequence_ok=False

## 如何复现

```bash
python -m react_agent.eval --dataset capability
python examples/publish_eval_snapshot.py --from-report <json路径>
```

> 学习用途快照：样本量有限，不代表生产基准。

## 备注

- git: `15cc5f5`
- source_report: `D:/agent_learning/react-agent/src/react_agent/eval/reports/eval_20260713_181154.json`
- archived_json: `docs/snapshots/capability_newcases_20260713.json`
- 评分器：capability 用规则打分；功能集用 must_contain/工具检查
