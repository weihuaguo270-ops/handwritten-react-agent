# 公开 Agent benchmark 子集（public_benchmark_snapshot_agent_20260717）

- **report_id:** `public_bench_20260717_020304`
- **bundle:** `public_agent_benchmark_subset_v1` v`1`
- **dataset:** `public_benchmark_subset.json`
- **modes:** `agent`
- **通过率:** **19/20（95.0%）**
- **Wilson 95% CI:** [76.4, 99.1]%
- **说明:** 公开子集 n=20（GSM8K×10 + HotpotQA×10）；offline 只验证匹配器；agent 数字绑定模型/日期，勿当全量榜
- **模型 / 环境:** DeepSeek · `REACT_AGENT_DISABLE_MCP=1` · `SKIP_RAG=1`
- **唯一失败:** `hotpot_val_0002` — gold=`Greenwich Village, New York City`，模型答 `New York City`（偏泛，按严格匹配计 FAIL）

## 按 benchmark

| benchmark | passed | total | rate |
|-----------|--------|-------|------|
| `gsm8k` | 10 | 10 | 100.0% |
| `hotpotqa` | 9 | 10 | 90.0% |

## 按 mode

| mode | passed | total | rate |
|------|--------|-------|------|
| `agent` | 19 | 20 | 95.0% |

## 明细

- [PASS] `gsm8k_test_0000` (gsm8k/agent): gold ok: pred=18 gold=18
- [PASS] `gsm8k_test_0001` (gsm8k/agent): gold ok: pred=3 gold=3
- [PASS] `gsm8k_test_0002` (gsm8k/agent): gold ok: pred=70000 gold=70000
- [PASS] `gsm8k_test_0003` (gsm8k/agent): gold ok: pred=540 gold=540
- [PASS] `gsm8k_test_0004` (gsm8k/agent): gold ok: pred=20 gold=20
- [PASS] `gsm8k_test_0005` (gsm8k/agent): gold ok: pred=64 gold=64
- [PASS] `gsm8k_test_0006` (gsm8k/agent): gold ok: pred=260 gold=260
- [PASS] `gsm8k_test_0007` (gsm8k/agent): gold ok: pred=160 gold=160
- [PASS] `gsm8k_test_0008` (gsm8k/agent): gold ok: pred=45 gold=45
- [PASS] `gsm8k_test_0009` (gsm8k/agent): gold ok: pred=460 gold=460
- [PASS] `hotpot_val_0000` (hotpotqa/agent): gold ok: contains
- [PASS] `hotpot_val_0001` (hotpotqa/agent): gold ok: contains
- [FAIL] `hotpot_val_0002` (hotpotqa/agent): gold mismatch: gold 'Greenwich Village, New York City' not found in prediction
- [PASS] `hotpot_val_0003` (hotpotqa/agent): gold ok: contains
- [PASS] `hotpot_val_0004` (hotpotqa/agent): gold ok: contains
- [PASS] `hotpot_val_0005` (hotpotqa/agent): gold ok: token_cover
- [PASS] `hotpot_val_0006` (hotpotqa/agent): gold ok: contains
- [PASS] `hotpot_val_0007` (hotpotqa/agent): gold ok: token_cover
- [PASS] `hotpot_val_0008` (hotpotqa/agent): gold ok: contains
- [PASS] `hotpot_val_0009` (hotpotqa/agent): gold ok: contains

## License

GSM8K: MIT (OpenAI). HotpotQA: CC BY-SA 4.0. Redistributed subset for evaluation only.

---

- git: `b4d29f6`
- archived_json: `docs/snapshots/public_benchmark_snapshot_agent_20260717.json`
- reproduce: `python examples/run_public_benchmark.py --modes agent --publish`
