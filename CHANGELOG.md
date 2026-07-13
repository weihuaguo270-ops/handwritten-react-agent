# Changelog

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
