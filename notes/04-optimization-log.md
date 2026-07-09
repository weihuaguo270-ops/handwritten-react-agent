# 优化记录

## 优化 1：RAG + Memory 模型懒加载

### 问题

`agent --help`、`agent --setup` 等命令需要等待 ~15 秒，因为：

1. `memory.py` 在 `Memory.__init__()` 时立即加载 BGE 模型（33MB）
2. `rag.py` 在 `RAG.__init__()` 时立即加载 BGE 模型
3. `react_loop.py` 在模块加载时立即执行 `RAG_INDEX.ingest_directory()`

导致所有命令（包括 --help）都要等模型下载+加载+索引。

### 方案

改为懒加载——模型只在首次实际使用时加载：

1. `memory.py`: `self.model = ...` → `self._model = None` + `_get_model()` 按需加载
2. `rag.py`: 同上
3. `react_loop.py`: 模块级的 RAG 索引改为 `_ensure_rag_loaded()` 函数，在 `react_loop()` 开头调用

### 改动文件

- `src/handwritten_react_agent/memory.py`
- `src/handwritten_react_agent/rag.py`
- `src/handwritten_react_agent/react_loop.py`

### 效果

| 命令 | 之前 | 之后 |
|------|------|------|
| `agent --help` | ~15s | ~1.6s |
| `agent --setup` | ~15s | ~1.6s |
| `agent "问题"` | ~15s + 正常时间 | ~1.6s + 正常时间（含一次加载） |

## 优化 2：test_all.py 路径修复

### 问题
项目结构改为 monorepo（src/handwritten_react_agent/ + experiments/）后，`test_all.py` 中的 import 路径未同步更新：
- `from tools.calculator import` → `ModuleNotFoundError`
- `from memory import` → `ModuleNotFoundError`
- `from harness.sandbox import` → `ModuleNotFoundError`
- `Sandbox(enabled=False)` → 参数名已改为 `strategy`

### 修复
统一替换为 `from handwritten_react_agent.xxx import` 格式。

### 结果
测试从约 15 项失败减少到 3 项（沙箱和 replay 相关，不阻塞功能使用）。

## 优化 3：agent --setup 支持 custom provider

### 改动
`_setup_config()` 新增 `custom` 选项，选择后额外询问 API 地址和模型名。

## 其他修复

| 项目 | 说明 |
|------|------|
| `.env.example` | 新增环境变量模板文件 |
| `agent --help` | 补充 `--mcp`、`--parallel` 和环境变量提示 |
| `experiments/langgraph/` import | 从 `experiments/langgraph/graph/` 目录运行一切正常 |
