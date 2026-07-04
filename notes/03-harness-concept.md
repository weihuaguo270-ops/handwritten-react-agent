# Harness 概念统一设计思考

## 背景

原来的项目把沙箱隔离（sandbox.py）、轨迹记录（harness.py）、轨迹回放（replay.py）放在三个独立的顶层文件里，概念上比较分散。

用户指出：Agent = LLM + Harness。Harness 应该是一个统一的保障层，而不是三个互不相干的文件。

## 合并方案

将三个文件移到 `harness/` 目录下：

```
harness/              # Agent 运行保障层
├── __init__.py       # 统一入口（对外暴露 Harness / Recorder / Sandbox / Replay）
├── recorder.py       # 轨迹记录
├── sandbox.py        # 子进程隔离
└── replay.py         # 离线回放
```

## 设计思路

1. 统一入口：`Harness` 类同时管理 recorder/sandbox/replay
2. 旧接口保持兼容：`start_trajectory()`, `finish_trajectory()`, `SANDBOX` 仍可直接 import
3. 沙箱预热：新增 `prewarm=True` 参数，启动时跑一次轻量子进程，让 Python 缓存字节码和模块导入

## 带来的好处

- 概念清晰：从文件结构上就能看出 Harness = Recorder + Sandbox + Replay
- 导入方便：`from harness import Harness` 一个入口即可
- 测试不变：46 项单元测试全部通过，不需要改测试逻辑
