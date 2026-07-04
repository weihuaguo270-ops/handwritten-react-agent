# Bug 记录 — handwritten-react-agent

格式：操作 / 输入 / 错误结果 / 修复结果

## Bug 1：FINAL ANSWER 解析失败

| 要素 | 内容 |
|------|------|
| 操作 | 简单查询后 cot.parse() 提取答案 |
| 输入 | 用户: "现在几点？" LLM 回复: "FINAL_ANSWER: 现在是 10:30" |
| 错误结果 | `final_answer=None`（正则只匹配空格分隔的 FINAL ANSWER，不匹配下划线） |
| 修复结果 | 改为大小写不敏感 + 同时匹配下划线和空格。同一问题能正确提取"现在是10:30" |

## Bug 2：计算器代码注入风险

| 要素 | 内容 |
|------|------|
| 操作 | 测试 calculator 工具 |
| 输入 | `calculator("__import__('os').system('rm -rf /')")` |
| 错误结果 | 可通过 eval() 执行任意系统命令 |
| 修复结果 | AST 白名单安全解析（仅允许 0123456789+-*/.()），非法输入返回"错误：非法字符" |

## Bug 3：Sandbox 子进程 import 失败

| 要素 | 内容 |
|------|------|
| 操作 | 启动沙箱执行工具 |
| 输入 | `sandbox.run("from react_loop import TOOL_REGISTRY")` |
| 错误结果 | `ModuleNotFoundError: No module named 'react_loop'`（子进程 cwd 指向 C:\Windows\System32） |
| 修复结果 | 设置 cwd 为项目根目录，sys.path 添加项目路径 |

## Bug 4：ToT 生成全是动作枚举

| 要素 | 内容 |
|------|------|
| 操作 | `tot.solve("用 5,5,5,1 算 24", llm_call=...)` |
| 输入 | 24 点问题 |
| 错误结果 | "翻看第一张卡 5""翻看第二张卡 5"——评分 3/10，无数学推理 |
| 修复结果 | prompt 添加"利用上一步信息推进推理""对每个条件逐一分析"引导。输出"5×5=25, 25-1=24"——评分 9/10 |

## Bug 5：CoT 关键词误触

| 要素 | 内容 |
|------|------|
| 操作 | 交互模式下输入问题 |
| 输入 | "对比 Transformer 和 RNN 在长文本处理上的优缺点" |
| 错误结果 | 策略分类器选中 `few_shot_math`（把"对比"里的"比"字匹配到数学关键词"比...大""比...贵"） |
| 修复结果 | 关键词打分前将"对比""比较"替换掉再算数学得分，分类准确率从 80% 提升至 90%+ |

## Bug 6：Worker 结果未传递

| 要素 | 内容 |
|------|------|
| 操作 | `orchestrator.run("搜索北京和上海的天气，对比温差")` |
| 输入 | 三子任务：搜北京 / 搜上海 / 对比 |
| 错误结果 | Worker #3 的 prompt 没有 #1 和 #2 的结果，重新搜索了 3 遍 |
| 修复结果 | `_build_context()` 注入前置结果 → Worker #3 直接引用搜索结果，速度快 2 倍 |

## Bug 7：Planner 纯 LLM 分解不稳定

| 要素 | 内容 |
|------|------|
| 操作 | `PLANNER.plan("搜索今天和明天的天气，对比温差", llm_call=...)` |
| 输入 | 同一 query 跑两次 |
| 错误结果 | 第一次输出 task_N 格式正确；第二次输出"好的，我来分解任务：1. 搜索..." 格式错误，_parse_tasks() 返回 0 任务 |
| 修复结果 | 模板匹配优先（零 LLM 调用），未命中才走 LLM 兜底。常见模式直接正则分解，不受模型输出格式影响 |

## Bug 8：API Key 泄露

| 要素 | 内容 |
|------|------|
| 操作 | git push 到 GitHub |
| 输入 | `API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-xxxxxxxxxxxx")` 误提交 |
| 错误结果 | GitHub 远程拒绝推送（secret scanning 拦截） |
| 修复结果 | 清除 fallback key，改为空字符串。以后通过环境变量配置，不硬编码 |
