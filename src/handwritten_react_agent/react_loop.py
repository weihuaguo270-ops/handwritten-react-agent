
"""
手写 ReAct Loop - 最小可用版本
不用任何框架，纯 Python + OpenAI 兼容 API
理解这个代码 = 理解了 Agent 最核心的机制

LLM 配置：通过 llm_config.json + LLM_PROVIDER 环境变量控制。
  export LLM_PROVIDER=deepseek    # 使用 DeepSeek
  export LLM_PROVIDER=openai      # 使用 OpenAI
  export LLM_PROVIDER=ollama      # 使用本地 Ollama
  默认使用 llm_config.json 中的 default 字段。
"""


import sys
import os
# ensure mcp_client.py can be found

import json
import re
import time
from urllib import request as req
from urllib.error import URLError
from urllib.parse import urlparse, quote
from handwritten_react_agent.mcp_client import MCPClient
from handwritten_react_agent.orchestrator import Orchestrator
from handwritten_react_agent.tot import TOT, set_tot_llm_call
from handwritten_react_agent.cot import COT
from handwritten_react_agent.prompts import ROLE_MANAGER
from handwritten_react_agent.context import CONTEXT
from handwritten_react_agent.harness import start_trajectory, current_trajectory, finish_trajectory
from handwritten_react_agent.harness import SANDBOX
from handwritten_react_agent.llm import LLM_DEFAULT, LLM
from handwritten_react_agent.tools import TOOL_REGISTRY, TOOL_DEFINITIONS
MCP_CLIENTS = []

DEFAULT_MCP_SERVERS = [
    ["uvx", "mcp-server-time"],
    # 取消注释下一行可启用文件系统 Server：
    ["C:/Program Files/nodejs/npx.cmd", "-y", "@modelcontextprotocol/server-filesystem", "D:/agent_learning/repo"],
]
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'


# ============================================================
# 记忆系统（独立模块见 memory.py）
# ============================================================
from handwritten_react_agent.memory import Memory

MEMORY = Memory()
from handwritten_react_agent.rag import RAG_INDEX, rag_query, RAG_TOOL_DEFINITION

# ============================================================
# 懒加载 RAG 知识库：只在首次需要时加载，避免 --help/--setup 等待
# ============================================================
_rag_loaded = False

def _ensure_rag_loaded():
    """首次调用时加载 RAG 知识库，后续不再重复"""
    global _rag_loaded
    if _rag_loaded:
        return
    _rag_loaded = True
    print("[启动] 正在加载 RAG 知识库...")
    _rag_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        n = RAG_INDEX.ingest_directory(_rag_dir)
        print(f"[启动] RAG 知识库就绪：{len(RAG_INDEX.chunks)} 个片段 (来自 {n} 个文件)")
    except Exception as e:
        print(f"[启动] RAG 知识库加载跳过: {e}")


# ============================================================
# 第一步：配置 — 通过 llm_config.json + LLM_PROVIDER 环境变量
# ============================================================
# 当前 provider 由 LLM_DEFAULT 决定（模块加载时自动从配置读取）
# 切换 provider 方式：
#   1. Windows: set LLM_PROVIDER=openai
#   2. Linux:   export LLM_PROVIDER=ollama
#   3. 修改 llm_config.json 中的 default 字段
#   4. 代码中手动创建：llm = LLM(provider="openai")
# 注意：不要提交 API Key 到 Git！
_current_llm = LLM_DEFAULT

# ============================================================
# 第二步：工具 — 由 tools/ 模块统一管理
# 新增工具只需在 tools/ 下加文件，tools/__init__.py 自动注册
# ============================================================
# TOOL_REGISTRY 和 TOOL_DEFINITIONS 已从 tools 中导入。
# 注意：TOOL_DEFINITIONS 在 main() 中会被 MCP 工具扩展。

# ============================================================
# 第三步：调用 LLM
# ============================================================
def call_llm(messages, max_retries=2, tool_defs=None,
             temperature=None, max_tokens=None):
    """调用 LLM，返回消息对象。使用 _current_llm（可通过 LLM_PROVIDER 切换）"""
    return _current_llm.chat(
        messages,
        tool_defs=tool_defs if tool_defs is not None else TOOL_DEFINITIONS,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
    )


# ToT 模块的 LLM 调用适配器（它在内部需要调 LLM 做生成和评估）
def _tot_llm_wrapper(prompt: str) -> str:
    """把 ToT 的 (prompt → reply) 签名适配成 react_loop 的 call_llm"""
    messages = [{"role": "user", "content": prompt}]
    result = call_llm(messages, tool_defs=[])
    return result.get("content", "")


set_tot_llm_call(_tot_llm_wrapper)


# ============================================================
# 第四步：执行工具
# ============================================================
def execute_tool_call(tool_call):
    func_name = tool_call["function"]["name"]
    try:
        arguments = json.loads(tool_call["function"]["arguments"])
    except json.JSONDecodeError:
        return '{"error": "参数解析失败"}'
    # 先查本地注册的工具
    if func_name in TOOL_REGISTRY:
        # 沙箱判断（off/auto/on 三策略）
        if SANDBOX.strategy != "off" and SANDBOX.should_sandbox(func_name):
            sandbox_result = SANDBOX.run(tool_call)
            if sandbox_result != "__SANDBOX_DISABLED__":
                return sandbox_result
        # 直接执行（沙箱关闭或 safe 工具）
        try:
            return str(TOOL_REGISTRY[func_name](**arguments))
        except Exception as e:
            return json.dumps({"error": f"执行错误: {e}"})
    # 不在本地注册表 → 尝试遍历所有 MCP Client
    for _mcp_client in MCP_CLIENTS:
        if func_name in [t["name"] for t in _mcp_client.tools]:
            try:
                print(f"  [MCP] 转发: {func_name}({json.dumps(arguments, ensure_ascii=False)[:100]})")
                return _mcp_client.call_tool(func_name, arguments)
            except Exception as e:
                return json.dumps({"error": f"MCP调用失败: {e}"})
    return json.dumps({"error": f"未知工具: {func_name}"})

# ============================================================
# 第五步：ReAct Loop 主循环（核心！）
# ============================================================
def react_loop(user_query, max_steps=10, tool_defs=None):
    _ensure_rag_loaded()
    base_prompt = """你是一个可以使用工具的 AI 助手。规则：
1. 用 THOUGHT / ACTION / OBSERVATION / FINAL ANSWER 格式
2. 最终答案用 FINAL ANSWER: 开头
3. 根据用户问题选择最合适的工具——包括本地工具和 MCP 远程工具
4. 搜索2次没结果就直接回答，不要继续搜"""
    # 角色注入 → CoT 注入（角色先定风格，CoT 再定推理方式）
    role_enhanced = ROLE_MANAGER.inject(base_prompt, query=user_query)
    system_prompt = COT.inject(role_enhanced, query=user_query)
    print(f"[角色] {ROLE_MANAGER.current_role_name()}")

    # 开始轨迹记录
    start_trajectory(user_query, _current_llm.model, system_prompt)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]

    print(f"\n{'='*60}")
    print(f"用户: {user_query}")
    print(f"{'='*60}\n")

    last_content = ""
    tools_were_used = False
    search_count = 0
    for step in range(1, max_steps + 1):
        print(f"--- Step {step}/{max_steps} ---")
        traj = current_trajectory()
        if traj:
            traj.start_step(step)

        # (1) 调 LLM（支持传入自定义工具列表）
        msg = call_llm(messages, tool_defs=tool_defs)
        last_content = msg.get("content", "") or ""
        if last_content.strip():
            print(f"[LLM思考] {last_content[:200]}")
        if traj:
            traj.add_thought(step, last_content)

        # (2) LLM 回复加入对话历史
        messages.append(msg)

        # (3) 检查 LLM 是否要调工具
        tool_calls = msg.get("tool_calls", [])

        if not tool_calls:
            # 检查是否有最终答案标记（大小写不敏感）
            fa_match = re.search(r'FINAL ANSWER:\s*(.*)', last_content, re.IGNORECASE | re.DOTALL)
            if fa_match:
                final = fa_match.group(1).strip()
                print(f"\n>>> 最终答案: {final}")
                finish_trajectory(final)
                return final
            # 上一步用了工具，这一步没调但给出了实质内容 → 作为答案
            if tools_were_used and len(last_content.strip()) > 10:
                print(f"\n>>> 最终答案: {last_content.strip()}")
                finish_trajectory(last_content.strip())
                return last_content
            # 连续 4 步寒暄（没调工具也不是明确答案）→ 结束
            if not tools_were_used and len(last_content.strip()) > 5 and step >= 4:
                print(f"\n(连续 {step} 步寒暄未调用工具，自动结束)")
                finish_trajectory(last_content)
                return last_content
            continue

        # 执行工具
        tools_were_used = True
        for tc in tool_calls:
            name = tc["function"]["name"]
            args = tc["function"]["arguments"]
            print(f"[调工具] {name}({args})")

            # 搜索次数限制（只阻止搜索，不影响其他工具）
            if name == "web_search":
                search_count += 1
                if search_count >= 4:
                    print(f"  (搜索已达上限，跳过)")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": "搜索已达上限，请基于已有信息回答"
                    })
                    continue

            result = execute_tool_call(tc)
            print(f"[工具返回] {result[:100]}")

            if traj:
                traj.add_tool_call(step, name, args, result)

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })

        # (5) 每步结束：检查上下文窗口，超限则自动管理
        before = len(messages)
        messages = CONTEXT.manage(messages)
        if len(messages) != before or CONTEXT.last_action:
            if CONTEXT.last_action:
                print(f"  [上下文] {CONTEXT.last_action}")

    print(f"\n(达到最大步骤 {max_steps}，停止)")
    if last_content.strip():
        print(f">>> 最终答案: {last_content.strip()}")
    finish_trajectory(last_content.strip() if last_content.strip() else "")
    return last_content

# ============================================================
# 运行测试
# ============================================================



# ============================================================
# 多 Agent 协作（Orchestrator-Worker 链式调用）
# ============================================================

def auto_extract_memory(user_query, assistant_answer):
    """从对话中自动提取值得记住的信息（独立函数，依赖 call_llm）"""
    if not assistant_answer or len(assistant_answer) < 20 or any(w in user_query for w in ["忘记", "删除"]):
        return 0
    
    prompt = (
        "从以下对话中提取**具体的事实性信息**。\n"
        "规则：\n"
        "- 提取具体信息，如姓名、职业、爱好、背景、联系方式等\n"
        "- 每个事实单独一行\n"
        "- 忽略闲聊、问候、临时问题\n"
        "- 如果用户明确告知个人信息，务必提取\n"
        "- 没有任何具体事实就输出空行\n\n"
        f"用户: {user_query}\n\n"
        f"助手: {assistant_answer}\n\n"
        "事实:"
    )
    
    msg = call_llm([
        {"role": "system", "content": "你是一个信息提取助手。"},
        {"role": "user", "content": prompt},
    ])
    
    raw = msg.get("content", "") or ""
    if raw.startswith("LLM失败") or raw.startswith("解析失败"):
        return 0
    
    facts = [f.strip() for f in raw.split("\n") if f.strip()]
    saved = 0
    for fact in facts:
        if len(fact) > 5 and not any(w in fact for w in ["LLM失败", "错误", "抱歉", "值得记住", "信息:", "没有提供", "没有任何", "事实:", "个人信息"]):
            action, detail = MEMORY.add_or_update(fact)
            if action == "added":
                saved += 1
            elif action == "updated":
                print(f"[记忆] 更新: \"{detail}\" → \"{fact}\"")
                saved += 1
    if saved > 0:
        print(f"[记忆] 自动记忆: 保存了 {saved} 条新信息")
    return saved


def multi_agent_chain(user_query, parallel=False):
    """多 Agent 协作（内部使用 Orchestrator 类）"""
    return Orchestrator(call_llm, react_loop, tool_definitions=TOOL_DEFINITIONS).execute(user_query, parallel=parallel)

def _setup_config():
    """交互式配置向导：创建/更新 llm_config.json"""
    import json
    # 找配置文件的写入路径（当前工作目录）
    config_path = os.path.join(os.getcwd(), "llm_config.json")

    # 如果已有配置，先加载
    config = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
        except Exception:
            pass

    # 提供者模板
    providers = config.get("providers", {})
    defaults = {
        "deepseek": {
            "description": "DeepSeek 官方 API",
            "base_url": "https://api.deepseek.com",
            "api_key": "",
            "model": "deepseek-v4-flash",
            "temperature": 0.7, "max_tokens": 2000,
        },
        "openai": {
            "description": "OpenAI API（兼容 GPT-4o 等）",
            "base_url": "https://api.openai.com/v1",
            "api_key": "",
            "model": "gpt-4o-mini",
            "temperature": 0.7, "max_tokens": 2000,
        },
        "ollama": {
            "description": "本地 Ollama（需先启动 ollama serve）",
            "base_url": "http://localhost:11434/v1",
            "api_key": "",
            "model": "qwen2.5:7b",
            "temperature": 0.7, "max_tokens": 2000,
        },
        "custom": {
            "description": "任意 OpenAI 兼容 API（自定义地址）",
            "base_url": "",
            "api_key": "",
            "model": "",
            "temperature": 0.7, "max_tokens": 2000,
        },
    }

    print(f"\n{'='*50}")
    print("  Agent 配置向导")
    print(f"{'='*50}")
    print(f"  配置文件将保存到: {config_path}\n")

    # 1. 选择默认 provider
    provider_names = list(defaults.keys())
    print("可用的 LLM 提供商：")
    for i, name in enumerate(provider_names, 1):
        desc = defaults[name]["description"]
        print(f"  {i}. {name} - {desc}")
    print()

    default_provider = config.get("default", "deepseek")
    default_idx = provider_names.index(default_provider) + 1 if default_provider in provider_names else 1
    choice = input(f"选择提供商 (1-{len(provider_names)}, 默认 {default_idx}): ").strip()
    selected_idx = int(choice) - 1 if choice.isdigit() and 1 <= int(choice) <= len(provider_names) else default_idx - 1
    selected_provider = provider_names[selected_idx]

    # 2. 输入 API Key
    existing_key = providers.get(selected_provider, {}).get("api_key", "")
    key_hint = f" (已有: {existing_key[:8]}...)" if existing_key and len(existing_key) > 8 else ""
    api_key = input(f"输入 {selected_provider} 的 API Key{key_hint}: ").strip()
    if not api_key and existing_key:
        api_key = existing_key

    # 3. 输入模型名（可选）
    default_model = providers.get(selected_provider, {}).get("model", defaults[selected_provider]["model"])
    model = input(f"模型名 (默认 {default_model}): ").strip()
    if not model:
        model = default_model

    # 3b. 如果是 custom provider，额外询问 base_url
    base_url = providers.get(selected_provider, {}).get("base_url", defaults[selected_provider]["base_url"])
    if selected_provider == "custom":
        default_url = base_url or "https://api.openai.com/v1"
        base_url = input(f"API 地址 (默认 {default_url}): ").strip()
        if not base_url:
            base_url = default_url
        default_model = model or "gpt-4o-mini"
        model = input(f"模型名 (默认 {default_model}): ").strip()
        if not model:
            model = default_model

    # 4. 构建配置
    providers[selected_provider] = {
        "description": defaults[selected_provider]["description"],
        "base_url": providers.get(selected_provider, {}).get("base_url", defaults[selected_provider]["base_url"]),
        "api_key": api_key,
        "model": model,
        "temperature": providers.get(selected_provider, {}).get("temperature", 0.7),
        "max_tokens": providers.get(selected_provider, {}).get("max_tokens", 2000),
    }
    config["default"] = selected_provider
    config["providers"] = providers

    # 5. 保存
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 配置已保存到: {config_path}")
    print(f"   默认提供商: {selected_provider}")
    print(f"   模型: {model}")
    print(f"\n现在可以运行: agent \"你的问题\"")


def cli_entry():
    """CLI 入口点：供 pip 安装后的 'agent' 命令使用"""
    main()


def main():
    """命令行入口：支持交互模式和单次问题模式。"""
    global TOOL_DEFINITIONS

    # 处理配置相关命令（不依赖 LLM）
    _sys_argv = sys.argv[1:] if len(sys.argv) > 1 else []
    if "--setup" in _sys_argv or "setup" in _sys_argv:
        _setup_config()
        return
    if "--help" in _sys_argv or "-h" in _sys_argv:
        print("用法:")
        print("  agent                         进入交互模式")
        print('  agent "你的问题"               单次提问')
        print("  agent --setup                 运行配置向导")
        print("  agent --parallel \"多任务问题\"  并行多 Agent 协作")
        print("  agent --mcp uvx mcp-server-xxx  连接额外 MCP 服务器")
        print("  agent --help                  显示帮助")
        print()
        print("首次使用建议先运行: agent --setup")
        print("环境变量: DEEPSEEK_API_KEY / OPENAI_API_KEY / LLM_PROVIDER")
        return

    if not _current_llm or not _current_llm.api_key.strip():
        _provider = os.environ.get("LLM_PROVIDER",
                                    _current_llm.provider_name if _current_llm else "?")
        print(f"错误：未配置 {_provider} 的 API Key。")
        print()
        print("请选择以下方式之一配置：")
        print(f"  1. 运行配置向导:   agent --setup")
        print(f"  2. 设置环境变量:    set {_provider.upper()}_API_KEY=sk-xxx  (Windows)")
        print(f"                        export {_provider.upper()}_API_KEY=sk-xxx  (Linux)")
        print(f"  3. 编辑配置文件:    llm_config.json（与 agent 命令同目录）")
        print()
        sys.exit(1)

    _sys_argv = sys.argv[1:]
    _parallel_mode = "--parallel" in _sys_argv
    if _parallel_mode:
        _sys_argv.remove("--parallel")
    _mcp_args_list = []
    while "--mcp" in _sys_argv:
        idx = _sys_argv.index("--mcp")
        if idx + 1 < len(_sys_argv):
            _mcp_args_list.append(_sys_argv[idx + 1].split())
        _sys_argv = _sys_argv[:idx] + _sys_argv[idx + 2:]
    if not _mcp_args_list:
        _mcp_args_list = DEFAULT_MCP_SERVERS
    for mcp_args in _mcp_args_list:
        cmd = mcp_args[0]
        args = mcp_args[1:]
        print("  [MCP] connect")
        try:
            client = MCPClient(cmd, args)
            client.connect()
            client.discover_tools()
            mcp_defs = client.to_tool_definitions()
            _suppress = {"get_time"}
            TOOL_DEFINITIONS = [t for t in TOOL_DEFINITIONS if t["function"]["name"] not in _suppress]
            TOOL_DEFINITIONS.extend(mcp_defs)
            MCP_CLIENTS.append(client)
            print(f"  -> 隐藏本地重复工具，合并 {len(mcp_defs)} 个 MCP 工具")
        except Exception as e:
            print(f"  -> 连接失败: {e}\n")

    _skip_query = False
    if _sys_argv:
        q = " ".join(_sys_argv)
        # 处理"忘记/删除"——直接删，不走 react_loop
        if "忘记" in q or "删除" in q:
            target = q.split("忘记", 1)[1].strip() if "忘记" in q else q.split("删除", 1)[1].strip()
            if target in ("所有", "全部"):
                MEMORY.clear()
                print("\n[记忆] 已清空所有记忆")
            elif target:
                n = MEMORY.remove(target)
                if n > 0:
                    print(f"\n[记忆] 已删除相关记忆")
                else:
                    print(f"\n[记忆] 未找到匹配的记忆")
            _skip_query = True
        
        if not _skip_query:
            memories = MEMORY.query(q)
            memory_context = ""
            if memories:
                memory_context = "\n".join([f"（相关记忆：{m['fact']}）" for m in memories])
        try:
            full_q = memory_context + q if memory_context and not _skip_query else q
            if any(w in q for w in ["同时", "并且", "还有", "另外", "且"]):
                result = multi_agent_chain(full_q, parallel=_parallel_mode)
            else:
                result = react_loop(full_q)
            if result:
                auto_extract_memory(q, result)
        except Exception as e:
            import traceback; traceback.print_exc()
        
        if "记住" in q:
            fact = q.split("记住", 1)[1].strip().lstrip(" ，,、。.：:")
            if fact:
                MEMORY.add(fact)
                print(f"\n[记忆] 已记住: {fact}")
    else:
        print("\n" + "=" * 50)
        print("  Agent 交互模式已启动")
        print("  " + "=" * 50)
        tool_list = " / ".join(list(TOOL_REGISTRY.keys()))
        for _c in MCP_CLIENTS:
            mcp_names = [t["name"] for t in _c.tools]
            tool_list += " / " + " / ".join(mcp_names)
        print(f"  可用工具：{tool_list}")
        print("  退出：输入 'exit' 或 '退出'")
        print("  " + "=" * 50 + "\n")
        first = True
        while True:
            q = input("\n你 > " if not first else "你 > ")
            first = False
            if q.lower() in ("exit", "退出", "quit"):
                print("再见！")
                break
            if not q.strip():
                continue
            if q == "记忆":
                print("\n已保存的记忆:")
                if MEMORY.facts:
                    for i, fact in enumerate(MEMORY.facts, 1):
                        print(f"  {i}. {fact}")
                else:
                    print("  （无）")
                continue
            memories = MEMORY.query(q)
            memory_context = ""
            if memories:
                memory_context = "\n".join([f"（相关记忆：{m['fact']}）" for m in memories])
            try:
                full_q = memory_context + q if memory_context else q
                if any(w in q for w in ["同时", "并且", "还有", "另外", "且"]):
                    result = multi_agent_chain(full_q)
                else:
                    result = react_loop(full_q)
                if result:
                    auto_extract_memory(q, result)
            except Exception as e:
                import traceback; traceback.print_exc()
            if "忘记" in q or "删除" in q:
                target = q.split("忘记", 1)[1].strip() if "忘记" in q else q.split("删除", 1)[1].strip()
                if target in ("所有", "全部"):
                    MEMORY.clear()
                    print("\n[记忆] 已清空所有记忆")
                elif target:
                    n = MEMORY.remove(target)
                    if n > 0:
                        print(f"\n[记忆] 已删除相关记忆")
                    else:
                        print(f"\n[记忆] 未找到匹配的记忆")
                continue  # 直接下一轮
            
            if "记住" in q:
                fact = q.split("记住", 1)[1].strip().lstrip(" ，,、。.：:")
                if fact and MEMORY.add(fact):
                    print(f"\n[记忆] 已记住: {fact}")
                    print(f"[记忆] 当前共 {len(MEMORY.facts)} 条")


if __name__ == "__main__":
    main()
