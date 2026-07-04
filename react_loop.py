
"""
手写 ReAct Loop - 最小可用版本
不用任何框架，纯 Python + OpenAI 兼容 API
理解这个代码 = 理解了 Agent 最核心的机制
"""


import sys
import os
# ensure mcp_client.py can be found
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import re
import time
from urllib import request as req
from urllib.error import URLError
from urllib.parse import urlparse, quote
from mcp_client import MCPClient
from orchestrator import Orchestrator
from cot import COT, COT_TOOL_DEFINITION, tool_switch_cot_strategy
from tot import TOT, TOT_TOOL_DEFINITION, tool_tot_reasoning, set_tot_llm_call
from prompts import ROLE_MANAGER, ROLE_TOOL_DEFINITION, tool_switch_role
from context import CONTEXT, CONTEXT_TOOL_DEFINITION, tool_switch_context_strategy
from harness import start_trajectory, current_trajectory, finish_trajectory
from harness import SANDBOX, SANDBOX_TOOL_DEFINITION, tool_toggle_sandbox
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
from memory import Memory

MEMORY = Memory()
from rag import RAG_INDEX, rag_query, RAG_TOOL_DEFINITION

# ============================================================
# 预加载 RAG 知识库：启动时自动索引项目文档
# ============================================================
print("[启动] 正在加载 RAG 知识库...")
_rag_dir = os.path.dirname(os.path.abspath(__file__))
try:
    n = RAG_INDEX.ingest_directory(_rag_dir)
    print(f"[启动] RAG 知识库就绪：{len(RAG_INDEX.chunks)} 个片段 (来自 {n} 个文件)")
except Exception as e:
    print(f"[启动] RAG 知识库加载跳过: {e}")


# ============================================================
# 第一步：配置
# ============================================================
# 方式一（推荐）：设置环境变量，避免 API Key 被提交到 Git
#   Windows: set DEEPSEEK_API_KEY=sk-xxx
#   Linux/Mac: export DEEPSEEK_API_KEY=sk-xxx
#
# 方式二：直接在下方填入 API Key（注意：不要提交到 Git）
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
#                           ↑ 环境变量名   ↑ 留空则不设默认值
BASE_URL = "https://api.deepseek.com"    # DeepSeek 官方 API 地址
MODEL = "deepseek-v4-flash"               # DeepSeek V4 Flash

# ============================================================
# 第二步：定义工具（就像 C 里声明函数）
# ============================================================
def tool_calculator(expression: str) -> str:
    """计算数学表达式（基于 AST 解析，安全无注入风险）"""
    import ast
    try:
        # 解析表达式为 AST，只允许数学运算
        tree = ast.parse(expression.strip(), mode='eval')
        ALLOWED_NODES = (ast.Expression, ast.Constant, ast.UnaryOp,
                         ast.UAdd, ast.USub, ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div)
        for node in ast.walk(tree):
            if not isinstance(node, ALLOWED_NODES):
                return "错误：表达式包含非法操作"
        # AST 通过检查，编译执行
        code = compile(tree, '<string>', 'eval')
        result = eval(code)
        return str(result)
    except SyntaxError:
        return "错误：表达式语法错误"
    except Exception as e:
        return f"计算错误：{e}"

def tool_get_time() -> str:
    """返回当前时间"""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def tool_web_search(query: str, max_results: int = 1) -> str:
    """搜索互联网，返回实时新闻结果"""
    try:

        max_results = min(max(1, max_results), 5)
        payload = json.dumps({
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "search",
                "arguments": {
                    "query": query,
                    "content_types": "news",
                    "max_results": max_results,
                    "zone": "intl"
                }
            },
            "id": 1
        }).encode()

        http_request = req.Request(
            "https://api.anysearch.com/mcp",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0"
            },
            method="POST"
        )

        with req.urlopen(http_request, timeout=15) as resp:
            data = json.loads(resp.read().decode())

        result_text = data.get("result", {}).get("content", [{}])[0].get("text", "")
        if not result_text or "Search Results" not in result_text:
            return "搜索未找到相关结果"

        # 解析 markdown 格式的结果
        results = []
        idx = 0
        for line in result_text.split("\n"):
            if line.startswith("### "):
                idx += 1
                title = line[4:].strip()
                title = title.split(". ", 1)[1] if ". " in title else title
                results.append(f"{idx}. {title}")
            elif line.startswith("- **URL**"):
                url = line.replace("- **URL**: ", "").strip()
                results.append(f"   链接: {url}")
            elif line.startswith("- ") and not line.startswith("- **"):
                snippet = line[2:].strip()
                if snippet:
                    results.append(f"   {snippet[:300]}")

        return "\n".join(results) if results else "搜索未找到相关结果"

    except Exception as e:
        return f"搜索出错: {e}"

def tool_fetch_page(url: str) -> str:
    """读取网页内容并提取正文"""
    try:
        # 验证 URL，防止 SSRF
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return f"不支持的协议: {parsed.scheme}"
        # 简单内网地址阻断
        netloc = parsed.netloc.split(":")[0]
        if netloc in ("127.0.0.1", "localhost", "0.0.0.0", "::1"):
            return "不允许访问内网地址"
        if netloc.startswith("10.") or netloc.startswith("172.") or netloc.startswith("192.168."):
            return "不允许访问内网地址"

        # 如果是维基百科，用 API 直接取纯文本
        if "wikipedia.org" in url:
            title = url.split("/wiki/")[-1].split("#")[0]
            netloc = urlparse(url).netloc
            api_url = (f"https://{netloc}/w/api.php"
                       f"?action=query&prop=extracts&explaintext"
                       f"&titles={quote(title)}&format=json&exchars=3000")
            r = req.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
            with req.urlopen(r, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            for pid, pdata in pages.items():
                if pid != "-1" and "extract" in pdata:
                    text = pdata["extract"].strip()
                    if len(text) > 3000:
                        text = text[:3000] + "\n\n...(截取)"
                    return text if text else "页面无内容"

        # 非维基百科：请求网页
        r = req.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        })
        with req.urlopen(r, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        # 提取 <p> 标签中的正文段落
        paras = re.findall(
            r'<p[^>]*>([^<]+(?:<[^/][^>]*>[^<]*</[^>]+>)?[^<]*)</p>',
            html, re.DOTALL
        )
        if paras:
            text = "\n".join(p.strip() for p in paras)
            text = re.sub(r'<[^>]+>', '', text)
            text = re.sub(r'\n{3,}', '\n\n', text).strip()
        else:
            text = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)
            text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', '\n', text)
            text = re.sub(r'\n[ \t]+\n', '\n', text)
            text = re.sub(r'\n{3,}', '\n\n', text).strip()

        if len(text) > 3000:
            text = text[:3000] + "\n\n...(截取)"
        return text if text else "页面无正文可提取"
    except Exception as e:
        return f"读取失败: {e}"

def tool_summarize(text: str, max_sentences: int = 5) -> str:
    """自动提取文本摘要（抽取式：取前几个关键句子）"""
    if not text or len(text) < 20:
        return "文本过短，无需摘要"

    # 按句号、问号、感叹号、换行分割句子
    sentences = re.split(r'[。！？\n]', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    if not sentences:
        return text[:500]

    # 取前 max_sentences 个有内容的句子
    summary = "。".join(sentences[:max_sentences]) + "。"
    if len(summary) > 1000:
        summary = summary[:1000] + "..."

    return summary



TOOL_REGISTRY = {
    "get_time": tool_get_time,
    "calculator": tool_calculator,
    "web_search": tool_web_search,
    "fetch_page": tool_fetch_page,
    "summarize": tool_summarize,
    "rag_query": rag_query,
    "switch_cot_strategy": tool_switch_cot_strategy,
    "tot_reasoning": tool_tot_reasoning,
    "switch_role": tool_switch_role,
    "switch_context_strategy": tool_switch_context_strategy,
    "toggle_sandbox": tool_toggle_sandbox,
    "start_dashboard": tool_start_dashboard,
}

# 工具的 JSON 描述（发给 LLM 让它知道能调什么）
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "搜索互联网新闻和网页，获取实时信息（基于AnySearch搜索引擎）",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "返回结果数量，默认1条。用户明确说了数量才传更大值"
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学表达式",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如 '2 + 3 * 4'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": "读取网页内容，输入URL返回正文文本",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "要读取的网页地址"
                    }
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize",
            "description": "自动提取文本摘要，输入长文本返回精简摘要",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "要总结的文本内容"
                    },
                    "max_sentences": {
                        "type": "integer",
                        "description": "摘要保留的句子数（默认5）"
                    }
                },
                "required": ["text"],
            },
        },
    },
    RAG_TOOL_DEFINITION,
    COT_TOOL_DEFINITION,
    TOT_TOOL_DEFINITION,
    ROLE_TOOL_DEFINITION,
    CONTEXT_TOOL_DEFINITION,
    SANDBOX_TOOL_DEFINITION,
    {
        "type": "function",
        "function": {
            "name": "start_dashboard",
            "description": "启动 Dashboard Web 界面（轨迹查看器 + 聊天面板），在浏览器中直观查看 Agent 的思考过程和工具调用",
            "parameters": {
                "type": "object",
                "properties": {
                    "port": {
                        "type": "integer",
                        "description": "端口号（默认 5050）"
                    }
                },
                "required": []
            }
        }
    },
]

# ============================================================
# 第三步：调用 LLM
# ============================================================
def call_llm(messages, max_retries=2, tool_defs=None):
    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": tool_defs if tool_defs is not None else TOOL_DEFINITIONS,
        "tool_choice": "auto",
        "temperature": 0.7,
        "max_tokens": 2000,
    }
    r = req.Request(
        f"{BASE_URL}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )
    for attempt in range(max_retries):
        try:
            with req.urlopen(r, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result["choices"][0]["message"]
        except URLError as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return {"role": "assistant", "content": f"LLM失败: {e}"}
        except json.JSONDecodeError as e:
            return {"role": "assistant", "content": f"解析失败: {e}"}
    return {"role": "assistant", "content": "超过最大重试"}


# ToT 模块的 LLM 调用适配器（它在内部需要调 LLM 做生成和评估）
def _tot_llm_wrapper(prompt: str) -> str:
    """把 ToT 的 (prompt → reply) 签名适配成 react_loop 的 call_llm"""
    messages = [{"role": "user", "content": prompt}]
    result = call_llm(messages, tool_defs=[])
    return result.get("content", "")


set_tot_llm_call(_tot_llm_wrapper)


DASHBOARD_PROCESS = []

def tool_start_dashboard(port: int = 5050) -> str:
    """启动 Dashboard Web 界面（轨迹查看器 + 聊天面板）"""
    import subprocess
    if DASHBOARD_PROCESS and DASHBOARD_PROCESS[0].poll() is None:
        return f"Dashboard 已在运行: http://127.0.0.1:{port}"
    try:
        server_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard", "server.py")
        if not os.path.exists(server_script):
            return f"错误: 找不到 dashboard/server.py"
        proc = subprocess.Popen(
            [sys.executable, server_script],
            cwd=os.path.dirname(server_script),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        DASHBOARD_PROCESS.append(proc)
        return f"Dashboard 已启动: http://127.0.0.1:{port}（记得在浏览器打开）"
    except Exception as e:
        return f"启动失败: {e}"


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
        # 沙箱启用时通过子进程执行
        if SANDBOX.enabled:
            sandbox_result = SANDBOX.run(tool_call)
            if sandbox_result != "__SANDBOX_DISABLED__":
                return sandbox_result
        # 直接执行（沙箱关闭或回退）
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
    start_trajectory(user_query, MODEL, system_prompt)

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
            if MEMORY.add(fact):
                saved += 1
    if saved > 0:
        print(f"[记忆] 自动记忆: 保存了 {saved} 条新信息")
    return saved


def multi_agent_chain(user_query, parallel=False):
    """多 Agent 协作（内部使用 Orchestrator 类）"""
    return Orchestrator(call_llm, react_loop, tool_definitions=TOOL_DEFINITIONS).execute(user_query, parallel=parallel)

def cli_entry():
    """CLI 入口点：供 pip 安装后的 'agent' 命令使用"""
    main()

def main():
    """命令行入口：支持交互模式和单次问题模式。"""
    global TOOL_DEFINITIONS

    if not API_KEY.strip():
        print("错误：未配置 DEEPSEEK_API_KEY，也没有在 react_loop.py 中设置 fallback API_KEY。")
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
            if "所有" in target or "全部" in target:
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
                if "所有" in target or "全部" in target:
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
