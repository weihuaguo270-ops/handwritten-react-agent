
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
import subprocess
import time
from urllib import request as req
from urllib.error import URLError
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os as _os
from mcp_client import MCPClient
from orchestrator import Orchestrator
MCP_CLIENTS = []

DEFAULT_MCP_SERVERS = [
    ["uvx", "mcp-server-time"],
    # 取消注释下一行可启用文件系统 Server：
    ["C:/Program Files/nodejs/npx.cmd", "-y", "@modelcontextprotocol/server-filesystem", "D:/agent_learning/repo"],
]
_os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'


# ============================================================
# 记忆系统
# ============================================================
class Memory:
    def __init__(self, save_path=r"D:\agent_learning\memory.json"):
        self.save_path = save_path
        self.facts = []
        self.vecs = []
        self.model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
        self._load()
    
    def _load(self):
        try:
            import json
            with open(self.save_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.facts = data.get("facts", [])
            import numpy as np
            self.vecs = [np.array(v) for v in data.get("vecs", [])]
            print(f"[记忆] 已加载 {len(self.facts)} 条记忆")
        except:
            self.facts = []
            self.vecs = []
    
    def _save(self):
        import json
        import numpy as np
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump({
                "facts": self.facts,
                "vecs": [[round(float(x), 4) for x in v] for v in self.vecs],
            }, f, ensure_ascii=False, separators=(",", ":"))
    
    def add(self, fact):
        if fact not in self.facts:
            self.facts.append(fact)
            vec = self.model.encode(fact)
            self.vecs.append(vec)
            self._save()
            return True
        return False
    
    def query(self, question, top_k=3):
        if not self.facts:
            return []
        try:
            q_vec = self.model.encode(question)
            scores = cosine_similarity([q_vec], self.vecs)[0]
            results = []
            for idx in scores.argsort()[::-1][:top_k]:
                if scores[idx] > 0.5:
                    results.append({"fact": self.facts[idx], "score": float(scores[idx])})
            return results
        except Exception:
            return []


MEMORY = Memory()




# 全局 MCP 客户端实例（默认 None，启动时通过参数初始化）


# ============================================================
# 第一步：配置（换成你的 API Key 和地址）
# ============================================================（换成你的 API Key 和地址）
# ============================================================
API_KEY='***'
BASE_URL = "https://api.deepseek.com"    # DeepSeek 官方地址
MODEL = "deepseek-v4-flash"               # DeepSeek V4 Flash

# ============================================================
# 第二步：定义工具（就像 C 里声明函数）
# ============================================================
def tool_calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "错误：非法字符"
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"计算错误：{e}"

def tool_get_time() -> str:
    """返回当前时间"""
    return time.strftime("%Y-%m-%d %H:%M:%S")


import urllib.request as _req
import urllib.parse as _parse
import urllib.parse as _up

def tool_web_search(query: str, max_results: int = 1) -> str:
    """搜索互联网，返回实时新闻结果"""
    try:
        import json as _json
        import urllib.request as _ur

        max_results = min(max(1, max_results), 5)  # 至少1条，最多5条
        payload = _json.dumps({
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

        req = _ur.Request(
            "https://api.anysearch.com/mcp",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0"
            },
            method="POST"
        )

        with _ur.urlopen(req, timeout=15) as resp:
            data = _json.loads(resp.read().decode())

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
        # 如果是维基百科，用 API 直接取纯文本
        if "wikipedia.org" in url:
            title = url.split("/wiki/")[-1].split("#")[0]
            from urllib.parse import quote as _q
            netloc = _up.urlparse(url).netloc
            api_url = (f"https://{netloc}/w/api.php"
                       f"?action=query&prop=extracts&explaintext"
                       f"&titles={_q(title)}&format=json&exchars=3000")
            r = _req.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
            with _req.urlopen(r, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            for pid, pdata in pages.items():
                if pid != "-1" and "extract" in pdata:
                    text = pdata["extract"].strip()
                    if len(text) > 3000:
                        text = text[:3000] + "\n\n...(截取)"
                    return text if text else "页面无内容"

        # 非维基百科：请求网页
        r = _req.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        })
        with _req.urlopen(r, timeout=10) as resp:
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
    system_prompt = """你是一个可以使用工具的 AI 助手。规则：
1. 用 THOUGHT / ACTION / OBSERVATION / FINAL ANSWER 格式
2. 最终答案用 FINAL ANSWER: 开头
3. 根据用户问题选择最合适的工具——包括本地工具和 MCP 远程工具
4. 搜索2次没结果就直接回答，不要继续搜"""

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

        # (1) 调 LLM（支持传入自定义工具列表）
        _tools = tool_defs if tool_defs is not None else None
        msg = call_llm(messages, tool_defs=_tools)
        last_content = msg.get("content", "") or ""
        if last_content.strip():
            print(f"[LLM思考] {last_content[:200]}")

        # (2) LLM 回复加入对话历史
        messages.append(msg)

        # (3) 检查 LLM 是否要调工具
        tool_calls = msg.get("tool_calls", [])

        if not tool_calls:
            # 检查是否有最终答案标记
            if "FINAL ANSWER:" in last_content.upper():
                final = last_content.split("FINAL ANSWER:", 1)[1].strip()
                print(f"\n>>> 最终答案: {final}")
                return final
            # 上一步用了工具，这一步没调但给出了实质内容 → 作为答案
            if tools_were_used and len(last_content.strip()) > 10:
                print(f"\n>>> 最终答案: {last_content.strip()}")
                return last_content
            # 连续 4 步寒暄（没调工具也不是明确答案）→ 结束
            if not tools_were_used and len(last_content.strip()) > 5 and step >= 4:
                print(f"\n(连续 {step} 步寒暄未调用工具，自动结束)")
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

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })

    print(f"\n(达到最大步骤 {max_steps}，停止)")
    if last_content.strip():
        print(f">>> 最终答案: {last_content.strip()}")
    return last_content

# ============================================================
# 运行测试
# ============================================================



# ============================================================
# 多 Agent 协作（Orchestrator-Worker 链式调用）
# ============================================================

def multi_agent_chain(user_query):
    """多 Agent 协作（内部使用 Orchestrator 类）"""
    return Orchestrator(call_llm, react_loop, tool_definitions=TOOL_DEFINITIONS).execute(user_query)

if __name__ == "__main__":
    import sys as _sys
    _sys_argv = _sys.argv[1:]
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

    if _sys_argv:
        q = " ".join(_sys_argv)
        memories = MEMORY.query(q)
        memory_context = ""
        if memories:
            memory_context = "\n".join([f"（相关记忆：{m['fact']}）" for m in memories])
        try:
            full_q = memory_context + q if memory_context else q
            if any(w in q for w in ["同时", "并且", "还有", "另外", "且"]):
                multi_agent_chain(full_q)
            else:
                react_loop(full_q)
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
                    multi_agent_chain(full_q)
                else:
                    react_loop(full_q)
            except Exception as e:
                import traceback; traceback.print_exc()
            if "记住" in q:
                fact = q.split("记住", 1)[1].strip().lstrip(" ，,、。.：:")
                if fact and MEMORY.add(fact):
                    print(f"\n[记忆] 已记住: {fact}")
                    print(f"[记忆] 当前共 {len(MEMORY.facts)} 条")
