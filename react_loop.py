
"""
手写 ReAct Loop - 最小可用版本
不用任何框架，纯 Python + OpenAI 兼容 API
理解这个代码 = 理解了 Agent 最核心的机制
"""

import json
import re
import time
from urllib import request as req
from urllib.error import URLError

# ============================================================
# 第一步：配置（换成你的 API Key 和地址）
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
    "calculator": tool_calculator,
    "get_time": tool_get_time,
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
            "name": "get_time",
            "description": "获取当前时间",
            "parameters": {"type": "object", "properties": {}},
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
def call_llm(messages, max_retries=2):
    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": TOOL_DEFINITIONS,
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
    if func_name in TOOL_REGISTRY:
        try:
            return str(TOOL_REGISTRY[func_name](**arguments))
        except Exception as e:
            return json.dumps({"error": f"执行错误: {e}"})
    return json.dumps({"error": f"未知工具: {func_name}"})

# ============================================================
# 第五步：ReAct Loop 主循环（核心！）
# ============================================================
def react_loop(user_query, max_steps=10):
    system_prompt = """你是一个可以使用工具的 AI 助手。规则：
1. 用 THOUGHT / ACTION / OBSERVATION / FINAL ANSWER 格式
2. 最终答案用 FINAL ANSWER: 开头
3. 搜索2次没结果就直接回答，不要继续搜"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]

    print(f"\n{'='*60}")
    print(f"用户: {user_query}")
    print(f"{'='*60}\n")

    no_tool_streak = 0      # 连续未调工具次数
    tools_were_used = False  # 上一步是否调了工具
    search_count = 0         # 搜索次数限制
    for step in range(1, max_steps + 1):
        print(f"--- Step {step}/{max_steps} ---")

        # (1) 调 LLM
        msg = call_llm(messages)
        content = msg.get("content", "") or ""
        if content.strip():
            print(f"[LLM思考] {content[:200]}")

        # (2) LLM 回复加入对话历史
        messages.append(msg)

        # (3) 检查 LLM 是否要调工具
        tool_calls = msg.get("tool_calls", [])

        if not tool_calls:
            # LLM 没调工具 -> 可能给出了最终答案
            if "FINAL ANSWER:" in content.upper():
                final = content.split("FINAL ANSWER:", 1)[1].strip()
                print(f"\n>>> 最终答案: {final}")
                return final
            # 如果上一步调用了工具、这一步没有调工具
            # 说明 LLM 正在根据工具结果给出回答 -> 返回这个回答
            if tools_were_used:
                print(f"\n>>> 最终答案: {content.strip()}")
                return content
            # 如果 LLM 两轮都没调工具 -> 可能是寒暄或空转 -> 也结束
            no_tool_streak += 1
            if no_tool_streak >= 2:
                print(f"\n(LLM 连续 {no_tool_streak} 轮未调工具，停止)")
                return content
            continue  # 继续下一轮

        # 记录这一轮调了工具，供下一轮判断
        tools_were_used = True

        # (4) 限制搜索次数（最多3次）
        for tc in tool_calls:
            if tc["function"]["name"] == "web_search":
                search_count += 1
        if search_count >= 4:
            # 搜索已超限，阻止搜索，让 LLM 用已有知识回答
            for tc in tool_calls:
                if tc["function"]["name"] == "web_search":
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": "搜索次数已达上限，请基于已有知识和已获取的信息回答"
                    })
            continue

        # (4) 执行工具
        for tc in tool_calls:
            name = tc["function"]["name"]
            args = tc["function"]["arguments"]
            print(f"[调工具] {name}({args})")

            result = execute_tool_call(tc)
            print(f"[工具返回] {result[:100]}")

            # 关键：工具结果作为 Observation 加回对话
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })

    print(f"\n(达到最大步骤 {max_steps}，停止)")
    print(f">>> 最终答案: {content.strip()}")
    return content

# ============================================================
# 运行测试
# ============================================================
if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) > 1:
        tests = [_sys.argv[1]]
    else:
        tests = [
        "先告诉我时间，再计算 100 / 7",
        "搜索一下2026年AI Agent的最新发展",
        "先搜索AI Agent的维基百科词条，打开第一条结果，然后总结内容（用中文）",
    ]
    for q in tests:
        react_loop(q)
        print("\n")
