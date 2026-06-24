
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
API_KEY=os.env...Y", "")
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

def tool_web_search(query: str, max_results: int = 5) -> str:
    """搜索维基百科获取信息（稳定、可靠、无需API Key）"""
    try:
        import json as _json

        # 先查中文维基，如果没有结果再查英文
        for lang, api_url in [("zh", "https://zh.wikipedia.org/w/api.php"),
                              ("en", "https://en.wikipedia.org/w/api.php")]:
            url = (f"{api_url}?action=query&list=search"
                   f"&srsearch={_parse.quote(query)}"
                   f"&format=json&srlimit={max_results}")
            r = _req.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with _req.urlopen(r, timeout=10) as resp:
                data = _json.loads(resp.read().decode("utf-8"))

            items = data.get("query", {}).get("search", [])
            if items:
                results = []
                for idx, item in enumerate(items[:max_results]):
                    title = item["title"]
                    snippet = re.sub(r'<[^>]+>', '', item["snippet"])
                    results.append(f"{idx+1}. {title}")
                    results.append(f"   {snippet.strip()[:200]}")
                return "\n".join(results)

        return "搜索未找到相关结果"

    except Exception as e:
        return f"搜索出错: {e}"
TOOL_REGISTRY = {
    "calculator": tool_calculator,
    "get_time": tool_get_time,
    "web_search": tool_web_search,
}

# 工具的 JSON 描述（发给 LLM 让它知道能调什么）
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "搜索互联网，获取实时信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "返回结果数量（默认5）"
                    }
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
    system_prompt = """你是一个可以使用工具的 AI 助手。
格式：
THOUGHT: 分析问题，是否需要工具
ACTION: 如果需要，调用工具
OBSERVATION: 工具返回了什么
THOUGHT: 继续推理
FINAL ANSWER: 最终答案

最终答案请用 FINAL ANSWER: 开头"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]

    print(f"\n{'='*60}")
    print(f"用户: {user_query}")
    print(f"{'='*60}\n")

    no_tool_streak = 0      # 连续未调工具次数
    tools_were_used = False  # 上一步是否调了工具
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
    tests = [
        "现在几点了？",
        "计算 (23 + 45) * 2 等于多少",
        "先告诉我时间，再计算 100 / 7",
        "搜索一下2026年AI Agent的最新发展",
    ]
    for q in tests:
        react_loop(q)
        print("\n")
