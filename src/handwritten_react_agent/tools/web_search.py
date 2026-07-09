"""搜索互联网（基于 AnySearch MCP 协议）"""
import json
from urllib import request as req


def web_search(query: str, max_results: int = 1) -> str:
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


TOOL_DEFINITION = {
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
}
