"""读取网页内容并提取正文（含 SSRF 防护）"""
import json
import re
from urllib import request as req
from urllib.parse import urlparse, quote


def fetch_page(url: str) -> str:
    """读取网页内容并提取正文"""
    try:
        # 验证 URL，防止 SSRF
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return f"不支持的协议: {parsed.scheme}"
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


TOOL_DEFINITION = {
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
}
