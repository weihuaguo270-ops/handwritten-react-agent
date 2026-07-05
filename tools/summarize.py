"""自动提取文本摘要（抽取式）"""
import re


def summarize(text: str, max_sentences: int = 5) -> str:
    """自动提取文本摘要（抽取式：取前几个关键句子）"""
    if not text or len(text) < 20:
        return "文本过短，无需摘要"
    sentences = re.split(r'[。！？\n]', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if not sentences:
        return text[:500]
    summary = "。".join(sentences[:max_sentences]) + "。"
    if len(summary) > 1000:
        summary = summary[:1000] + "..."
    return summary


TOOL_DEFINITION = {
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
}
