"""
RAG 向量检索 — 替代手写 rag.py

基于 FAISS + HuggingFace Embeddings 的文档检索。
复用已有 rag_index.json 或从文件目录重新索引。
"""

import json
import os
import glob
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool

_EMBEDDINGS = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
_RAG_INDEX_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rag_index.json")

_store = None


def _get_store():
    global _store
    if _store is not None:
        return _store

    if os.path.exists(_RAG_INDEX_PATH):
        with open(_RAG_INDEX_PATH, encoding="utf-8") as f:
            data = json.load(f)
        chunks = data.get("chunks", [])
        if chunks:
            _store = FAISS.from_texts(chunks, _EMBEDDINGS)
            print(f"[RAG] 已加载 {len(chunks)} 个文档片段")
            return _store

    _store = FAISS.from_texts(["[占位]"], _EMBEDDINGS)
    return _store


def ingest(file_path: str) -> bool:
    """加载单个文件到向量库"""
    if not os.path.exists(file_path):
        return False
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    if not text.strip():
        return False

    store = _get_store()
    store.add_texts([text])
    return True


def ingest_directory(dir_path: str) -> int:
    """加载目录下所有支持的文档"""
    total = 0
    for ext in ["*.md", "*.py", "*.txt", "*.yaml", "*.yml"]:
        for f in sorted(glob.glob(os.path.join(dir_path, ext))):
            if ingest(f):
                total += 1
    print(f"[RAG] 目录加载完成，共 {total} 个文件")
    return total


# ============================================================
# 网页 RAG — 通过 MCP Server 检索网页内容
# ============================================================

import urllib.request
import urllib.parse
import re as _re

_WEB_RAG_ENABLED = False


def _enable_web_rag():
    """启用网页 RAG（检查依赖是否可用）"""
    global _WEB_RAG_ENABLED
    try:
        # 检查是否能访问网络
        urllib.request.urlopen("https://www.baidu.com", timeout=5)
        _WEB_RAG_ENABLED = True
    except Exception:
        _WEB_RAG_ENABLED = False


def _fetch_page_text(url: str) -> str:
    """抓取网页并提取纯文本"""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        return f""

    # 简单提取文本（去除 HTML 标签）
    text = _re.sub(r'<[^>]+>', ' ', html)
    text = _re.sub(r'\s+', ' ', text).strip()
    return text[:3000]  # 限制长度


def _search_and_fetch(query: str, top_k: int = 3) -> list:
    """搜索 + 抓取网页内容"""
    try:
        # 用 DuckDuckGo 搜索
        url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json"
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        results = []
        abstract = data.get("AbstractText", "")
        if abstract:
            results.append(("摘要", abstract))

        related = data.get("RelatedTopics", [])
        for r in related[:top_k]:
            if isinstance(r, dict):
                text = r.get("Text", "")
                url2 = r.get("FirstURL", "")
                if text:
                    # 尝试抓取网页内容
                    page_text = _fetch_page_text(url2) if url2 else ""
                    content = page_text if page_text else text
                    results.append((url2 or "相关结果", content[:2000]))

        return results
    except Exception as e:
        return []


@tool
def web_rag(query: str) -> str:
    """
    从互联网网页中检索与问题相关的详细内容。
    当用户需要获取网页上的具体信息、详细文档、最新资讯时使用。
    与 web_search 的区别：web_rag 会抓取网页内容返回完整片段，而 web_search 只返回搜索摘要。

    参数:
        query: 搜索关键词或问题
    """
    results = _search_and_fetch(query)
    if not results:
        return "未从互联网检索到相关信息"

    parts = ["以下是从互联网网页中检索到的相关信息：\n"]
    for i, (source, content) in enumerate(results, 1):
        parts.append(f"[{i}] 来源: {source}")
        parts.append(content)
        parts.append("---")
    return "\n".join(parts)


@tool
def rag_query(query: str, top_k: int = 3) -> str:
    """
    从本地文档库中检索与问题相关的知识。
    当用户问到产品文档、API 文档、项目知识库内容时使用。

    参数:
        query: 搜索关键词或问题
        top_k: 返回结果数量（默认3）
    """
    store = _get_store()
    docs = store.similarity_search(query, k=top_k)
    if not docs or (len(docs) == 1 and docs[0].page_content == "[占位]"):
        return "未在本地文档中找到相关信息"
    parts = ["以下是从文档中检索到的相关信息：\n"]
    for i, d in enumerate(docs, 1):
        parts.append(f"[{i}] {d.page_content[:500]}")
    return "\n\n".join(parts)
